//===- TileIRToMLIR.cpp - Tile IR to MLIR conversion --------*- C++ -*-===//
//
// Part of the tileir-to-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Lowers Tile IR (the cuda_tile dialect) to the gpu or func, scf, arith, math,
// memref, vector and ub dialects. It is a strict dialect conversion: every
// cuda_tile op and type must be converted, or the pass fails.
//
// The type converter defines the value model:
//   - a tile becomes a vector, a rank-0 tile a scalar;
//   - a pointer becomes an unranked memref `memref<*xT>`, and a ranked pointer
//     tile becomes its base pointer plus a `vector<...xindex>` of lane
//     offsets, so accesses through it become gathers and scatters;
//   - tensor, partition and strided views become the strided memref of their
//     buffer; each access reads the tiling from the view type;
//   - tokens become nothing, as the lowered memory ops keep program order.
// Semantics that a target op cannot express are either rejected (e.g. non-weak
// ordering of loads) or recorded as `tir-dropped-*` attributes (see "Dropped
// semantics").
//
// The file holds shared helpers and pattern templates, the patterns roughly in
// alphabetical order of the cuda_tile op, the type converter and pattern list,
// and the pass, which applies the rewrites of PostConversion.h afterwards.
//
// Not lowered: AssertOp, AtomicCASTkoOp, AtomicRedViewTkoOp, BreakOp,
// IntToPtrOp, LoopOp, MmafScaledOp, PrintTkoOp, PtrToIntOp, accesses through a
// gather_scatter_view, and atomic_rmw_tko on pointer tiles. The
// --tileir-ptr-to-view pass raises the pointer accesses it recognizes to view
// accesses beforehand.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TileIRToMLIR/Passes.h"

#include "PostConversion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "llvm/ADT/TypeSwitch.h"

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTILEIRTOMLIRPASS
#include "mlir/Conversion/TileIRToMLIR/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// The ranked memref type of a tensor_view. Its offset is dynamic because the
/// view's base pointer may have been advanced, e.g. to a batch, and
/// reinterpret_cast offsets are absolute to the buffer.
static MemRefType tensorViewToMemRefType(cuda_tile::TensorViewType tvTy,
                                         const TypeConverter &converter) {
  auto shape = tvTy.getShape();
  auto strides = tvTy.getStrides();
  Type elemTy = converter.convertType(tvTy.getElementType());

  SmallVector<int64_t> memrefShape(shape.begin(), shape.end());
  SmallVector<int64_t> memrefStrides(strides.begin(), strides.end());
  auto layout = StridedLayoutAttr::get(
      elemTy.getContext(), /*offset=*/ShapedType::kDynamic, memrefStrides);
  return MemRefType::get(memrefShape, elemTy, layout);
}

/// `memref<size x T, strided<[1], offset: ?>>`: a 1-D view at a pointer whose
/// offset is known only at runtime.
static MemRefType get1DDynamicOffsetMemRefType(Type elemTy, int64_t size,
                                               Attribute memorySpace = {}) {
  return MemRefType::get({size}, elemTy,
                         StridedLayoutAttr::get(elemTy.getContext(),
                                                ShapedType::kDynamic,
                                                SmallVector<int64_t>{1}),
                         memorySpace);
}

/// The layout of an access through a partition_view or strided_view. Both
/// place tiles on a grid and differ only in how far the tile base advances per
/// index step (`viewStrides`): by the tile size for a partition_view, and by
/// the traversal stride for a strided_view, whose tiles may overlap or leave
/// gaps.
struct ViewInfo {
  Value memref;                   // Converted memref backing the view
  SmallVector<int64_t> tileShape; // Tile dimensions (per tile dim)
  SmallVector<int64_t>
      viewStrides;             // Base advance per index step (per tile dim)
  SmallVector<int32_t> dimMap; // Mapping from tile dims to tensor_view dims
  // Null if out-of-bounds loads yield unspecified values.
  cuda_tile::PaddingValueAttr paddingValue;
};

/// The layout of `view`, whose converted memref is `convertedView`. Fails for
/// a gather_scatter_view, whose sparse dimension needs a gather rather than a
/// transfer.
///
/// The view type verifiers guarantee that the tile and the tensor_view have
/// the same rank, that dim_map is a permutation, and that tile sizes and
/// strides are positive.
static FailureOr<ViewInfo> getViewInfo(Operation *op, Value view,
                                       Value convertedView,
                                       ConversionPatternRewriter &rewriter) {
  ViewInfo info;
  info.memref = convertedView;

  auto fill = [&](auto viewTy, ArrayRef<int32_t> advance) {
    ArrayRef<int32_t> tile = viewTy.getTileShape().asArrayRef();
    info.tileShape.assign(tile.begin(), tile.end());
    info.viewStrides.assign(advance.begin(), advance.end());
    info.dimMap.assign(viewTy.getDimMap().begin(), viewTy.getDimMap().end());
    info.paddingValue = viewTy.getPaddingValue();
  };

  if (auto pvType = dyn_cast<cuda_tile::PartitionViewType>(view.getType())) {
    fill(pvType, pvType.getTileShape().asArrayRef());
    return info;
  }

  if (auto svType = dyn_cast<cuda_tile::StridedViewType>(view.getType())) {
    fill(svType, svType.getTraversalStrides().asArrayRef());
    return info;
  }

  return rewriter.notifyMatchFailure(
      op, "view kind is not supported by the transfer-based lowering");
}

/// Map a cuda_tile rounding mode to the equivalent arith rounding mode, if a
/// direct 1:1 equivalent exists; otherwise std::nullopt.
static std::optional<arith::RoundingMode>
mapRoundingModeToArith(cuda_tile::RoundingMode rounding) {
  using CtRM = cuda_tile::RoundingMode;
  switch (rounding) {
  case CtRM::NEAREST_EVEN:
    return arith::RoundingMode::to_nearest_even;
  case CtRM::ZERO:
    return arith::RoundingMode::toward_zero;
  case CtRM::NEGATIVE_INF:
    return arith::RoundingMode::downward;
  case CtRM::POSITIVE_INF:
    return arith::RoundingMode::upward;
  case CtRM::APPROX:
  case CtRM::FULL:
  case CtRM::NEAREST_INT_TO_ZERO:
    return std::nullopt;
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Dropped semantics
//
// Source semantics that the lowered op cannot represent are recorded on it as
// discardable attributes. Semantics that the lowered op represents are never
// recorded, except for rounding modes under `drop-rounding-modes`.
//===----------------------------------------------------------------------===//

static constexpr StringLiteral kDroppedFlushToZero =
    "tir-dropped-flush-to-zero";
static constexpr StringLiteral kDroppedRounding = "tir-dropped-rounding";
static constexpr StringLiteral kDroppedOverflow = "tir-dropped-overflow";
static constexpr StringLiteral kDroppedOptimizationHints =
    "tir-dropped-optimization-hints";
static constexpr StringLiteral kDroppedMemoryOrdering =
    "tir-dropped-memory-ordering";
static constexpr StringLiteral kDroppedMemoryScope = "tir-dropped-memory-scope";

/// Record a set `flush_to_zero`, which has no arith/math equivalent.
static void preserveDroppedFlushToZero(OpBuilder &builder, bool flushToZero,
                                       Operation *newOp) {
  if (flushToZero)
    newOp->setAttr(kDroppedFlushToZero, builder.getUnitAttr());
}

/// Record a source rounding mode that `newOp` does not represent.
static void preserveDroppedRounding(OpBuilder &builder,
                                    cuda_tile::RoundingMode rounding,
                                    Operation *newOp) {
  newOp->setAttr(
      kDroppedRounding,
      builder.getStringAttr(cuda_tile::stringifyRoundingMode(rounding)));
}

/// Record integer-overflow flags that `newOp` does not represent. `none`
/// carries no information and is not recorded.
static void preserveDroppedOverflow(OpBuilder &builder,
                                    cuda_tile::IntegerOverflow overflow,
                                    Operation *newOp) {
  if (overflow == cuda_tile::IntegerOverflow::NONE)
    return;
  newOp->setAttr(
      kDroppedOverflow,
      builder.getStringAttr(cuda_tile::stringifyIntegerOverflow(overflow)));
}

/// Record the `optimization_hints` of `op`, which the lowered ops do not use.
template <typename OpT>
static void preserveDroppedOptHints(OpT op, Operation *newOp) {
  if (auto hints = op.getOptimizationHintsAttr())
    newOp->setAttr(kDroppedOptimizationHints, hints.getValue());
}

/// Map cuda_tile integer-overflow flags to arith integer-overflow flags.
static arith::IntegerOverflowFlags
mapIntegerOverflowFlags(cuda_tile::IntegerOverflow overflow) {
  using IO = cuda_tile::IntegerOverflow;
  switch (overflow) {
  case IO::NONE:
    return arith::IntegerOverflowFlags::none;
  case IO::NSW:
    return arith::IntegerOverflowFlags::nsw;
  case IO::NUW:
    return arith::IntegerOverflowFlags::nuw;
  case IO::NW:
    return arith::IntegerOverflowFlags::nsw | arith::IntegerOverflowFlags::nuw;
  }
  return arith::IntegerOverflowFlags::none;
}

/// Cast between `index` and an integer type with a sign-extending index_cast.
/// Returns null for other types.
static Value castValueToType(OpBuilder &builder, Location loc, Value value,
                             Type targetType) {
  if (value.getType() == targetType)
    return value;
  if ((isa<IndexType>(value.getType()) && isa<IntegerType>(targetType)) ||
      (isa<IntegerType>(value.getType()) && isa<IndexType>(targetType)))
    return arith::IndexCastOp::create(builder, loc, targetType, value);
  return Value();
}

/// Replace `op` with the signed or unsigned target op selected by `signedness`,
/// forwarding `args` to the target op builder.
template <typename SignedDstOp, typename UnsignedDstOp, typename SrcOp,
          typename... Args>
static Operation *replaceBySignedness(ConversionPatternRewriter &rewriter,
                                     SrcOp op, cuda_tile::Signedness signedness,
                                     Args &&...args) {
  if (signedness == cuda_tile::Signedness::Unsigned)
    return rewriter
        .template replaceOpWithNewOp<UnsignedDstOp>(
            op, std::forward<Args>(args)...)
        .getOperation();
  return rewriter
      .template replaceOpWithNewOp<SignedDstOp>(op, std::forward<Args>(args)...)
      .getOperation();
}

/// Convert the operation result type with the current type converter or emit a
/// match failure with `reason`.
template <typename OpT>
static FailureOr<Type>
getConvertedResultTypeOrFail(OpT op, const TypeConverter *converter,
                             ConversionPatternRewriter &rewriter,
                             StringRef reason) {
  Type resultTy = converter->convertType(op.getResult().getType());
  if (!resultTy) {
    (void)rewriter.notifyMatchFailure(op, reason);
    return failure();
  }
  return resultTy;
}

static SmallVector<int64_t> getReducedVectorShape(VectorType sourceType,
                                                  uint32_t dim) {
  SmallVector<int64_t> shape(sourceType.getShape().begin(),
                             sourceType.getShape().end());
  shape.erase(shape.begin() + dim);
  return shape;
}

/// Convert `value` to the semantics of `type`. Only used for the exact tf32 ->
/// f32 widening of the CPU target.
static APFloat convertFloat(APFloat value, FloatType type) {
  bool losesInfo = false;
  value.convert(type.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                &losesInfo);
  assert(!losesInfo && "expected an exact float conversion");
  return value;
}

/// `denseAttr` as elements of `newType`, which has as many elements and the
/// converted element type; floats are converted (tf32 -> f32 on the CPU).
static DenseElementsAttr retypeDenseElements(DenseElementsAttr denseAttr,
                                             ShapedType newType) {
  if (auto floatTy = dyn_cast<FloatType>(newType.getElementType());
      floatTy && floatTy != denseAttr.getElementType())
    denseAttr = cast<DenseFPElementsAttr>(denseAttr).mapValues(
        floatTy, [&](const APFloat &value) {
          return convertFloat(value, floatTy).bitcastToAPInt();
        });
  return denseAttr.reshape(newType);
}

/// The static memref type of a cuda_tile.global: the shape of its 1-D
/// initializer with the converted element type.
static FailureOr<MemRefType> getGlobalMemRefTypeOrFail(
    cuda_tile::GlobalOp globalOp, const TypeConverter &converter,
    ConversionPatternRewriter &rewriter, Operation *diagnosticOp) {
  auto initTy = dyn_cast<ShapedType>(globalOp.getValue().getType());
  if (!initTy || !initTy.hasStaticShape())
    return rewriter.notifyMatchFailure(
        diagnosticOp,
        "global initializer must be a statically shaped elements attribute");

  if (initTy.getRank() != 1)
    return rewriter.notifyMatchFailure(
        diagnosticOp,
        "global initializer must be 1-D to match cuda_tile.global semantics");

  return MemRefType::get(initTy.getShape(),
                         converter.convertType(initTy.getElementType()));
}

/// The indexing maps and iterator types of a vector.contract that computes a
/// matrix product.
struct MmaContractionSpec {
  AffineMap mapA;
  AffineMap mapB;
  AffineMap mapC;
  SmallVector<Attribute> iterTypes;
};

/// The contraction of an mmaf or mmai with a result of rank 2 ([M, N]) or
/// rank 3 (batched, [B, M, N]).
static FailureOr<MmaContractionSpec>
buildMmaContractionSpec(MLIRContext *ctx, int64_t resultRank) {
  if (resultRank != 2 && resultRank != 3)
    return failure();

  bool batched = (resultRank == 3);
  MmaContractionSpec spec;

  auto parAttr =
      vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel);
  auto redAttr =
      vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction);
  auto d0 = getAffineDimExpr(0, ctx);
  auto d1 = getAffineDimExpr(1, ctx);
  auto d2 = getAffineDimExpr(2, ctx);
  if (!batched) {
    spec.mapA = AffineMap::get(3, 0, {d0, d2}, ctx);
    spec.mapB = AffineMap::get(3, 0, {d2, d1}, ctx);
    spec.mapC = AffineMap::get(3, 0, {d0, d1}, ctx);
    spec.iterTypes = {parAttr, parAttr, redAttr};
  } else {
    auto d3 = getAffineDimExpr(3, ctx);
    spec.mapA = AffineMap::get(4, 0, {d0, d1, d3}, ctx);
    spec.mapB = AffineMap::get(4, 0, {d0, d3, d2}, ctx);
    spec.mapC = AffineMap::get(4, 0, {d0, d1, d2}, ctx);
    spec.iterTypes = {parAttr, parAttr, parAttr, redAttr};
  }
  return spec;
}

/// Convert `SrcOp` to `DstOp` with the converted result type and operands.
template <typename SrcOp, typename DstOp>
struct DirectConversion : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultTy =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "cannot convert result type");
    rewriter.template replaceOpWithNewOp<DstOp>(op, TypeRange{resultTy},
                                                adaptor.getOperands());
    return success();
  }
};

/// Replace `SrcOp` by its converted first operand, which the op only annotates
/// or views differently.
template <typename SrcOp>
struct ForwardOperand : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithMultiple(op, adaptor.getOperands().take_front());
    return success();
  }
};

/// Convert `SrcOp` to `SignedOp` or `UnsignedOp` according to its signedness,
/// with the converted result type and operands.
template <typename SrcOp, typename SignedOp, typename UnsignedOp>
struct SignednessConversion : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultTy =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "cannot convert result type");
    replaceBySignedness<SignedOp, UnsignedOp>(rewriter, op, op.getSignedness(),
                                              TypeRange{resultTy},
                                              adaptor.getOperands());
    return success();
  }
};

/// Base of the patterns that depend on the pass options.
template <typename SrcOp, typename Base = OpConversionPattern<SrcOp>>
struct OptionsPattern : public Base {
  OptionsPattern(const TypeConverter &converter, MLIRContext *ctx,
                 const ConvertTileIRToMLIRPassOptions &options)
      : Base(converter, ctx), options(options) {}

  ConvertTileIRToMLIRPassOptions options;
};

/// Convert a unary op with `flush_to_zero` to a math op, recording a set
/// `flush_to_zero` as dropped.
template <typename SrcOp, typename DstOp>
struct ConvertUnaryFlushToZeroOp : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp =
        rewriter.template replaceOpWithNewOp<DstOp>(op, adaptor.getSource());
    preserveDroppedFlushToZero(rewriter, op.getFlushToZero(), newOp);
    return success();
  }
};

/// Convert a unary op with a rounding mode to a math op. `Exact` is the
/// rounding the math op implements, and `rounding<approx>` maps to the `afn`
/// (allow approximate functions) flag. Other modes, and all modes under
/// `drop-rounding-modes`, are recorded as dropped, as is a set `flush_to_zero`
/// when `PreserveFtz` is set.
template <typename SrcOp, typename DstOp, cuda_tile::RoundingMode Exact,
          bool PreserveFtz>
struct ConvertUnaryApproxMathOp : public OptionsPattern<SrcOp> {
  using OptionsPattern<SrcOp>::OptionsPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    cuda_tile::RoundingMode rounding = op.getRoundingMode();
    bool keepRounding = !this->options.dropRoundingModes;
    bool approx = keepRounding && rounding == cuda_tile::RoundingMode::APPROX;
    auto newOp = rewriter.template replaceOpWithNewOp<DstOp>(
        op, adaptor.getSource(),
        arith::FastMathFlagsAttr::get(rewriter.getContext(),
                                      approx ? arith::FastMathFlags::afn
                                             : arith::FastMathFlags::none));
    if (!approx && !(keepRounding && rounding == Exact))
      preserveDroppedRounding(rewriter, rounding, newOp);
    if constexpr (PreserveFtz)
      preserveDroppedFlushToZero(rewriter, op.getFlushToZero(), newOp);
    return success();
  }
};

/// Convert a float binary op to its arith op.
///
/// arith computes `rounding<nearest_even>` by default and the directed modes
/// with its `roundingmode`. `ApproxFlag`, unless `none`, is the fastmath flag
/// that expresses `rounding<approx>`. Other modes, all modes under
/// `drop-rounding-modes`, and a set `flush_to_zero` are recorded as dropped.
template <typename SrcOp, typename DstOp,
          arith::FastMathFlags ApproxFlag = arith::FastMathFlags::none>
struct ConvertBinaryFloatOp : public OptionsPattern<SrcOp> {
  using OptionsPattern<SrcOp>::OptionsPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rounding = op.getRoundingMode();
    bool ftz = op.getFlushToZero();
    arith::FastMathFlags fmf = arith::FastMathFlags::none;
    arith::RoundingModeAttr roundingAttr;
    bool roundingRepresented = false;
    if (!this->options.dropRoundingModes) {
      if (rounding == cuda_tile::RoundingMode::NEAREST_EVEN) {
        roundingRepresented = true;
      } else if (ApproxFlag != arith::FastMathFlags::none &&
                 rounding == cuda_tile::RoundingMode::APPROX) {
        fmf = ApproxFlag;
        roundingRepresented = true;
      } else if (std::optional<arith::RoundingMode> mode =
                     mapRoundingModeToArith(rounding)) {
        roundingAttr =
            arith::RoundingModeAttr::get(rewriter.getContext(), *mode);
        roundingRepresented = true;
      }
    }
    auto newOp = rewriter.template replaceOpWithNewOp<DstOp>(
        op, adaptor.getLhs(), adaptor.getRhs(),
        arith::FastMathFlagsAttr::get(rewriter.getContext(), fmf),
        roundingAttr);
    if (!roundingRepresented)
      preserveDroppedRounding(rewriter, rounding, newOp);
    preserveDroppedFlushToZero(rewriter, ftz, newOp);
    return success();
  }
};

/// Convert an integer binary op with overflow flags (addi, subi, muli, shli).
template <typename SrcOp, typename DstOp>
struct ConvertBinaryLhsRhsWithOverflowOp : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto overflowAttr = arith::IntegerOverflowFlagsAttr::get(
        rewriter.getContext(), mapIntegerOverflowFlags(op.getOverflow()));
    rewriter.template replaceOpWithNewOp<DstOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs(), overflowAttr);
    return success();
  }
};

/// Convert ftoi or itof to the signed or unsigned arith cast, which rounds as
/// `ExpectedRounding`, the only mode the op verifier accepts.
template <typename SrcOp, typename SignedDstOp, typename UnsignedDstOp,
          cuda_tile::RoundingMode ExpectedRounding>
struct ConvertFromToSignednessCastWithRoundingOp
    : public OptionsPattern<SrcOp> {
  using OptionsPattern<SrcOp>::OptionsPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool roundingRepresented = !this->options.dropRoundingModes &&
                               op.getRoundingMode() == ExpectedRounding;

    auto resultTy =
        getConvertedResultTypeOrFail(op, this->getTypeConverter(), rewriter,
                                     "cannot convert cast result type");
    if (failed(resultTy))
      return failure();

    Operation *newOp = replaceBySignedness<SignedDstOp, UnsignedDstOp>(
        rewriter, op, op.getSignedness(), resultTy.value(), adaptor.getFrom());
    if (!roundingRepresented)
      preserveDroppedRounding(rewriter, op.getRoundingMode(), newOp);
    return success();
  }
};

/// Flatten the 1:N adaptor operands of a pattern into a single value list.
static SmallVector<Value> flattenValues(ArrayRef<ValueRange> values) {
  SmallVector<Value> flat;
  for (ValueRange range : values)
    llvm::append_range(flat, range);
  return flat;
}

/// Split `newValues` into one range per type of `origTypes`, sized by what the
/// type converts to: none for a token, two for a pointer tile.
static SmallVector<ValueRange>
groupByConvertedTypes(const TypeConverter &converter, TypeRange origTypes,
                      ValueRange newValues) {
  SmallVector<ValueRange> groups;
  for (Type type : origTypes) {
    SmallVector<Type> convertedTypes;
    (void)converter.convertType(type, convertedTypes);
    groups.push_back(newValues.take_front(convertedTypes.size()));
    newValues = newValues.drop_front(convertedTypes.size());
  }
  return groups;
}

/// Base for patterns of ops with token operands. Tokens convert to no values,
/// so they are null in the 1:1 adaptor; other 1:N operands fail the match.
template <typename SourceOp>
struct TokenDroppingPattern : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<SourceOp>::OpAdaptor;
  using OneToNOpAdaptor =
      typename OpConversionPattern<SourceOp>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Value> operands;
    for (ValueRange values : adaptor.getOperands()) {
      if (values.size() > 1)
        return rewriter.notifyMatchFailure(op, "unexpected 1:N operand");
      operands.push_back(values.empty() ? Value() : values.front());
    }
    const OpConversionPattern<SourceOp> &self = *this;
    return self.matchAndRewrite(op, OpAdaptor(operands, adaptor), rewriter);
  }
};

/// Erase token-producing ops (make_token / join_tokens).
template <typename SrcOp>
struct EraseTokenOp : public TokenDroppingPattern<SrcOp> {
  using TokenDroppingPattern<SrcOp>::TokenDroppingPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename TokenDroppingPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert cuda_tile terminators (continue / yield) to scf.yield.
template <typename SrcOp>
struct ConvertToScfYield : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.template replaceOpWithNewOp<scf::YieldOp>(
        op, flattenValues(adaptor.getOperands()));
    return success();
  }
};

/// The six launch coordinates that `append-grid-args` appends to the arguments
/// of entry functions: tile block id x/y/z, then grid dimension x/y/z.
struct AppendedGridArgLayout {
  static constexpr unsigned kNumArgs = 6;
  static constexpr unsigned kBlockIdBase = 0;
  static constexpr unsigned kGridDimBase = 3;

  /// The argument index of coordinate `dim` of the triple at `argBase`, in a
  /// function with `numArgs` arguments.
  static unsigned argIndex(unsigned numArgs, unsigned argBase,
                           gpu::Dimension dim) {
    unsigned startIdx = numArgs - kNumArgs;
    switch (dim) {
    case gpu::Dimension::x:
      return startIdx + argBase + 0;
    case gpu::Dimension::y:
      return startIdx + argBase + 1;
    case gpu::Dimension::z:
      return startIdx + argBase + 2;
    }
    llvm_unreachable("unhandled gpu dimension");
  }
};

/// Convert a query of three grid coordinates (get_tile_block_id,
/// get_num_tile_blocks) to the appended function arguments at `ArgBase` under
/// `append-grid-args`, and to `GpuDimOp` ops on the GPU target otherwise.
template <typename SrcOp, typename GpuDimOp, unsigned ArgBase>
struct ConvertDimQueryOp : public OptionsPattern<SrcOp> {
  using OptionsPattern<SrcOp>::OptionsPattern;

  LogicalResult
  matchAndRewrite(SrcOp op, typename OpConversionPattern<SrcOp>::OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultTy =
        this->getTypeConverter()->convertType(op.getResult(0).getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "cannot convert result type");

    FunctionOpInterface parentFunc;
    if (this->options.appendGridArgs) {
      parentFunc = op->template getParentOfType<FunctionOpInterface>();
      if (!parentFunc ||
          parentFunc.getNumArguments() < AppendedGridArgLayout::kNumArgs)
        return rewriter.notifyMatchFailure(
            op, "expected enclosing function with appended launch-coordinate "
                "arguments");
    } else if (this->options.target != TileIRTarget::GPU) {
      return rewriter.notifyMatchFailure(
          op, "dim-query lowering on non-GPU targets requires "
              "append-grid-args=true");
    }

    SmallVector<Value, 3> results;
    for (gpu::Dimension dim :
         {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z}) {
      Value raw =
          parentFunc
              ? Value(parentFunc.getArgument(AppendedGridArgLayout::argIndex(
                    parentFunc.getNumArguments(), ArgBase, dim)))
              : Value(GpuDimOp::create(rewriter, loc, dim));
      Value casted = castValueToType(rewriter, loc, raw, resultTy);
      if (!casted)
        return rewriter.notifyMatchFailure(
            op, "cannot cast dim query result to target type");
      results.push_back(casted);
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

/// Convert maxf or minf to the arith op that propagates NaNs (maximumf,
/// minimumf) or not (maxnumf, minnumf), as `propagate_nan` requests.
template <typename SrcOp, typename NanPropagatingOp, typename NanSuppressingOp>
struct ConvertMinMaxFOp : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool ftz = op.getFlushToZero();
    Operation *newOp = op.getPropagateNan()
                           ? rewriter
                                 .template replaceOpWithNewOp<NanPropagatingOp>(
                                     op, adaptor.getLhs(), adaptor.getRhs())
                                 .getOperation()
                           : rewriter
                                 .template replaceOpWithNewOp<NanSuppressingOp>(
                                     op, adaptor.getLhs(), adaptor.getRhs())
                                 .getOperation();
    preserveDroppedFlushToZero(rewriter, ftz, newOp);
    return success();
  }
};

/// Match a single-operand reduce or scan whose body applies one binary op to
/// its two arguments. Returns the vector combining kind of that op, the
/// converted operand and its type, and the identity.
template <typename OpT>
static FailureOr<
    std::tuple<vector::CombiningKind, Value, VectorType, TypedAttr>>
matchSingleOperandCombiningOp(OpT op, ValueRange convertedOperands,
                              ConversionPatternRewriter &rewriter) {
  if (op.getOperands().size() != 1)
    return rewriter.notifyMatchFailure(
        op, "multi-operand reductions are not supported");

  auto bodyFailure = [&]() -> LogicalResult {
    return rewriter.notifyMatchFailure(
        op, "cannot determine combining kind from body");
  };

  Block &block = op.getBody().front();
  if (block.getNumArguments() != 2)
    return bodyFailure();

  auto yieldOp = dyn_cast<cuda_tile::YieldOp>(block.getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return bodyFailure();

  Operation *combiningOp = nullptr;
  for (Operation &bodyOp : block.without_terminator()) {
    if (combiningOp)
      return bodyFailure();
    combiningOp = &bodyOp;
  }
  if (!combiningOp)
    return bodyFailure();

  if (combiningOp->getNumOperands() != 2 || combiningOp->getNumResults() != 1 ||
      yieldOp.getOperand(0) != combiningOp->getResult(0))
    return bodyFailure();

  auto lhsArg = dyn_cast<BlockArgument>(combiningOp->getOperand(0));
  auto rhsArg = dyn_cast<BlockArgument>(combiningOp->getOperand(1));
  if (!lhsArg || !rhsArg || lhsArg.getOwner() != &block ||
      rhsArg.getOwner() != &block ||
      lhsArg.getArgNumber() == rhsArg.getArgNumber())
    return bodyFailure();

  auto kind =
      llvm::TypeSwitch<Operation *, FailureOr<vector::CombiningKind>>(
          combiningOp)
          .template Case<cuda_tile::AddFOp, cuda_tile::AddIOp>(
              [](auto) { return vector::CombiningKind::ADD; })
          .template Case<cuda_tile::MulFOp, cuda_tile::MulIOp>(
              [](auto) { return vector::CombiningKind::MUL; })
          .template Case<cuda_tile::MaxFOp>([](cuda_tile::MaxFOp bodyOp) {
            return bodyOp.getPropagateNan() ? vector::CombiningKind::MAXIMUMF
                                            : vector::CombiningKind::MAXNUMF;
          })
          .template Case<cuda_tile::MinFOp>([](cuda_tile::MinFOp bodyOp) {
            return bodyOp.getPropagateNan() ? vector::CombiningKind::MINIMUMF
                                            : vector::CombiningKind::MINNUMF;
          })
          .template Case<cuda_tile::MaxIOp>([](cuda_tile::MaxIOp bodyOp) {
            return bodyOp.getSignedness() == cuda_tile::Signedness::Unsigned
                       ? vector::CombiningKind::MAXUI
                       : vector::CombiningKind::MAXSI;
          })
          .template Case<cuda_tile::MinIOp>([](cuda_tile::MinIOp bodyOp) {
            return bodyOp.getSignedness() == cuda_tile::Signedness::Unsigned
                       ? vector::CombiningKind::MINUI
                       : vector::CombiningKind::MINSI;
          })
          .template Case<cuda_tile::AndIOp>(
              [](auto) { return vector::CombiningKind::AND; })
          .template Case<cuda_tile::OrIOp>(
              [](auto) { return vector::CombiningKind::OR; })
          .template Case<cuda_tile::XOrIOp>(
              [](auto) { return vector::CombiningKind::XOR; })
          .Default([](Operation *) { return failure(); });
  if (failed(kind))
    return bodyFailure();

  Value source = convertedOperands.front();
  auto srcVecTy = dyn_cast<VectorType>(source.getType());
  if (!srcVecTy)
    return rewriter.notifyMatchFailure(op, "source is not a vector");

  if (srcVecTy.getRank() == 0)
    return rewriter.notifyMatchFailure(op,
                                       "source vector must have positive rank");

  if (op.getDim() >= static_cast<uint32_t>(srcVecTy.getRank()))
    return rewriter.notifyMatchFailure(op,
                                       "reduction dimension is out of bounds");

  if (op.getIdentities().size() != 1)
    return rewriter.notifyMatchFailure(op,
                                       "requires exactly one identity value");

  auto identityAttr = dyn_cast<TypedAttr>(op.getIdentities()[0]);
  if (!identityAttr)
    return rewriter.notifyMatchFailure(op, "identity is not a typed attribute");

  if (identityAttr.getType() != srcVecTy.getElementType())
    return rewriter.notifyMatchFailure(
        op, "identity type does not match the source element type");

  return std::make_tuple(*kind, source, srcVecTy, identityAttr);
}

/// Reject load/store_tko orderings and scopes that the lowering would weaken.
///
/// They lower to plain memref and vector accesses without ordering, so only
/// `weak` without a scope is accepted. atomic_rmw_tko differs: its
/// memref.atomic_rmw is acq_rel, at least as strong as any ordering, so it
/// records the ordering as dropped instead.
template <typename TkoOp>
static LogicalResult checkCommonTkoGuards(TkoOp op,
                                          ConversionPatternRewriter &rewriter) {
  if (op.getMemoryOrderingSemantics() !=
      cuda_tile::MemoryOrderingSemantics::WEAK)
    return rewriter.notifyMatchFailure(
        op, "only `weak` memory_ordering_semantics is supported");
  if (op.getMemoryScope())
    return rewriter.notifyMatchFailure(
        op, "memory_scope is not supported by this lowering");
  return success();
}

/// The operands of the vector.transfer_read/write for a view access.
struct TransferViewAccessPlan {
  ViewInfo viewInfo;
  SmallVector<Value> memrefIndices;
  AffineMap permutationMap;
  SmallVector<bool> inBounds;
};

/// Plan the transfer that accesses tile `convertedIndices` of `view`.
///
/// Tile index i, scaled by the base advance of the view, indexes memref
/// dimension dim_map[i], and the permutation map sends that dimension to
/// vector dimension i. A dimension is in bounds only if its extent is static
/// and the last tile that starts inside the tensor also ends inside it; the
/// transfer masks the other dimensions.
static FailureOr<TransferViewAccessPlan>
buildTransferViewAccessPlan(ConversionPatternRewriter &rewriter, Operation *op,
                            Value view, Value convertedView,
                            ValueRange convertedIndices) {
  auto viewInfoOr = getViewInfo(op, view, convertedView, rewriter);
  if (failed(viewInfoOr))
    return failure();
  ViewInfo viewInfo = std::move(*viewInfoOr);

  // The op verifiers guarantee one index per tile dimension.
  unsigned rank = viewInfo.tileShape.size();
  Location loc = op->getLoc();
  auto *ctx = rewriter.getContext();

  // View indices are unsigned.
  auto nswFlag = arith::IntegerOverflowFlagsAttr::get(
      ctx, arith::IntegerOverflowFlags::nsw);
  SmallVector<Value> memrefIndices(rank);
  for (unsigned i = 0; i < rank; ++i) {
    Value tileIndex = arith::IndexCastUIOp::create(
        rewriter, loc, rewriter.getIndexType(), convertedIndices[i]);
    Value strideVal =
        arith::ConstantIndexOp::create(rewriter, loc, viewInfo.viewStrides[i]);
    memrefIndices[viewInfo.dimMap[i]] =
        arith::MulIOp::create(rewriter, loc, tileIndex, strideVal, nswFlag);
  }

  SmallVector<AffineExpr> permExprs;
  permExprs.reserve(rank);
  for (int32_t td : viewInfo.dimMap)
    permExprs.push_back(getAffineDimExpr(td, ctx));
  auto permutationMap = AffineMap::get(rank, 0, permExprs, ctx);

  auto memrefTy = cast<MemRefType>(viewInfo.memref.getType());
  auto memrefShape = memrefTy.getShape();
  SmallVector<bool> inBounds(rank, false);
  for (unsigned i = 0; i < rank; ++i) {
    int64_t ext = memrefShape[viewInfo.dimMap[i]];
    if (ext == ShapedType::kDynamic)
      continue;
    // Count the tiles that start inside the tensor, then check whether the
    // last one ends inside it.
    int64_t stride = viewInfo.viewStrides[i];
    int64_t numTiles = (ext + stride - 1) / stride;
    int64_t lastBase = numTiles > 0 ? (numTiles - 1) * stride : 0;
    inBounds[i] = (lastBase + viewInfo.tileShape[i] <= ext);
  }

  return TransferViewAccessPlan{std::move(viewInfo), std::move(memrefIndices),
                                permutationMap, std::move(inBounds)};
}

//===----------------------------------------------------------------------===//
// Conversion Patterns, roughly in alphabetical order of the cuda_tile op
//===----------------------------------------------------------------------===//
using ConvertAddF = ConvertBinaryFloatOp<cuda_tile::AddFOp, arith::AddFOp>;

using ConvertAddI =
    ConvertBinaryLhsRhsWithOverflowOp<cuda_tile::AddIOp, arith::AddIOp>;

/// Convert cuda_tile.alloca to a memref.alloca of `num_elem` elements, cast to
/// the pointer type `memref<*xT>`. The `global` variant, whose address is
/// shared across tile threads, is rejected: the memref has no memory space
/// that could express the sharing.
struct ConvertAlloca : public OpConversionPattern<cuda_tile::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::AllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getGlobal())
      return rewriter.notifyMatchFailure(
          op, "global (cross-thread shareable) alloca has no equivalent in the "
              "unranked memref pointer model");

    auto resultTy = getConvertedResultTypeOrFail(
        op, getTypeConverter(), rewriter, "cannot convert alloca result type");
    if (failed(resultTy))
      return failure();
    auto unrankedTy = dyn_cast<UnrankedMemRefType>(resultTy.value());
    if (!unrankedTy)
      return rewriter.notifyMatchFailure(
          op, "alloca result did not convert to an unranked memref");

    // The verifier guarantees the non-zero power of two memref.alloca needs.
    auto rankedTy = MemRefType::get(
        {static_cast<int64_t>(op.getNumElem())}, unrankedTy.getElementType(),
        MemRefLayoutAttrInterface{}, unrankedTy.getMemorySpace());
    Value alloca =
        memref::AllocaOp::create(rewriter, op.getLoc(), rankedTy,
                                 rewriter.getI64IntegerAttr(op.getAlignment()));
    rewriter.replaceOpWithNewOp<memref::CastOp>(op, unrankedTy, alloca);
    return success();
  }
};

/// Convert cuda_tile.broadcast to vector.broadcast.
///
/// Both ops stretch size-1 dimensions without changing the rank. A pointer
/// tile keeps its base and broadcasts its offsets.
struct ConvertBroadcast : public OpConversionPattern<cuda_tile::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::BroadcastOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertType(op.getType(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "cannot convert result type");

    SmallVector<Value> results;
    for (auto [source, resultTy] :
         llvm::zip_equal(adaptor.getSource(), resultTypes)) {
      if (auto vectorTy = dyn_cast<VectorType>(resultTy))
        results.push_back(vector::BroadcastOp::create(rewriter, op.getLoc(),
                                                      vectorTy, source));
      else
        results.push_back(source);
    }
    rewriter.replaceOpWithMultiple(op, {results});
    return success();
  }
};

/// Convert cuda_tile.cat to two vector.insert_strided_slice ops into a poison
/// vector: lhs at offset 0, and rhs behind it along the concatenated dimension.
struct ConvertCat : public OpConversionPattern<cuda_tile::CatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "cannot convert result type");

    auto dstVecTy = cast<VectorType>(resultTy);
    int64_t rank = dstVecTy.getRank();
    int64_t concatDim = op.getDim();
    Location loc = op.getLoc();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto lhsVecTy = cast<VectorType>(lhs.getType());

    Value dest = ub::PoisonOp::create(rewriter, loc, dstVecTy);
    SmallVector<int64_t> lhsOffsets(rank, 0);
    SmallVector<int64_t> strides(rank, 1);
    Value withLhs = vector::InsertStridedSliceOp::create(
        rewriter, loc, lhs, dest, lhsOffsets, strides);

    SmallVector<int64_t> rhsOffsets(rank, 0);
    rhsOffsets[concatDim] = lhsVecTy.getDimSize(concatDim);

    rewriter.replaceOpWithNewOp<vector::InsertStridedSliceOp>(
        op, rhs, withLhs, rhsOffsets, strides);
    return success();
  }
};

/// Map a cuda_tile comparison predicate to the arith.cmpf predicate that is
/// ordered or unordered as requested.
static arith::CmpFPredicate
mapCmpFPredicate(cuda_tile::ComparisonPredicate pred, bool ordered) {
  using CP = cuda_tile::ComparisonPredicate;
  using AP = arith::CmpFPredicate;
  switch (pred) {
  case CP::EQUAL:
    return ordered ? AP::OEQ : AP::UEQ;
  case CP::NOT_EQUAL:
    return ordered ? AP::ONE : AP::UNE;
  case CP::LESS_THAN:
    return ordered ? AP::OLT : AP::ULT;
  case CP::LESS_THAN_OR_EQUAL:
    return ordered ? AP::OLE : AP::ULE;
  case CP::GREATER_THAN:
    return ordered ? AP::OGT : AP::UGT;
  case CP::GREATER_THAN_OR_EQUAL:
    return ordered ? AP::OGE : AP::UGE;
  }
  llvm_unreachable("unhandled cuda_tile comparison predicate");
}

/// Map a cuda_tile comparison predicate to the arith.cmpi predicate with the
/// requested signedness. Equality predicates are signedness-agnostic.
static arith::CmpIPredicate
mapCmpIPredicate(cuda_tile::ComparisonPredicate pred, bool isUnsigned) {
  using CP = cuda_tile::ComparisonPredicate;
  using AP = arith::CmpIPredicate;
  switch (pred) {
  case CP::EQUAL:
    return AP::eq;
  case CP::NOT_EQUAL:
    return AP::ne;
  case CP::LESS_THAN:
    return isUnsigned ? AP::ult : AP::slt;
  case CP::LESS_THAN_OR_EQUAL:
    return isUnsigned ? AP::ule : AP::sle;
  case CP::GREATER_THAN:
    return isUnsigned ? AP::ugt : AP::sgt;
  case CP::GREATER_THAN_OR_EQUAL:
    return isUnsigned ? AP::uge : AP::sge;
  }
  llvm_unreachable("unhandled cuda_tile comparison predicate");
}

/// Convert cuda_tile.cmpf to arith.cmpf.
struct ConvertCmpF : public OpConversionPattern<cuda_tile::CmpFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool ordered = op.getComparisonOrdering() ==
                   cuda_tile::ComparisonOrdering::ORDERED;
    rewriter.replaceOpWithNewOp<arith::CmpFOp>(
        op, mapCmpFPredicate(op.getComparisonPredicate(), ordered),
        adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

/// Convert cuda_tile.cmpi to arith.cmpi.
struct ConvertCmpI : public OpConversionPattern<cuda_tile::CmpIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool isUnsigned = op.getSignedness() == cuda_tile::Signedness::Unsigned;
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(
        op, mapCmpIPredicate(op.getComparisonPredicate(), isUnsigned),
        adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

/// Convert cuda_tile.constant to an arith.constant of the converted type.
struct ConvertConstant : public OpConversionPattern<cuda_tile::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType ||
        !(isa<VectorType>(resultType) || resultType.isIntOrFloat()))
      return rewriter.notifyMatchFailure(op, "unsupported constant type");

    auto shapedTy = dyn_cast<VectorType>(resultType);
    DenseElementsAttr value = retypeDenseElements(
        op.getValue(), shapedTy ? ShapedType(shapedTy)
                                : RankedTensorType::get({}, resultType));
    TypedAttr attr = shapedTy
                         ? TypedAttr(value)
                         : cast<TypedAttr>(value.getSplatValue<Attribute>());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, attr);
    return success();
  }
};

using ConvertContinue = ConvertToScfYield<cuda_tile::ContinueOp>;

/// Convert cuda_tile.divf to arith.divf; `rounding<approx>` maps to the `arcp`
/// (allow reciprocal) flag.
using ConvertDivF = ConvertBinaryFloatOp<cuda_tile::DivFOp, arith::DivFOp,
                                         arith::FastMathFlags::arcp>;

/// Convert cuda_tile.divi to the arith division with the same rounding:
/// zero -> divsi/divui, positive_inf -> ceildivsi/ceildivui, negative_inf ->
/// floordivsi (unsigned floor division is divui).
///
/// The rounding mode defines the integer result, so it is never dropped, not
/// even under `drop-rounding-modes`.
struct ConvertDivI : public OpConversionPattern<cuda_tile::DivIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::DivIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    cuda_tile::Signedness signedness = op.getSignedness();
    switch (op.getRounding()) {
    case cuda_tile::RoundingMode::ZERO:
      replaceBySignedness<arith::DivSIOp, arith::DivUIOp>(rewriter, op,
                                                          signedness, lhs, rhs);
      return success();
    case cuda_tile::RoundingMode::POSITIVE_INF:
      replaceBySignedness<arith::CeilDivSIOp, arith::CeilDivUIOp>(
          rewriter, op, signedness, lhs, rhs);
      return success();
    case cuda_tile::RoundingMode::NEGATIVE_INF:
      replaceBySignedness<arith::FloorDivSIOp, arith::DivUIOp>(
          rewriter, op, signedness, lhs, rhs);
      return success();
    default:
      return rewriter.notifyMatchFailure(op, "unsupported divi rounding mode");
    }
  }
};

/// Convert cuda_tile.entry to a gpu.func kernel (GPU target) or a func.func
/// (CPU target) with the converted argument types. `append-grid-args` appends
/// the launch coordinates (see AppendedGridArgLayout), and `known-block-size`
/// sets the `known_block_size` of the gpu.func.
struct ConvertEntry : public OptionsPattern<cuda_tile::EntryOp> {
  using OptionsPattern::OptionsPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::EntryOp entryOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = entryOp.getContext();
    Location loc = entryOp.getLoc();
    Block *entryBlock = &entryOp.getBody().front();
    unsigned numArgs = entryBlock->getNumArguments();

    const TypeConverter *tc = getTypeConverter();
    TypeConverter::SignatureConversion sigConv(numArgs);
    SmallVector<Type> funcArgTypes;

    funcArgTypes.reserve(numArgs + AppendedGridArgLayout::kNumArgs);
    for (unsigned i = 0; i < numArgs; ++i) {
      Type origTy = entryBlock->getArgument(i).getType();
      Type converted = tc->convertType(origTy);
      if (!converted)
        return rewriter.notifyMatchFailure(entryOp,
                                           "cannot convert entry arg type");
      funcArgTypes.push_back(converted);
      sigConv.addInputs(i, converted);
    }

    if (options.appendGridArgs) {
      SmallVector<Type> launchArgTypes(AppendedGridArgLayout::kNumArgs,
                                       IntegerType::get(ctx, 32));
      funcArgTypes.append(launchArgTypes.begin(), launchArgTypes.end());
      sigConv.addInputs(launchArgTypes);
    }

    auto funcType = FunctionType::get(ctx, funcArgTypes, {});

    FailureOr<Block *> convertedBlock =
        rewriter.convertRegionTypes(&entryOp.getBody(), *tc, &sigConv);
    if (failed(convertedBlock))
      return failure();

    if (options.target == TileIRTarget::GPU) {
      // gpu.func creates its entry block; merge the converted body into it.
      auto gpuFunc =
          gpu::GPUFuncOp::create(rewriter, loc, entryOp.getSymName(), funcType);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       rewriter.getUnitAttr());
      if (!options.knownBlockSize.empty())
        gpuFunc.setKnownBlockSizeAttr(
            rewriter.getDenseI32ArrayAttr(options.knownBlockSize));
      preserveDroppedOptHints(entryOp, gpuFunc);
      Block *gpuBlock = &gpuFunc.getBody().front();
      rewriter.mergeBlocks(*convertedBlock, gpuBlock, gpuBlock->getArguments());
    } else {
      auto func =
          func::FuncOp::create(rewriter, loc, entryOp.getSymName(), funcType);
      preserveDroppedOptHints(entryOp, func);
      rewriter.inlineRegionBefore(entryOp.getBody(), func.getBody(),
                                  func.getBody().end());
    }
    rewriter.eraseOp(entryOp);
    return success();
  }
};

/// Convert cuda_tile.exp to math.exp, which has full precision.
using ConvertExp = ConvertUnaryApproxMathOp<cuda_tile::ExpOp, math::ExpOp,
                                            cuda_tile::RoundingMode::FULL,
                                            /*PreserveFtz=*/false>;

/// Convert cuda_tile.exp2 to math.exp2.
using ConvertExp2 = ConvertUnaryFlushToZeroOp<cuda_tile::Exp2Op, math::Exp2Op>;

/// Convert cuda_tile.extract, which returns slice [i_0, ..., i_{n-1}] of a
/// source of shape D cut into slices of the result shape R. The indices are
/// dynamic, which vector.extract_strided_slice does not support, so:
///   1. shape_cast the source to <S_0 x R_0 x ... x S_{n-1} x R_{n-1}>, where
///      S_k = D_k / R_k, which keeps the row-major element order;
///   2. transpose the slice dimensions S_k to the front (a no-op for rank 1);
///   3. vector.extract the slice at the indices.
struct ConvertExtract : public OpConversionPattern<cuda_tile::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "cannot convert result type");

    Value source = adaptor.getSource();
    Location loc = op.getLoc();

    // The only slice is the source, e.g. of a rank-0 tile.
    if (source.getType() == resultTy) {
      rewriter.replaceOp(op, source);
      return success();
    }

    auto srcVecTy = cast<VectorType>(source.getType());
    auto dstVecTy = cast<VectorType>(resultTy);
    int64_t rank = srcVecTy.getRank();

    SmallVector<int64_t> interleavedShape;
    interleavedShape.reserve(2 * rank);
    for (auto [d, r] :
         llvm::zip_equal(srcVecTy.getShape(), dstVecTy.getShape())) {
      interleavedShape.push_back(d / r); // S_k
      interleavedShape.push_back(r);     // R_k
    }
    auto interleavedTy =
        VectorType::get(interleavedShape, srcVecTy.getElementType());
    Value reshaped =
        vector::ShapeCastOp::create(rewriter, loc, interleavedTy, source);

    Value extractSource = reshaped;
    if (rank > 1) {
      SmallVector<int64_t> perm;
      perm.reserve(2 * rank);
      for (int64_t k = 0; k < rank; ++k)
        perm.push_back(2 * k);
      for (int64_t k = 0; k < rank; ++k)
        perm.push_back(2 * k + 1);
      extractSource =
          vector::TransposeOp::create(rewriter, loc, reshaped, perm);
    }

    // The indices are unsigned.
    Type indexTy = rewriter.getIndexType();
    SmallVector<OpFoldResult> positions = llvm::map_to_vector(
        adaptor.getIndices(), [&](Value idx) -> OpFoldResult {
          return arith::IndexCastUIOp::create(rewriter, loc, indexTy, idx)
              .getResult();
        });
    rewriter.replaceOpWithNewOp<vector::ExtractOp>(op, extractSource,
                                                   positions);
    return success();
  }
};

/// Convert cuda_tile.fma to math.fma, which rounds once to nearest even.
struct ConvertFma : public OptionsPattern<cuda_tile::FmaOp> {
  using OptionsPattern::OptionsPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::FmaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rounding = op.getRoundingMode();
    bool ftz = op.getFlushToZero();
    auto newOp = rewriter.replaceOpWithNewOp<math::FmaOp>(
        op, adaptor.getLhs(), adaptor.getRhs(), adaptor.getAcc());
    if (options.dropRoundingModes ||
        rounding != cuda_tile::RoundingMode::NEAREST_EVEN)
      preserveDroppedRounding(rewriter, rounding, newOp);
    preserveDroppedFlushToZero(rewriter, ftz, newOp);
    return success();
  }
};

/// Convert cuda_tile.for to scf.for over `index`.
///
/// The bounds are sign-extended to `index`, or zero-extended for `unsignedCmp`,
/// which maps to scf.for's `unsignedCmp`. The body is merged into the new loop
/// with the induction variable cast back to its original type. Loops whose
/// induction variable indexes tiles are rescaled after the conversion (see
/// rescaleTileLoops).
struct ConvertFor : public OpConversionPattern<cuda_tile::ForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ForOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type indexTy = rewriter.getIndexType();
    auto toIndex = [&](ValueRange bound) -> Value {
      if (op.getUnsignedCmp())
        return arith::IndexCastUIOp::create(rewriter, loc, indexTy,
                                            bound.front());
      return arith::IndexCastOp::create(rewriter, loc, indexTy, bound.front());
    };
    Value lb = toIndex(adaptor.getLowerBound());
    Value ub = toIndex(adaptor.getUpperBound());
    Value step = toIndex(adaptor.getStep());

    auto newForOp = scf::ForOp::create(
        rewriter, loc, lb, ub, step, flattenValues(adaptor.getInitValues()),
        /*bodyBuilder=*/nullptr, op.getUnsignedCmp());

    if (failed(
            rewriter.convertRegionTypes(&op.getRegion(), *getTypeConverter())))
      return failure();

    // Move the body into the new loop, replacing its implicit yield.
    Block *oldBody = op.getBody();
    Block *newBody = newForOp.getBody();
    if (newBody->mightHaveTerminator())
      rewriter.eraseOp(newBody->getTerminator());

    SmallVector<Value> replacingValues;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(newBody);
    replacingValues.push_back(arith::IndexCastOp::create(
        rewriter, loc, oldBody->getArgument(0).getType(),
        newForOp.getInductionVar()));
    for (auto arg : newForOp.getRegionIterArgs())
      replacingValues.push_back(arg);

    rewriter.mergeBlocks(oldBody, newBody, replacingValues);
    rewriter.replaceOpWithMultiple(
        op, groupByConvertedTypes(*getTypeConverter(), op.getResultTypes(),
                                  newForOp.getResults()));
    return success();
  }
};

/// Convert cuda_tile.ftof to arith.extf, arith.truncf, or arith.convertf for
/// formats of the same width.
///
/// Widening is exact, so its rounding mode is only recorded under
/// `drop-rounding-modes`. truncf and convertf take the rounding mode if arith
/// has it; otherwise the mode is recorded as dropped.
struct ConvertFToF : public OptionsPattern<cuda_tile::FToFOp> {
  using OptionsPattern::OptionsPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::FToFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getConvertedResultTypeOrFail(
        op, getTypeConverter(), rewriter, "cannot convert ftof result type");
    if (failed(resultTy))
      return failure();

    // On the CPU, tf32 converts to f32, so a conversion between the two keeps
    // the value.
    if (adaptor.getFrom().getType() == resultTy.value()) {
      rewriter.replaceOp(op, adaptor.getFrom());
      return success();
    }

    auto getFloatWidth = [](Type ty) -> unsigned {
      if (auto fTy = dyn_cast<FloatType>(ty))
        return fTy.getWidth();
      if (auto vTy = dyn_cast<VectorType>(ty))
        if (auto eTy = dyn_cast<FloatType>(vTy.getElementType()))
          return eTy.getWidth();
      return 0;
    };

    unsigned srcWidth = getFloatWidth(adaptor.getFrom().getType());
    unsigned dstWidth = getFloatWidth(resultTy.value());
    if (!srcWidth || !dstWidth)
      return rewriter.notifyMatchFailure(op,
                                         "ftof expects float or vector<float>");

    if (srcWidth < dstWidth) {
      auto extOp = rewriter.replaceOpWithNewOp<arith::ExtFOp>(
          op, resultTy.value(), adaptor.getFrom());
      if (options.dropRoundingModes)
        preserveDroppedRounding(rewriter, op.getRoundingMode(), extOp);
      return success();
    }

    auto arithRounding = options.dropRoundingModes
                             ? std::optional<arith::RoundingMode>()
                             : mapRoundingModeToArith(op.getRoundingMode());
    arith::RoundingModeAttr roundingAttr;
    if (arithRounding)
      roundingAttr =
          arith::RoundingModeAttr::get(rewriter.getContext(), *arithRounding);
    Operation *castOp;
    if (srcWidth > dstWidth)
      castOp = rewriter.replaceOpWithNewOp<arith::TruncFOp>(
          op, resultTy.value(), adaptor.getFrom(), roundingAttr,
          /*fastmath=*/arith::FastMathFlagsAttr{});
    else
      castOp = rewriter.replaceOpWithNewOp<arith::ConvertFOp>(
          op, resultTy.value(), adaptor.getFrom(), roundingAttr,
          /*fastmath=*/arith::FastMathFlagsAttr{});
    if (!arithRounding)
      preserveDroppedRounding(rewriter, op.getRoundingMode(), castOp);
    return success();
  }
};

using ConvertFToI = ConvertFromToSignednessCastWithRoundingOp<
    cuda_tile::FToIOp, arith::FPToSIOp, arith::FPToUIOp,
    cuda_tile::RoundingMode::NEAREST_INT_TO_ZERO>;

/// Convert cuda_tile.get_global to a memref.get_global of the global, which is
/// a cuda_tile.global or an already converted memref.global, cast to the
/// pointer type.
struct ConvertGetGlobal : public OpConversionPattern<cuda_tile::GetGlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::GetGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *symbolOp =
        SymbolTable::lookupNearestSymbolFrom(op, op.getNameAttr());
    if (!symbolOp)
      return rewriter.notifyMatchFailure(op,
                                         "referenced global symbol not found");

    FailureOr<MemRefType> rankedMemRefTy = failure();
    if (auto tileirGlobal = dyn_cast<cuda_tile::GlobalOp>(symbolOp)) {
      rankedMemRefTy = getGlobalMemRefTypeOrFail(
          tileirGlobal, *getTypeConverter(), rewriter, op);
    } else if (auto memrefGlobal = dyn_cast<memref::GlobalOp>(symbolOp)) {
      auto memrefTy = dyn_cast<MemRefType>(memrefGlobal.getType());
      if (!memrefTy)
        return rewriter.notifyMatchFailure(
            op, "referenced memref.global does not have a ranked memref type");
      rankedMemRefTy = memrefTy;
    } else {
      return rewriter.notifyMatchFailure(
          op,
          "referenced symbol is neither cuda_tile.global nor memref.global");
    }
    if (failed(rankedMemRefTy))
      return failure();

    auto resultTy = getConvertedResultTypeOrFail(
        op, getTypeConverter(), rewriter, "cannot convert get_global result");
    if (failed(resultTy))
      return failure();

    Value getGlobal = memref::GetGlobalOp::create(
        rewriter, op.getLoc(), *rankedMemRefTy, op.getNameAttr());

    if (getGlobal.getType() != resultTy.value()) {
      auto dstTy = dyn_cast<BaseMemRefType>(resultTy.value());
      auto srcTy = dyn_cast<BaseMemRefType>(getGlobal.getType());
      if (!dstTy || !srcTy || !memref::CastOp::areCastCompatible(srcTy, dstTy))
        return rewriter.notifyMatchFailure(
            op, "cannot cast memref.get_global result to converted pointer "
                "type");
      getGlobal = memref::CastOp::create(rewriter, op.getLoc(),
                                         resultTy.value(), getGlobal);
    }

    rewriter.replaceOp(op, getGlobal);
    return success();
  }
};

/// Convert cuda_tile.get_index_space_shape. Dimension i of the index space has
/// ceildiv(tensor_shape[dim_map[i]], viewStrides[i]) tiles, including a partial
/// edge tile.
struct ConvertGetIndexSpaceShape
    : public OpConversionPattern<cuda_tile::GetIndexSpaceShapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::GetIndexSpaceShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto viewInfoOr = getViewInfo(op, op.getSrc(), adaptor.getSrc(), rewriter);
    if (failed(viewInfoOr))
      return failure();
    ViewInfo viewInfo = std::move(*viewInfoOr);

    Location loc = op.getLoc();
    unsigned rank = viewInfo.tileShape.size();
    auto memrefTy = cast<MemRefType>(viewInfo.memref.getType());
    auto memrefShape = memrefTy.getShape();

    SmallVector<Value> results;
    for (unsigned i = 0; i < rank; ++i) {
      int64_t stride = viewInfo.viewStrides[i];
      unsigned tensorDim = viewInfo.dimMap[i];
      Type resultTy =
          getTypeConverter()->convertType(op->getResult(i).getType());
      if (!resultTy)
        return rewriter.notifyMatchFailure(
            op, "cannot convert get_index_space_shape result type");

      int64_t dimSize = memrefShape[tensorDim];
      Value castedResult;
      if (dimSize != ShapedType::kDynamic) {
        int64_t numTiles = (dimSize + stride - 1) / stride;
        Value cst = arith::ConstantIndexOp::create(rewriter, loc, numTiles);
        castedResult = castValueToType(rewriter, loc, cst, resultTy);
      } else {
        Value dimVal = memref::DimOp::create(
            rewriter, loc, viewInfo.memref,
            arith::ConstantIndexOp::create(rewriter, loc, tensorDim));
        Value strideVal = arith::ConstantIndexOp::create(rewriter, loc, stride);
        Value divResult =
            arith::CeilDivUIOp::create(rewriter, loc, dimVal, strideVal);
        castedResult = castValueToType(rewriter, loc, divResult, resultTy);
      }

      if (!castedResult)
        return rewriter.notifyMatchFailure(
            op, "cannot cast index_space_shape result to converted type");
      results.push_back(castedResult);
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

using ConvertGetNumTileBlocks =
    ConvertDimQueryOp<cuda_tile::GetNumTileBlocksOp, gpu::GridDimOp,
                      AppendedGridArgLayout::kGridDimBase>;

/// Convert cuda_tile.get_tensor_shape to the extents of the converted memref,
/// cast with index_castui since the shape is unsigned.
struct ConvertGetTensorShape
    : public OpConversionPattern<cuda_tile::GetTensorShapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::GetTensorShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memrefTy = dyn_cast<MemRefType>(adaptor.getSrc().getType());
    if (!memrefTy)
      return rewriter.notifyMatchFailure(
          op, "tensor_view source did not convert to a ranked memref");

    Location loc = op.getLoc();
    auto castIndexResultTo = [&](Value indexVal, Type dstTy) -> Value {
      if (dstTy == rewriter.getIndexType())
        return indexVal;
      if (isa<IntegerType>(dstTy))
        return arith::IndexCastUIOp::create(rewriter, loc, dstTy, indexVal);
      return Value();
    };

    SmallVector<Value> results;
    auto shape = memrefTy.getShape();
    for (auto [i, dimSize] : llvm::enumerate(shape)) {
      Type resultTy =
          getTypeConverter()->convertType(op->getResult(i).getType());
      if (!resultTy)
        return rewriter.notifyMatchFailure(
            op, "cannot convert get_tensor_shape result type");

      Value dimAsIndex;
      if (dimSize != ShapedType::kDynamic) {
        dimAsIndex = arith::ConstantIndexOp::create(rewriter, loc, dimSize);
      } else {
        dimAsIndex = memref::DimOp::create(
            rewriter, loc, adaptor.getSrc(),
            arith::ConstantIndexOp::create(rewriter, loc, i));
      }

      Value casted = castIndexResultTo(dimAsIndex, resultTy);
      if (!casted)
        return rewriter.notifyMatchFailure(
            op, "cannot cast tensor_shape result to converted type");
      results.push_back(casted);
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

using ConvertGetTileBlockId =
    ConvertDimQueryOp<cuda_tile::GetTileBlockIdOp, gpu::BlockIdOp,
                      AppendedGridArgLayout::kBlockIdBase>;

/// Convert cuda_tile.global to a mutable memref.global with the same name,
/// initializer and alignment.
struct ConvertGlobal : public OpConversionPattern<cuda_tile::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memrefTy =
        getGlobalMemRefTypeOrFail(op, *getTypeConverter(), rewriter, op);
    if (failed(memrefTy))
      return failure();

    // memref.global requires a tensor-typed initializer of the same shape.
    DenseElementsAttr initAttr = op.getValue();
    auto tensorTy = RankedTensorType::get(memrefTy->getShape(),
                                          memrefTy->getElementType());
    if (initAttr.getType() != tensorTy)
      initAttr = retypeDenseElements(initAttr, tensorTy);

    IntegerAttr alignmentAttr;
    if (op.getAlignment() != 0)
      alignmentAttr = rewriter.getI64IntegerAttr(op.getAlignment());

    rewriter.replaceOpWithNewOp<memref::GlobalOp>(
        op, op.getSymName(), /*sym_visibility=*/StringAttr(), *memrefTy,
        initAttr, /*constant=*/false, alignmentAttr);
    return success();
  }
};

/// Convert cuda_tile.if to scf.if.
struct ConvertIf : public OpConversionPattern<cuda_tile::IfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::IfOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "cannot convert if result types");

    bool hasElse = !op.getElseRegion().empty();
    auto newIfOp = scf::IfOp::create(rewriter, op.getLoc(), resultTypes,
                                     adaptor.getCondition().front(), hasElse);

    // Replace the auto-created (empty) scf.if block by the cuda_tile.if one.
    auto moveBody = [&](Block *oldBlock, Block *newBlock) {
      if (newBlock->mightHaveTerminator())
        rewriter.eraseOp(newBlock->getTerminator());
      rewriter.mergeBlocks(oldBlock, newBlock, {});
    };
    moveBody(op.getThenBlock(), newIfOp.thenBlock());
    if (hasElse)
      moveBody(op.getElseBlock(), newIfOp.elseBlock());

    rewriter.replaceOpWithMultiple(
        op, groupByConvertedTypes(*getTypeConverter(), op.getResultTypes(),
                                  newIfOp.getResults()));
    return success();
  }
};

/// Convert cuda_tile.iota to vector.step, cast to the element type.
struct ConvertIota : public OpConversionPattern<cuda_tile::IotaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::IotaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getConvertedResultTypeOrFail(
        op, getTypeConverter(), rewriter, "cannot convert iota result type");
    if (failed(resultTy))
      return failure();

    auto dstVecTy = dyn_cast<VectorType>(resultTy.value());
    if (!dstVecTy || dstVecTy.getRank() != 1)
      return rewriter.notifyMatchFailure(
          op, "iota expects a 1-D vector result after type conversion");

    auto indexVecTy =
        VectorType::get(dstVecTy.getShape(), rewriter.getIndexType());
    Value step = vector::StepOp::create(rewriter, op.getLoc(), indexVecTy);
    rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(op, dstVecTy, step);
    return success();
  }
};

using ConvertIToF = ConvertFromToSignednessCastWithRoundingOp<
    cuda_tile::IToFOp, arith::SIToFPOp, arith::UIToFPOp,
    cuda_tile::RoundingMode::NEAREST_EVEN>;

/// Convert cuda_tile.load_view_tko to a vector.transfer_read (see
/// buildTransferViewAccessPlan). Out-of-bounds lanes read the view's
/// `padding_value`, or poison without one.
struct ConvertLoadViewTko
    : public OptionsPattern<cuda_tile::LoadViewTkoOp,
                            TokenDroppingPattern<cuda_tile::LoadViewTkoOp>> {
  using OptionsPattern::OptionsPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::LoadViewTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (failed(checkCommonTkoGuards(op, rewriter)))
      return failure();

    auto vecTy = dyn_cast_or_null<VectorType>(
        getTypeConverter()->convertType(op.getTile().getType()));
    if (!vecTy)
      return rewriter.notifyMatchFailure(op, "cannot convert tile type");

    auto plan = buildTransferViewAccessPlan(
        rewriter, op, op.getView(), adaptor.getView(), adaptor.getIndex());
    if (failed(plan))
      return failure();

    Value padding;
    if (!plan->viewInfo.paddingValue) {
      padding = ub::PoisonOp::create(rewriter, loc, vecTy.getElementType());
    } else if (auto fty = dyn_cast<FloatType>(vecTy.getElementType())) {
      const llvm::fltSemantics &sem = fty.getFloatSemantics();
      APFloat val = APFloat::getZero(sem, /*Negative=*/false);
      switch (plan->viewInfo.paddingValue.getValue()) {
      case cuda_tile::PaddingValue::zero:
        val = APFloat::getZero(sem, /*Negative=*/false);
        break;
      case cuda_tile::PaddingValue::neg_zero:
        val = APFloat::getZero(sem, /*Negative=*/true);
        break;
      case cuda_tile::PaddingValue::nan:
        val = APFloat::getNaN(sem);
        break;
      case cuda_tile::PaddingValue::pos_inf:
        val = APFloat::getInf(sem, /*Negative=*/false);
        break;
      case cuda_tile::PaddingValue::neg_inf:
        val = APFloat::getInf(sem, /*Negative=*/true);
        break;
      }
      padding = arith::ConstantFloatOp::create(rewriter, loc, fty, val);
    } else {
      padding = arith::ConstantIntOp::create(rewriter, loc,
                                             vecTy.getElementType(), 0);
    }
    SmallVector<bool> inBounds = options.assumeInBounds
                                     ? SmallVector<bool>(vecTy.getRank(), true)
                                     : plan->inBounds;

    auto readOp = vector::TransferReadOp::create(
        rewriter, loc, vecTy, plan->viewInfo.memref, plan->memrefIndices,
        AffineMapAttr::get(plan->permutationMap), padding,
        /*mask=*/Value(), rewriter.getBoolArrayAttr(inBounds));
    preserveDroppedOptHints(op, readOp);

    rewriter.replaceOp(op, {readOp.getResult(), Value()});
    return success();
  }
};

/// The offset of the unranked pointer `unrankedBase` (`memref<*xT>`) into its
/// buffer. Advancing a pointer must add to this offset, since reinterpret_cast
/// offsets are absolute to the buffer.
///
/// Pointer arguments of functions have offset 0 by the calling convention (see
/// Passes.td), so none is read for them. This keeps every use of such an
/// argument a zero-offset reinterpret_cast, which
/// ConvertMemrefArgsToRankedMemref needs to promote it. Other block arguments,
/// e.g. loop-carried pointers, may have an offset and are queried.
static OpFoldResult
recoverUnrankedPtrOffset(ConversionPatternRewriter &rewriter, Location loc,
                         Value unrankedBase) {
  if (auto blockArg = dyn_cast<BlockArgument>(unrankedBase)) {
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (blockArg.getOwner()->isEntryBlock() &&
        isa_and_nonnull<FunctionOpInterface>(parentOp))
      return rewriter.getIndexAttr(0);
  }
  auto unrankedTy = cast<UnrankedMemRefType>(unrankedBase.getType());
  auto rankedTy = get1DDynamicOffsetMemRefType(unrankedTy.getElementType(),
                                               ShapedType::kDynamic,
                                               unrankedTy.getMemorySpace());
  Value ranked = memref::CastOp::create(rewriter, loc, rankedTy, unrankedBase);
  auto meta = memref::ExtractStridedMetadataOp::create(rewriter, loc, ranked);
  return meta.getOffset();
}

/// A rank-0 memref at the unranked pointer `unrankedBase`, keeping the offset
/// of the pointer; an offset of 0 would address the start of the buffer.
static Value
reinterpretScalarPtrPreservingOffset(ConversionPatternRewriter &rewriter,
                                     Location loc, Value unrankedBase) {
  auto unrankedTy = cast<UnrankedMemRefType>(unrankedBase.getType());
  OpFoldResult off = recoverUnrankedPtrOffset(rewriter, loc, unrankedBase);
  auto rank0Ty = MemRefType::get(
      {}, unrankedTy.getElementType(),
      StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic, {}),
      unrankedTy.getMemorySpace());
  return memref::ReinterpretCastOp::create(
             rewriter, loc, rank0Ty, unrankedBase,
             /*offset=*/off,
             /*sizes=*/SmallVector<OpFoldResult>{},
             /*strides=*/SmallVector<OpFoldResult>{})
      .getResult();
}

/// Convert cuda_tile.make_tensor_view to a memref.reinterpret_cast of the base
/// pointer at its offset, with the shape and strides of the view.
///
/// cuda_tile specifies the dynamic shape and stride operands as unsigned, but
/// they are sign-extended: --tileir-ptr-to-view derives them from signed
/// pointer offsets and mask bounds, which can be negative. Unsigned values of
/// 2^31 or more in i32 operands are therefore not supported.
struct ConvertMakeTensorView
    : public OpConversionPattern<cuda_tile::MakeTensorViewOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::MakeTensorViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = dyn_cast_or_null<MemRefType>(
        getTypeConverter()->convertType(op.getType()));
    if (!resultTy)
      return rewriter.notifyMatchFailure(
          op, "tensor_view did not convert to a ranked memref");

    Location loc = op.getLoc();
    auto toIndex = [&](ValueRange values) {
      return llvm::map_to_vector(values, [&](Value value) -> Value {
        return arith::IndexCastOp::create(rewriter, loc,
                                          rewriter.getIndexType(), value);
      });
    };
    auto tvType = cast<cuda_tile::TensorViewType>(op.getType());
    SmallVector<OpFoldResult> sizes = getMixedValues(
        tvType.getShape(), toIndex(adaptor.getDynamicShape()), rewriter);
    SmallVector<OpFoldResult> strides = getMixedValues(
        tvType.getStrides(), toIndex(adaptor.getDynamicStrides()), rewriter);
    OpFoldResult offset =
        recoverUnrankedPtrOffset(rewriter, loc, adaptor.getBase());

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, resultTy, adaptor.getBase(), offset, sizes, strides);
    return success();
  }
};

using ConvertMaxF =
    ConvertMinMaxFOp<cuda_tile::MaxFOp, arith::MaximumFOp, arith::MaxNumFOp>;

using ConvertMinF =
    ConvertMinMaxFOp<cuda_tile::MinFOp, arith::MinimumFOp, arith::MinNumFOp>;

/// Convert cuda_tile.mmaf to a vector.contract that accumulates into `acc`.
struct ConvertMmaF : public OpConversionPattern<cuda_tile::MmaFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::MmaFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return failure();

    auto vecResultTy = cast<VectorType>(resultType);
    auto spec =
        buildMmaContractionSpec(rewriter.getContext(), vecResultTy.getRank());
    if (failed(spec))
      return rewriter.notifyMatchFailure(
          op, "only 2D or 3D (batched) mmaf is supported");

    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        op, adaptor.getLhs(), adaptor.getRhs(), adaptor.getAcc(),
        rewriter.getAffineMapArrayAttr({spec->mapA, spec->mapB, spec->mapC}),
        rewriter.getArrayAttr(spec->iterTypes), vector::CombiningKind::ADD);
    return success();
  }
};

/// Convert cuda_tile.mmai to a vector.contract. The operands are first extended
/// to the accumulator type by their signedness, since vector.contract would
/// sign-extend them.
struct ConvertMmaI : public OpConversionPattern<cuda_tile::MmaIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::MmaIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return failure();

    auto vecResultTy = cast<VectorType>(resultType);
    auto spec =
        buildMmaContractionSpec(rewriter.getContext(), vecResultTy.getRank());
    if (failed(spec))
      return rewriter.notifyMatchFailure(
          op, "only 2D or 3D (batched) mmai is supported");

    Type accElemTy = vecResultTy.getElementType();
    auto extendToAcc = [&](Value operand,
                           cuda_tile::Signedness signedness) -> Value {
      auto operandTy = cast<VectorType>(operand.getType());
      if (operandTy.getElementType() == accElemTy)
        return operand;
      auto extTy = operandTy.clone(accElemTy);
      if (signedness == cuda_tile::Signedness::Unsigned)
        return arith::ExtUIOp::create(rewriter, op.getLoc(), extTy, operand);
      return arith::ExtSIOp::create(rewriter, op.getLoc(), extTy, operand);
    };
    Value lhs = extendToAcc(adaptor.getLhs(), op.getSignednessLhs());
    Value rhs = extendToAcc(adaptor.getRhs(), op.getSignednessRhs());
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        op, lhs, rhs, adaptor.getAcc(),
        rewriter.getAffineMapArrayAttr({spec->mapA, spec->mapB, spec->mapC}),
        rewriter.getArrayAttr(spec->iterTypes), vector::CombiningKind::ADD);
    return success();
  }
};

/// Convert cuda_tile.module to a gpu.module of the same name (GPU target), or
/// inline its contents into the enclosing module (CPU target).
struct ConvertModule : public OptionsPattern<cuda_tile::ModuleOp> {
  using OptionsPattern::OptionsPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ModuleOp tileirMod, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (options.target == TileIRTarget::GPU) {
      auto gpuMod = gpu::GPUModuleOp::create(rewriter, tileirMod.getLoc(),
                                             tileirMod.getSymName());
      Block *oldBody = &tileirMod.getBody().front();
      Block *newBody = gpuMod.getBody();
      if (newBody->mightHaveTerminator())
        rewriter.inlineBlockBefore(oldBody, newBody->getTerminator());
      else
        rewriter.inlineBlockBefore(oldBody, newBody, newBody->end());
    } else {
      rewriter.inlineBlockBefore(&tileirMod.getBody().front(), tileirMod);
    }

    rewriter.eraseOp(tileirMod);
    return success();
  }
};

using ConvertMulF = ConvertBinaryFloatOp<cuda_tile::MulFOp, arith::MulFOp>;

/// Convert cuda_tile.mulhii, which is unsigned, to the high half of
/// arith.mului_extended.
struct ConvertMulhiI : public OpConversionPattern<cuda_tile::MulhiIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::MulhiIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ext = arith::MulUIExtendedOp::create(rewriter, op.getLoc(),
                                              adaptor.getX(), adaptor.getY());
    rewriter.replaceOp(op, ext.getHigh());
    return success();
  }
};

using ConvertMulI =
    ConvertBinaryLhsRhsWithOverflowOp<cuda_tile::MulIOp, arith::MulIOp>;

/// Advance the pointer `ptr` (`memref<*xT>`) by `offset` elements (`index`).
///
/// The result is a unit view whose descriptor carries the accumulated offset,
/// which consumers recover with recoverUnrankedPtrOffset.
static Value offsetPointer(ConversionPatternRewriter &rewriter, Location loc,
                           Value ptr, Value offset) {
  // reinterpret_cast's offset is absolute to the underlying buffer.
  OpFoldResult totalOffset = offset;
  OpFoldResult ptrOffset = recoverUnrankedPtrOffset(rewriter, loc, ptr);
  if (!isZeroInteger(ptrOffset))
    totalOffset =
        arith::AddIOp::create(
            rewriter, loc,
            getValueOrCreateConstantIndexOp(rewriter, loc, ptrOffset), offset)
            .getResult();

  auto ptrTy = cast<UnrankedMemRefType>(ptr.getType());
  Value view = memref::ReinterpretCastOp::create(
      rewriter, loc,
      get1DDynamicOffsetMemRefType(ptrTy.getElementType(), /*size=*/1,
                                   ptrTy.getMemorySpace()),
      ptr, totalOffset, SmallVector<OpFoldResult>{rewriter.getIndexAttr(1)},
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(1)});
  return memref::CastOp::create(rewriter, loc, ptrTy, view);
}

/// Convert scalar cuda_tile.offset on pointer tiles to a memref view of the
/// advanced pointer (see offsetPointer).
struct ConvertOffsetScalarPtr
    : public OpConversionPattern<cuda_tile::OffsetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::OffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value offset = castValueToType(rewriter, op.getLoc(), adaptor.getOffset(),
                                   rewriter.getIndexType());
    if (!offset)
      return rewriter.notifyMatchFailure(
          op, "offset addend could not be converted to index");
    rewriter.replaceOp(
        op, offsetPointer(rewriter, op.getLoc(), adaptor.getPtr(), offset));
    return success();
  }
};

/// Convert cuda_tile.negi to arith.subi(0, source).
struct ConvertNegI : public OpConversionPattern<cuda_tile::NegIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::NegIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto overflow = op.getOverflow();
    Type ty = adaptor.getSource().getType();
    auto zeroAttr = rewriter.getZeroAttr(ty);
    if (!zeroAttr)
      return rewriter.notifyMatchFailure(
          op, "cannot create zero value for negi source type");
    Value zero = arith::ConstantOp::create(rewriter, op.getLoc(), ty, zeroAttr);
    auto newOp = rewriter.replaceOpWithNewOp<arith::SubIOp>(
        op, zero, adaptor.getSource());
    preserveDroppedOverflow(rewriter, overflow, newOp);
    return success();
  }
};

/// Convert a scalar cuda_tile.load_ptr_tko to a memref.load from the pointer.
struct ConvertLoadPtrTkoScalar
    : public TokenDroppingPattern<cuda_tile::LoadPtrTkoOp> {
  using TokenDroppingPattern::TokenDroppingPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::LoadPtrTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileTy = cast<cuda_tile::TileType>(op.getResult().getType());
    if (!tileTy.getShape().empty())
      return rewriter.notifyMatchFailure(
          op, "only scalar (rank-0) load_ptr_tko is supported here");
    if (op.getMask())
      return rewriter.notifyMatchFailure(
          op, "masked scalar load_ptr_tko is not supported");
    if (failed(checkCommonTkoGuards(op, rewriter)))
      return failure();

    if (!isa<UnrankedMemRefType>(adaptor.getSource().getType()))
      return rewriter.notifyMatchFailure(
          op, "expected unranked memref pointer source");

    Location loc = op.getLoc();
    Value rc = reinterpretScalarPtrPreservingOffset(rewriter, loc,
                                                    adaptor.getSource());
    auto loadOp = memref::LoadOp::create(rewriter, loc, rc, ValueRange{});
    preserveDroppedOptHints(op, loadOp);
    rewriter.replaceOp(op, {loadOp.getResult(), Value()});
    return success();
  }
};

/// Convert a scalar cuda_tile.store_ptr_tko to a memref.store to the pointer.
struct ConvertStorePtrTkoScalar
    : public TokenDroppingPattern<cuda_tile::StorePtrTkoOp> {
  using TokenDroppingPattern::TokenDroppingPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::StorePtrTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileTy = cast<cuda_tile::TileType>(op.getValue().getType());
    if (!tileTy.getShape().empty())
      return rewriter.notifyMatchFailure(
          op, "only scalar (rank-0) store_ptr_tko is supported here");
    if (op.getMask())
      return rewriter.notifyMatchFailure(
          op, "masked scalar store_ptr_tko is not supported");
    if (failed(checkCommonTkoGuards(op, rewriter)))
      return failure();

    if (!isa<UnrankedMemRefType>(adaptor.getDestination().getType()))
      return rewriter.notifyMatchFailure(
          op, "expected unranked memref pointer destination");

    Location loc = op.getLoc();
    Value rc = reinterpretScalarPtrPreservingOffset(rewriter, loc,
                                                    adaptor.getDestination());
    auto storeOp = memref::StoreOp::create(rewriter, loc, adaptor.getValue(),
                                           rc, ValueRange{});
    preserveDroppedOptHints(op, storeOp);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert ranked cuda_tile.offset on pointer tiles: the tile keeps its base
/// and adds the sign-extended offsets to the offsets of its lanes.
struct ConvertOffsetRanked : public OpConversionPattern<cuda_tile::OffsetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::OffsetOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange ptr = adaptor.getPtr();
    if (ptr.size() != 2)
      return rewriter.notifyMatchFailure(op, "scalar offset handled elsewhere");

    Location loc = op.getLoc();
    Value offset = adaptor.getOffset().front();
    auto offsetTy = cast<VectorType>(offset.getType());
    offset = arith::IndexCastOp::create(
        rewriter, loc,
        VectorType::get(offsetTy.getShape(), rewriter.getIndexType()), offset);
    Value offsets = arith::AddIOp::create(rewriter, loc, ptr.back(), offset);
    rewriter.replaceOpWithMultiple(op, {{ptr.front(), offsets}});
    return success();
  }
};

/// The 1-D memref that the lane offsets of a pointer tile with base `base`
/// index. It keeps a dynamic offset: `base` may be an advanced pointer, and a
/// static zero offset would make the gather/scatter address computation
/// ignore the descriptor's offset.
static Value getLaneMemRef(OpBuilder &builder, Location loc, Value base) {
  auto baseTy = cast<UnrankedMemRefType>(base.getType());
  return memref::CastOp::create(
      builder, loc,
      get1DDynamicOffsetMemRefType(baseTy.getElementType(),
                                   ShapedType::kDynamic,
                                   baseTy.getMemorySpace()),
      base);
}

/// The mask of a pointer tile access of `shape`: the converted `mask` operand,
/// or all-true when there is none.
static Value getMaskOrAllTrue(OpBuilder &builder, Location loc, ValueRange mask,
                              ArrayRef<int64_t> shape) {
  if (!mask.empty())
    return mask.front();
  Value trueVal = arith::ConstantIntOp::create(builder, loc, 1, 1);
  return vector::BroadcastOp::create(
      builder, loc, VectorType::get(shape, builder.getI1Type()), trueVal);
}

/// Convert a ranked cuda_tile.load_ptr_tko to a vector.gather from the base of
/// the pointer tile at its lane offsets. Masked-off lanes read the padding
/// value, or zero without one.
struct ConvertLoadPtrTkoRanked
    : public OpConversionPattern<cuda_tile::LoadPtrTkoOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::LoadPtrTkoOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange source = adaptor.getSource();
    if (source.size() != 2)
      return rewriter.notifyMatchFailure(op, "scalar load handled elsewhere");
    if (failed(checkCommonTkoGuards(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    auto resultVecTy = cast<VectorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    Value base = getLaneMemRef(rewriter, loc, source.front());
    Value mask = getMaskOrAllTrue(rewriter, loc, adaptor.getMask(),
                                  resultVecTy.getShape());
    Value passThru;
    if (op.getPaddingValue())
      passThru = adaptor.getPaddingValue().front();
    else
      passThru = arith::ConstantOp::create(rewriter, loc, resultVecTy,
                                           rewriter.getZeroAttr(resultVecTy));

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto gatherOp =
        vector::GatherOp::create(rewriter, loc, resultVecTy, base,
                                 ValueRange{c0}, source.back(), mask, passThru);
    preserveDroppedOptHints(op, gatherOp);
    rewriter.replaceOpWithMultiple(op, {{gatherOp.getResult()}, {}});
    return success();
  }
};

/// Convert a ranked cuda_tile.store_ptr_tko to a vector.scatter to the base of
/// the pointer tile at its lane offsets.
struct ConvertStorePtrTkoRanked
    : public OpConversionPattern<cuda_tile::StorePtrTkoOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::StorePtrTkoOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange destination = adaptor.getDestination();
    if (destination.size() != 2)
      return rewriter.notifyMatchFailure(op, "scalar store handled elsewhere");
    if (failed(checkCommonTkoGuards(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    Value value = adaptor.getValue().front();
    Value base = getLaneMemRef(rewriter, loc, destination.front());
    Value mask = getMaskOrAllTrue(rewriter, loc, adaptor.getMask(),
                                  cast<VectorType>(value.getType()).getShape());
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto scatterOp = vector::ScatterOp::create(
        rewriter, loc, /*resultType=*/Type(), base, ValueRange{c0},
        destination.back(), mask, value);
    preserveDroppedOptHints(op, scatterOp);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Map a cuda_tile atomic_rmw mode to its arith kind; MAX and MIN are signed,
/// XCHG is `assign`.
static arith::AtomicRMWKind
mapAtomicRMWMode(cuda_tile::AtomicRMWMode mode) {
  switch (mode) {
  case cuda_tile::AtomicRMWMode::AND:
    return arith::AtomicRMWKind::andi;
  case cuda_tile::AtomicRMWMode::OR:
    return arith::AtomicRMWKind::ori;
  case cuda_tile::AtomicRMWMode::XOR:
    return arith::AtomicRMWKind::xori;
  case cuda_tile::AtomicRMWMode::ADD:
    return arith::AtomicRMWKind::addi;
  case cuda_tile::AtomicRMWMode::ADDF:
    return arith::AtomicRMWKind::addf;
  case cuda_tile::AtomicRMWMode::MAX:
    return arith::AtomicRMWKind::maxs;
  case cuda_tile::AtomicRMWMode::MIN:
    return arith::AtomicRMWKind::mins;
  case cuda_tile::AtomicRMWMode::UMAX:
    return arith::AtomicRMWKind::maxu;
  case cuda_tile::AtomicRMWMode::UMIN:
    return arith::AtomicRMWKind::minu;
  case cuda_tile::AtomicRMWMode::XCHG:
    return arith::AtomicRMWKind::assign;
  }
  llvm_unreachable("unhandled cuda_tile atomic_rmw mode");
}

/// Convert a scalar cuda_tile.atomic_rmw_tko to a memref.atomic_rmw at the
/// pointer; both return the old value.
///
/// memref.atomic_rmw lowers to an acq_rel atomic at system scope, which is at
/// least as strong as any ordering and scope. They are therefore recorded as
/// dropped rather than rejected as for load/store_tko (see
/// checkCommonTkoGuards).
struct ConvertAtomicRMWTko
    : public TokenDroppingPattern<cuda_tile::AtomicRMWTkoOp> {
  using TokenDroppingPattern::TokenDroppingPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::AtomicRMWTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileTy = cast<cuda_tile::TileType>(op.getResult().getType());
    if (!tileTy.getShape().empty())
      return rewriter.notifyMatchFailure(
          op, "only scalar (rank-0) atomic_rmw_tko is supported here");
    if (auto mask = op.getMask()) {
      // memref.atomic_rmw has no mask, so only a constant true one is accepted.
      auto maskCst = mask.getDefiningOp<cuda_tile::ConstantOp>();
      bool maskStaticallyTrue =
          mask.getType().getShape().empty() && maskCst &&
          maskCst.getValue().isSplat() &&
          maskCst.getValue().getSplatValue<llvm::APInt>().getBoolValue();
      if (!maskStaticallyTrue)
        return rewriter.notifyMatchFailure(
            op, "masked scalar atomic_rmw_tko is not supported");
    }

    if (!isa<UnrankedMemRefType>(adaptor.getPointers().getType()))
      return rewriter.notifyMatchFailure(
          op, "expected unranked memref pointer source");

    Location loc = op.getLoc();
    Value rc = reinterpretScalarPtrPreservingOffset(rewriter, loc,
                                                    adaptor.getPointers());
    auto rmw = memref::AtomicRMWOp::create(rewriter, loc,
                                           mapAtomicRMWMode(op.getMode()),
                                           adaptor.getArg(), rc,
                                           /*indices=*/ValueRange{});
    rmw->setAttr(
        kDroppedMemoryOrdering,
        rewriter.getStringAttr(cuda_tile::stringifyMemoryOrderingSemantics(
            op.getMemoryOrderingSemantics())));
    rmw->setAttr(kDroppedMemoryScope,
                 rewriter.getStringAttr(
                     cuda_tile::stringifyMemoryScope(op.getMemoryScope())));
    rewriter.replaceOp(op, {rmw.getResult(), Value()});
    return success();
  }
};

/// Convert cuda_tile.permute to vector.transpose.
struct ConvertPermute : public OpConversionPattern<cuda_tile::PermuteOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::PermuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<int64_t> perm(op.getPermutation().begin(),
                              op.getPermutation().end());
    rewriter.replaceOpWithNewOp<vector::TransposeOp>(op, adaptor.getSource(),
                                                     perm);
    return success();
  }
};

/// Convert cuda_tile.ptr_to_ptr between pointers that convert to the same type,
/// e.g. tf32 and f32 pointers on the CPU. Other casts fail: they would change
/// the element type of an unranked memref, which memref.cast cannot.
struct ConvertPtrToPtrCastOrFail
    : public OpConversionPattern<cuda_tile::PtrToPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::PtrToPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy =
        getConvertedResultTypeOrFail(op, this->getTypeConverter(), rewriter,
                                     "cannot convert cast result type");
    if (failed(resultTy))
      return failure();

    Value source = adaptor.getSource();
    auto resultMemRefTy = dyn_cast<BaseMemRefType>(resultTy.value());
    auto sourceMemRefTy = dyn_cast<BaseMemRefType>(source.getType());
    if (!resultMemRefTy || !sourceMemRefTy)
      return rewriter.notifyMatchFailure(
          op, "ptr_to_ptr requires memref source/result after type conversion");

    if (sourceMemRefTy == resultMemRefTy) {
      rewriter.replaceOp(op, source);
      return success();
    }

    if (!memref::CastOp::areCastCompatible(sourceMemRefTy, resultMemRefTy))
      return rewriter.notifyMatchFailure(
          op, "ptr_to_ptr cannot be represented as memref.cast");

    rewriter.replaceOpWithNewOp<memref::CastOp>(op, resultTy.value(), source);
    return success();
  }
};

/// Convert cuda_tile.reduce to vector.multi_reduction, or to vector.reduction
/// for a 1-D tile (see matchSingleOperandCombiningOp).
struct ConvertReduce : public OpConversionPattern<cuda_tile::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pre =
        matchSingleOperandCombiningOp(op, adaptor.getOperands(), rewriter);
    if (failed(pre))
      return failure();
    auto [kind, source, srcVecTy, identityAttr] = *pre;

    Type resultTy = getTypeConverter()->convertType(op.getResult(0).getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "cannot convert result type");

    Location loc = op.getLoc();
    uint32_t dim = op.getDim();

    if (auto dstVecTy = dyn_cast<VectorType>(resultTy)) {
      SmallVector<int64_t> expectedShape = getReducedVectorShape(srcVecTy, dim);
      if (!llvm::equal(dstVecTy.getShape(), expectedShape) ||
          dstVecTy.getElementType() != srcVecTy.getElementType())
        return rewriter.notifyMatchFailure(
            op, "reduce result vector type does not match source shape with "
                "the reduced dimension removed");

      Value acc = arith::ConstantOp::create(
          rewriter, loc, dstVecTy,
          SplatElementsAttr::get(dstVecTy, identityAttr));
      rewriter.replaceOpWithNewOp<vector::MultiDimReductionOp>(
          op, kind, source, acc,
          rewriter.getDenseI64ArrayAttr({static_cast<int64_t>(dim)}));
    } else {
      if (srcVecTy.getRank() != 1 || resultTy != srcVecTy.getElementType())
        return rewriter.notifyMatchFailure(
            op, "scalar reduce results require a 1-D source and matching "
                "element type");

      Value acc =
          arith::ConstantOp::create(rewriter, loc, resultTy, identityAttr);
      rewriter.replaceOpWithNewOp<vector::ReductionOp>(op, kind, source, acc);
    }
    return success();
  }
};

/// Convert cuda_tile.reshape to vector.shape_cast. A rank-0 source, which is a
/// scalar, is broadcast instead, and a rank-0 result is extracted.
///
/// A pointer tile keeps its base and reshapes its offsets, which are zero for
/// a scalar pointer. A scalar pointer result is the base advanced by the
/// offset of the single lane.
struct ConvertReshape : public OpConversionPattern<cuda_tile::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ReshapeOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertType(op.getType(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "cannot convert result type");

    Location loc = op.getLoc();
    ValueRange source = adaptor.getSource();
    if (isa<cuda_tile::PointerType>(op.getType().getElementType())) {
      Value base = source.front();
      Value offsets = source.size() == 2 ? source.back() : Value();
      if (resultTypes.size() == 1) {
        if (offsets) {
          SmallVector<int64_t> lane(
              cast<VectorType>(offsets.getType()).getRank(), 0);
          base = offsetPointer(
              rewriter, loc, base,
              vector::ExtractOp::create(rewriter, loc, offsets, lane));
        }
        rewriter.replaceOp(op, base);
        return success();
      }
      auto offsetsTy = cast<VectorType>(resultTypes.back());
      if (offsets)
        offsets =
            vector::ShapeCastOp::create(rewriter, loc, offsetsTy, offsets);
      else
        offsets = vector::BroadcastOp::create(
            rewriter, loc, offsetsTy,
            arith::ConstantIndexOp::create(rewriter, loc, 0));
      rewriter.replaceOpWithMultiple(op, {{base, offsets}});
      return success();
    }

    Value src = source.front();
    auto srcVecTy = dyn_cast<VectorType>(src.getType());
    auto dstVecTy = dyn_cast<VectorType>(resultTypes.front());
    if (srcVecTy && dstVecTy) {
      rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, dstVecTy, src);
    } else if (!srcVecTy && dstVecTy) {
      rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, dstVecTy, src);
    } else if (srcVecTy && !dstVecTy) {
      SmallVector<int64_t> indices(srcVecTy.getRank(), 0);
      rewriter.replaceOpWithNewOp<vector::ExtractOp>(op, src, indices);
    } else {
      rewriter.replaceOp(op, src);
    }
    return success();
  }
};

/// Convert cuda_tile.return to gpu.return (gpu target) or func.return (cpu
/// target).
struct ConvertReturn : public OptionsPattern<cuda_tile::ReturnOp> {
  using OptionsPattern::OptionsPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (options.target == TileIRTarget::GPU)
      rewriter.replaceOpWithNewOp<gpu::ReturnOp>(op);
    else
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

/// Convert cuda_tile.rsqrt to math.rsqrt.
using ConvertRsqrt =
    ConvertUnaryFlushToZeroOp<cuda_tile::RsqrtOp, math::RsqrtOp>;

/// Convert cuda_tile.scan, which is inclusive, to an inclusive vector.scan (see
/// matchSingleOperandCombiningOp).
struct ConvertScan : public OpConversionPattern<cuda_tile::ScanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getReverse())
      return rewriter.notifyMatchFailure(
          op, "reverse scan is not representable in vector.scan");

    auto pre =
        matchSingleOperandCombiningOp(op, adaptor.getOperands(), rewriter);
    if (failed(pre))
      return failure();
    auto [kind, source, srcVecTy, identityAttr] = *pre;

    Type resultTy = getTypeConverter()->convertType(op.getResult(0).getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "cannot convert scan result type");
    if (resultTy != source.getType())
      return rewriter.notifyMatchFailure(op,
                                         "scan result type must match the "
                                         "source vector type after conversion");

    Location loc = op.getLoc();
    int64_t dim = static_cast<int64_t>(op.getDim());

    // Each scan starts from the identity.
    SmallVector<int64_t> initShape = getReducedVectorShape(srcVecTy, dim);
    auto initTy = VectorType::get(initShape, srcVecTy.getElementType());

    Value initVal = arith::ConstantOp::create(
        rewriter, loc, initTy, SplatElementsAttr::get(initTy, identityAttr));

    auto scanOp = vector::ScanOp::create(rewriter, loc, kind, source, initVal,
                                         /*reduction_dim=*/dim,
                                         /*inclusive=*/true);
    rewriter.replaceOp(op, scanOp.getDest());
    return success();
  }
};

using ConvertShLI =
    ConvertBinaryLhsRhsWithOverflowOp<cuda_tile::ShLIOp, arith::ShLIOp>;

/// Convert cuda_tile.sqrt to math.sqrt, which is correctly rounded.
using ConvertSqrt =
    ConvertUnaryApproxMathOp<cuda_tile::SqrtOp, math::SqrtOp,
                             cuda_tile::RoundingMode::NEAREST_EVEN,
                             /*PreserveFtz=*/true>;

/// Convert cuda_tile.store_view_tko to a vector.transfer_write (see
/// buildTransferViewAccessPlan), which skips out-of-bounds lanes as the view
/// semantics require.
struct ConvertStoreViewTko
    : public OptionsPattern<cuda_tile::StoreViewTkoOp,
                            TokenDroppingPattern<cuda_tile::StoreViewTkoOp>> {
  using OptionsPattern::OptionsPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::StoreViewTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (failed(checkCommonTkoGuards(op, rewriter)))
      return failure();

    auto plan = buildTransferViewAccessPlan(
        rewriter, op, op.getView(), adaptor.getView(), adaptor.getIndex());
    if (failed(plan))
      return failure();

    SmallVector<bool> inBounds =
        options.assumeInBounds
            ? SmallVector<bool>(plan->viewInfo.tileShape.size(), true)
            : plan->inBounds;

    auto writeOp = vector::TransferWriteOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{}, adaptor.getTile(),
        plan->viewInfo.memref, plan->memrefIndices,
        AffineMapAttr::get(plan->permutationMap),
        /*mask=*/Value(), rewriter.getBoolArrayAttr(inBounds));
    preserveDroppedOptHints(op, writeOp);

    rewriter.eraseOp(op);
    return success();
  }
};

using ConvertSubF = ConvertBinaryFloatOp<cuda_tile::SubFOp, arith::SubFOp>;

using ConvertSubI =
    ConvertBinaryLhsRhsWithOverflowOp<cuda_tile::SubIOp, arith::SubIOp>;

/// Convert cuda_tile.tanh to math.tanh, which has full precision.
using ConvertTanH = ConvertUnaryApproxMathOp<cuda_tile::TanHOp, math::TanhOp,
                                             cuda_tile::RoundingMode::FULL,
                                             /*PreserveFtz=*/false>;

/// Convert cuda_tile.trunci to arith.trunci with the same overflow flags.
struct ConvertTruncI : public OpConversionPattern<cuda_tile::TruncIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::TruncIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getConvertedResultTypeOrFail(
        op, getTypeConverter(), rewriter, "cannot convert trunci result type");
    if (failed(resultTy))
      return failure();

    auto overflowAttr = arith::IntegerOverflowFlagsAttr::get(
        rewriter.getContext(), mapIntegerOverflowFlags(op.getOverflow()));
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(
        op, resultTy.value(), adaptor.getFrom(), overflowAttr);
    return success();
  }
};

using ConvertYield = ConvertToScfYield<cuda_tile::YieldOp>;

//===----------------------------------------------------------------------===//
// Type converter and conversion pattern population
//===----------------------------------------------------------------------===//

/// Populate the type conversions of the value model (see the file header).
static void populateTileIRToMLIRTypeConverter(TypeConverter &converter,
                                              MLIRContext *ctx,
                                              TileIRTarget target) {
  // Types of other dialects are kept; cuda_tile types must be converted below.
  converter.addConversion([](Type type) -> std::optional<Type> {
    if (isa<cuda_tile::CudaTileDialect>(type.getDialect()))
      return std::nullopt;
    return type;
  });

  // CPU has no tf32 representation; widen it (exactly) to f32 wherever it is
  // used as an element type.
  if (target == TileIRTarget::CPU)
    converter.addConversion(
        [ctx](FloatTF32Type) -> Type { return Float32Type::get(ctx); });

  // cuda_tile.tile<MxNxelemTy> -> vector<MxNxelemTy> (ranked tiles)
  // cuda_tile.tile<elemTy> (scalar, rank 0) -> elemTy
  // Pointer tiles: tile<ptr<T>> -> memref<*xT>, and a ranked
  // tile<MxNxptr<T>> -> (memref<*xT>, vector<MxNxindex>), i.e. its base
  // pointer and the element offsets of its lanes from that base.
  converter.addConversion([ctx, &converter](
                              cuda_tile::TileType tileTy,
                              SmallVectorImpl<Type> &results) -> LogicalResult {
    auto shape = tileTy.getShape();
    Type elemTy = tileTy.getElementType();

    if (auto ptrTy = dyn_cast<cuda_tile::PointerType>(elemTy)) {
      Type pointeeTy = converter.convertType(ptrTy.getPointeeType());
      if (!pointeeTy)
        return failure();
      results.push_back(UnrankedMemRefType::get(pointeeTy, {}));
      if (!shape.empty())
        results.push_back(VectorType::get(shape, IndexType::get(ctx)));
      return success();
    }
    elemTy = converter.convertType(elemTy);
    if (!elemTy || !elemTy.isIntOrFloat())
      return failure();
    results.push_back(shape.empty() ? elemTy : VectorType::get(shape, elemTy));
    return success();
  });

  // Views convert to the memref of their tensor_view. Each access reads the
  // tiling (tile shape, strides, dim_map, padding) from the view type.
  converter.addConversion([&converter](cuda_tile::TensorViewType tvTy) -> Type {
    return tensorViewToMemRefType(tvTy, converter);
  });
  converter.addConversion(
      [&converter](cuda_tile::PartitionViewType pvTy) -> Type {
        return tensorViewToMemRefType(pvTy.getTensorView(), converter);
      });
  converter.addConversion(
      [&converter](cuda_tile::StridedViewType svTy) -> Type {
        return tensorViewToMemRefType(svTy.getTensorView(), converter);
      });
  converter.addConversion(
      [&converter](cuda_tile::GatherScatterViewType gsTy) -> Type {
        return tensorViewToMemRefType(gsTy.getTensorView(), converter);
      });

  // Tokens only order memory operations, which the lowered IR orders by
  // program order, so they convert to no values.
  converter.addConversion(
      [](cuda_tile::TokenType, SmallVectorImpl<Type> &) { return success(); });
}

/// Populate the patterns of all lowered cuda_tile ops.
static void populateTileIRToMLIRConversionPatterns(
    TypeConverter &converter, RewritePatternSet &patterns,
    const ConvertTileIRToMLIRPassOptions &options) {
  MLIRContext *ctx = patterns.getContext();
  // Patterns that depend on the pass options.
  patterns.add<ConvertAddF, ConvertSubF, ConvertMulF, ConvertDivF, ConvertEntry,
               ConvertExp, ConvertFma, ConvertFToF, ConvertFToI,
               ConvertGetNumTileBlocks, ConvertGetTileBlockId, ConvertIToF,
               ConvertLoadViewTko, ConvertModule, ConvertReturn, ConvertSqrt,
               ConvertStoreViewTko, ConvertTanH>(converter, ctx, options);
  // Ops that map to one target op with the same operands, or to one of their
  // operands.
  patterns.add<
      DirectConversion<cuda_tile::AbsFOp, math::AbsFOp>,
      DirectConversion<cuda_tile::AbsIOp, math::AbsIOp>,
      DirectConversion<cuda_tile::AndIOp, arith::AndIOp>,
      DirectConversion<cuda_tile::Atan2Op, math::Atan2Op>,
      DirectConversion<cuda_tile::BitcastOp, arith::BitcastOp>,
      DirectConversion<cuda_tile::CeilOp, math::CeilOp>,
      DirectConversion<cuda_tile::CosOp, math::CosOp>,
      DirectConversion<cuda_tile::CosHOp, math::CoshOp>,
      DirectConversion<cuda_tile::FloorOp, math::FloorOp>,
      DirectConversion<cuda_tile::LogOp, math::LogOp>,
      DirectConversion<cuda_tile::Log2Op, math::Log2Op>,
      DirectConversion<cuda_tile::NegFOp, arith::NegFOp>,
      DirectConversion<cuda_tile::OrIOp, arith::OrIOp>,
      DirectConversion<cuda_tile::PackOp, vector::BitCastOp>,
      DirectConversion<cuda_tile::PowOp, math::PowFOp>,
      DirectConversion<cuda_tile::RemFOp, arith::RemFOp>,
      DirectConversion<cuda_tile::SelectOp, arith::SelectOp>,
      DirectConversion<cuda_tile::SinOp, math::SinOp>,
      DirectConversion<cuda_tile::SinHOp, math::SinhOp>,
      DirectConversion<cuda_tile::TanOp, math::TanOp>,
      DirectConversion<cuda_tile::UnpackOp, vector::BitCastOp>,
      DirectConversion<cuda_tile::XOrIOp, arith::XOrIOp>,
      ForwardOperand<cuda_tile::AssumeOp>,
      ForwardOperand<cuda_tile::MakeGatherScatterViewOp>,
      ForwardOperand<cuda_tile::MakePartitionViewOp>,
      ForwardOperand<cuda_tile::MakeStridedViewOp>,
      SignednessConversion<cuda_tile::ExtIOp, arith::ExtSIOp, arith::ExtUIOp>,
      SignednessConversion<cuda_tile::MaxIOp, arith::MaxSIOp, arith::MaxUIOp>,
      SignednessConversion<cuda_tile::MinIOp, arith::MinSIOp, arith::MinUIOp>,
      SignednessConversion<cuda_tile::RemIOp, arith::RemSIOp, arith::RemUIOp>,
      SignednessConversion<cuda_tile::ShRIOp, arith::ShRSIOp, arith::ShRUIOp>>(
      converter, ctx);
  patterns.add<
      ConvertAddI, ConvertAlloca, ConvertBroadcast, ConvertCat, ConvertCmpF,
      ConvertCmpI, ConvertConstant, ConvertContinue, ConvertAtomicRMWTko,
      ConvertDivI, ConvertExp2, ConvertExtract, ConvertFor, ConvertGetGlobal,
      ConvertGetIndexSpaceShape, ConvertGetTensorShape, ConvertGlobal,
      ConvertIf, ConvertIota, EraseTokenOp<cuda_tile::JoinTokensOp>,
      ConvertLoadPtrTkoRanked, ConvertLoadPtrTkoScalar, ConvertMakeTensorView,
      EraseTokenOp<cuda_tile::MakeTokenOp>, ConvertMaxF, ConvertMinF,
      ConvertMmaF, ConvertMmaI, ConvertMulhiI, ConvertMulI, ConvertOffsetRanked,
      ConvertOffsetScalarPtr, ConvertNegI, ConvertPermute,
      ConvertPtrToPtrCastOrFail, ConvertReduce, ConvertReshape, ConvertRsqrt,
      ConvertScan, ConvertShLI, ConvertStorePtrTkoRanked,
      ConvertStorePtrTkoScalar, ConvertSubI, ConvertTruncI, ConvertYield>(
      converter, ctx);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

/// Converts all cuda_tile ops, then applies the rewrites of PostConversion.h.
struct ConvertTileIRToMLIRPass
    : public impl::ConvertTileIRToMLIRPassBase<ConvertTileIRToMLIRPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    if (!knownBlockSize.empty() && knownBlockSize.size() != 3) {
      module.emitError("'known-block-size' expects exactly three values");
      return signalPassFailure();
    }

    TypeConverter typeConverter;
    populateTileIRToMLIRTypeConverter(typeConverter, ctx, target);

    RewritePatternSet patterns(ctx);
    populateTileIRToMLIRConversionPatterns(typeConverter, patterns,
                                           {target, appendGridArgs,
                                            dropRoundingModes, assumeInBounds,
                                            llvm::to_vector(knownBlockSize)});

    // TileIR ops are illegal; the ops they lower to are legal once all their
    // types are legal.
    ConversionTarget conversionTarget(*ctx);
    conversionTarget.addIllegalDialect<cuda_tile::CudaTileDialect>();
    auto hasLegalTypes = [&](Operation *op) {
      return typeConverter.isLegal(op);
    };
    if (target == TileIRTarget::GPU)
      conversionTarget.addDynamicallyLegalDialect<gpu::GPUDialect>(
          hasLegalTypes);
    else
      conversionTarget.addDynamicallyLegalDialect<func::FuncDialect>(
          hasLegalTypes);
    conversionTarget.addDynamicallyLegalDialect<
        arith::ArithDialect, math::MathDialect, memref::MemRefDialect,
        scf::SCFDialect, ub::UBDialect, vector::VectorDialect>(hasLegalTypes);

    if (failed(applyPartialConversion(module, conversionTarget,
                                      std::move(patterns))))
      return signalPassFailure();

    tileir::rescaleTileLoops(module);
    tileir::scopeLoopAllocations(module);

    RewritePatternSet postConversionPatterns(ctx);
    tileir::populatePostConversionPatterns(postConversionPatterns);
    walkAndApplyPatterns(module, std::move(postConversionPatterns));

    // gpu.module ops need a container module.
    if (target == TileIRTarget::GPU)
      module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                      UnitAttr::get(ctx));
  }
};

} // namespace
