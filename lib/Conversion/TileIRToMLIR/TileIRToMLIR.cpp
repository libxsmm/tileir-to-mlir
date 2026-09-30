//===- TileIRToMLIR.cpp - Tile IR to MLIR conversion --------*- C++ -*-===//
//
// Part of the tileir-to-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Conversion pass from Tile IR to GPU/vector/scf/arith/memref ops.
//
// The patterns registered in populateTileIRToMLIRConversionPatterns are the
// authoritative list of supported ops. Any cuda_tile op left without a pattern
// stays illegal and makes the pass fail with a conversion diagnostic.
//
// A few ops are intentionally not lowered because they have no faithful
// representation in the target dialects; they must be removed by an earlier
// pass or are rejected here: AssertOp, AtomicCASTkoOp, AtomicRedViewTkoOp,
// BreakOp, IntToPtrOp, LoopOp, MmafScaledOp, PrintTkoOp, PtrToIntOp. A
// gather_scatter_view can be created but not accessed.
//
// Ranked pointer tiles are lowered to gathers and scatters. The
// --tileir-ptr-to-view pass raises the ones it can to view accesses first.
// AtomicRMWTkoOp is only lowered for scalar pointers.
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

/// Derive the ranked MemRefType that corresponds to a tensor_view type.
///
/// The memref always carries a *dynamic* offset in its strided layout. A
/// tensor_view may start at an arbitrary position within its buffer when its
/// base pointer was pre-shifted by a scalar `offset` op (e.g. a per-batch /
/// per-channel output base). `memref.reinterpret_cast`'s offset is absolute to
/// the underlying buffer, so the memref type must be able to represent that
/// (possibly non-zero) offset for it to survive make_tensor_view and the
/// downstream transfer lowering. Strides come straight from the tensor_view
/// (and may themselves be dynamic).
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

/// Build a 1-D memref type with unit stride and dynamic offset.
///
/// This is the canonical transient view type used when lowering unranked
/// pointer values (`memref<*xT>`) through reinterpret_cast-based arithmetic or
/// metadata extraction. `size` is either a static extent (e.g. 1) or
/// `ShapedType::kDynamic`.
static MemRefType get1DDynamicOffsetMemRefType(Type elemTy, int64_t size,
                                               Attribute memorySpace = {}) {
  return MemRefType::get({size}, elemTy,
                         StridedLayoutAttr::get(elemTy.getContext(),
                                                ShapedType::kDynamic,
                                                SmallVector<int64_t>{1}),
                         memorySpace);
}

/// Layout information extracted from a tile-view operand at a use site.
///
/// Covers both `partition_view` and `strided_view`, which share the same
/// "rectangular tile laid out on a grid" access shape. The only structural
/// difference is how far the tile base advances between adjacent index-space
/// positions:
///   - partition_view: the base advances by `tile_shape[i]` (tiles tile the
///     tensor exactly, no overlap, no gaps).
///   - strided_view:   the base advances by `traversal_strides[i]` (tiles may
///     overlap when stride < tile, or leave gaps when stride > tile).
/// This advance is captured in `viewStrides`; everything downstream (memref
/// offset, index-space-shape, in-bounds analysis) is expressed in terms of it.
struct ViewInfo {
  Value memref;                   // Converted memref backing the view
  SmallVector<int64_t> tileShape; // Tile dimensions (per tile dim)
  SmallVector<int64_t>
      viewStrides;             // Base advance per index step (per tile dim)
  SmallVector<int32_t> dimMap; // Mapping from tile dims to tensor_view dims
  // Optional padding value attribute from the view type; null if the view does
  // not specify one (i.e. OOB loads yield unspecified values).
  cuda_tile::PaddingValueAttr paddingValue;
};

/// Extract tile-view layout info from `view`'s type and pair it with the
/// already type-converted memref `convertedView`. Returns failure (with a
/// match-failure note) for view kinds that the transfer-based lowering cannot
/// model (e.g. gather_scatter_view, whose sparse dimension requires gather /
/// scatter rather than a contiguous transfer).
///
/// The view type verifiers guarantee that the tile and the tensor_view have
/// the same rank, that dim_map is a permutation, and that tile sizes and
/// strides are positive.
static FailureOr<ViewInfo> getViewInfo(Operation *op, Value view,
                                       Value convertedView,
                                       ConversionPatternRewriter &rewriter) {
  ViewInfo info;
  info.memref = convertedView;

  // partition_view and strided_view differ only in the per-dim base advance.
  auto fill = [&](auto viewTy, ArrayRef<int32_t> advance) {
    ArrayRef<int32_t> tile = viewTy.getTileShape().asArrayRef();
    info.tileShape.assign(tile.begin(), tile.end());
    info.viewStrides.assign(advance.begin(), advance.end());
    info.dimMap.assign(viewTy.getDimMap().begin(), viewTy.getDimMap().end());
    info.paddingValue = viewTy.getPaddingValue();
  };

  // partition_view tiles tile the tensor exactly: advance == tile extent.
  if (auto pvType = dyn_cast<cuda_tile::PartitionViewType>(view.getType())) {
    fill(pvType, pvType.getTileShape().asArrayRef());
    return info;
  }

  // strided_view advances the tile base by the traversal stride.
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

/// Cast between index and integer types when required by lowered ops.
///
/// Returns a null Value if the cast is not supported (only index<->integer and
/// the identity case are handled). Callers must check the result and bail (via
/// notifyMatchFailure) on null.
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

/// Rebuild `denseAttr` as a DenseElementsAttr of `newType`, which has the same
/// number of elements and whose element type may differ by the element type
/// conversion of the type converter (e.g. tile->vector or tile->tensor).
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

/// Build the ranked memref type corresponding to a cuda_tile.global
/// definition.
///
/// cuda_tile.global stores a DenseElementsAttr payload and semantically
/// materializes a static allocation initialized at module load time. We lower
/// that allocation to memref.global with a ranked static memref type matching
/// the payload shape and converted element type.
static FailureOr<MemRefType> getGlobalMemRefTypeOrFail(
    cuda_tile::GlobalOp globalOp, const TypeConverter &converter,
    ConversionPatternRewriter &rewriter, Operation *diagnosticOp) {
  auto initTy = dyn_cast<ShapedType>(globalOp.getValue().getType());
  if (!initTy || !initTy.hasStaticShape())
    return rewriter.notifyMatchFailure(
        diagnosticOp,
        "global initializer must be a statically shaped elements attribute");

  // cuda_tile.global semantics are linear and 1-D in the source dialect.
  if (initTy.getRank() != 1)
    return rewriter.notifyMatchFailure(
        diagnosticOp,
        "global initializer must be 1-D to match cuda_tile.global semantics");

  return MemRefType::get(initTy.getShape(),
                         converter.convertType(initTy.getElementType()));
}

struct MmaContractionSpec {
  AffineMap mapA;
  AffineMap mapB;
  AffineMap mapC;
  SmallVector<Attribute> iterTypes;
};

/// Build vector.contract indexing maps and iterator attributes for
/// matmul-style contractions used by both mmaf and mmai lowerings.
///
/// Supported ranks:
///   - rank 2 result: unbatched [M, N]
///   - rank 3 result: batched   [B, M, N]
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
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands().front());
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

/// Convert a unary source-based op to a math op that takes no FastMath flags,
/// preserving `flush_to_zero` as `tir-dropped-flush-to-zero` when set.
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
/// (allow approximate functions) FastMath flag. Other modes, and all modes
/// under `drop-rounding-modes`, are preserved as `tir-dropped-rounding`. When
/// `PreserveFtz` is set, `flush_to_zero` is preserved as
/// `tir-dropped-flush-to-zero`.
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

/// Convert float binary ops to arith float ops.
///
/// `rounding<nearest_even>` is the default arith semantics and the directed
/// modes map to the op's `roundingmode`. `ApproxFlag`, when not `none`,
/// represents `rounding<approx>` through that FastMath flag (e.g. `arcp` for
/// divf). Any other rounding mode, and every mode under `drop-rounding-modes`,
/// is preserved as `tir-dropped-rounding`. `flush_to_zero` has no arith
/// equivalent and is preserved as `tir-dropped-flush-to-zero`.
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

/// Convert integer binary ops that carry overflow flags (addi, subi, shli).
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

/// Convert signedness-directed casts that also require an exact rounding mode.
///
/// Used by:
///   - cuda_tile.ftoi  -> arith.fptosi / arith.fptoui
///   - cuda_tile.itof  -> arith.sitofp / arith.uitofp
///
/// The cast is emitted regardless of the source rounding mode; a mode differing
/// from `ExpectedRounding` is preserved as `tir-dropped-rounding`.
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

/// Group `newValues` by the number of types each of `origTypes` converts to,
/// e.g. to replace the results of an op whose token results were dropped.
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

/// Base for patterns of ops with token operands. Tokens convert to no values
/// (see the type converter), so their operands are null in the 1:1 adaptor.
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

/// Layout of the six trailing launch-coordinate arguments appended when
/// `append-grid-args=true` (see ConvertEntry). They appear in
/// the order: tile block id x/y/z, then grid dim x/y/z.
struct AppendedGridArgLayout {
  /// Total number of trailing launch-coordinate arguments.
  static constexpr unsigned kNumArgs = 6;
  /// Offset of the tile-block-id triple from the start of the launch
  /// coordinates.
  static constexpr unsigned kBlockIdBase = 0;
  /// Offset of the grid-dimension triple from the start of the launch
  /// coordinates.
  static constexpr unsigned kGridDimBase = 3;

  /// Look up the appended function-argument index carrying the value for the
  /// query
  /// whose triple starts at offset `argBase` (kBlockIdBase or kGridDimBase)
  /// along `dim`.
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

/// Convert a cuda_tile op that returns three i32 values (one per grid
/// dimension) into the launch coordinates for the active target.
///
///   - GPU: three GPU dimension-query ops (x, y, z), e.g.
///     get_tile_block_id -> gpu.block_id and get_num_tile_blocks ->
///     gpu.grid_dim.
///   - appendGridArgs=true: read matching trailing function arguments,
///     whose indices are looked up via AppendedGridArgLayout starting at
///     `ArgBase`.
///   - appendGridArgs=false: lower to gpu dimension-query ops only when
///     `target=gpu`; otherwise fail conversion.
///
/// Each result is cast to the converted result type as needed.
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

    // When requested, source launch coordinates from trailing function
    // arguments (of any function-like parent, e.g. func.func or gpu.func).
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

/// Convert cuda_tile.maxf/minf based on propagate_nan.
///
/// propagate_nan dispatches to arith.maximumf/minimumf (NaN propagating) vs
/// arith.maxnumf/minnumf (NaN suppressing). flush_to_zero has no arith
/// equivalent and is preserved as `tir-dropped-flush-to-zero`.
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

/// Common pre-flight checks for cuda_tile.reduce/scan lowerings.
///
/// Extracts the combining kind from the body, validates that the op has a
/// single operand and a vector-typed converted source, and returns the
/// identity attribute. On failure, calls `notifyMatchFailure` with an
/// appropriate reason.
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

/// Validate shared load/store_tko constraints before lowering.
///
/// load/store_tko lower to *non-atomic* memref.load/memref.store, which provide
/// no ordering guarantees. Accepting anything stronger than `weak` (e.g.
/// acquire/release) would therefore silently weaken the program's semantics, so
/// such ordering is rejected rather than dropped. This differs deliberately
/// from atomic_rmw_tko, whose target op (memref.atomic_rmw) is acq_rel and thus
/// always at least as strong as the requested ordering, allowing it to accept
/// and merely annotate the dropped ordering/scope.
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

/// Keeps the information needed by vector.transfer_read / transfer_write to
/// access a memref through a tile view (partition_view or strided_view).
struct TransferViewAccessPlan {
  ViewInfo viewInfo;
  SmallVector<Value> memrefIndices;
  AffineMap permutationMap;
  SmallVector<bool> inBounds;
};

/// Build a TransferViewAccessPlan for a load_view_tko or store_view_tko.
///
/// Translate tile-view indices into the concrete memref indices, permutation
/// map, and in-bounds flags required by vector.transfer_read/write.
/// 1. Cast each tile-level index to `index` and scale by the view's per-dim
///    base advance (`viewStrides[i]`): tile_shape for partition_view,
///    traversal_strides for strided_view.
/// 2. Place the scaled index into the memref-dimension slot given by dim_map.
/// 3. Build a permutation_map whose i-th result references memref dimension
///    dim_map[i], so vector dim i reads/writes that tensor dimension.
/// 4. Set inBounds[i] = true only when the tensor extent is static and the
///    last in-bounds tile base plus the tile extent still fits within it (i.e.
///    no tile, including the trailing one, ever runs past the tensor extent).
///    For partition_view this reduces to "extent divisible by tile extent";
///    for strided_view it also rejects overlapping/gapped layouts whose edge
///    tiles spill out of bounds, deferring those lanes to the masked path.
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

  // Build memref indices in tensor-dimension order. View indices are unsigned.
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
    // Number of in-bounds tile bases along this dimension (partial edge tiles
    // are included), then check whether the trailing tile fits entirely.
    int64_t stride = viewInfo.viewStrides[i];
    int64_t numTiles = (ext + stride - 1) / stride;
    int64_t lastBase = numTiles > 0 ? (numTiles - 1) * stride : 0;
    inBounds[i] = (lastBase + viewInfo.tileShape[i] <= ext);
  }

  return TransferViewAccessPlan{std::move(viewInfo), std::move(memrefIndices),
                                permutationMap, std::move(inBounds)};
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//
using ConvertAddF = ConvertBinaryFloatOp<cuda_tile::AddFOp, arith::AddFOp>;

using ConvertAddI =
    ConvertBinaryLhsRhsWithOverflowOp<cuda_tile::AddIOp, arith::AddIOp>;

/// Convert cuda_tile.alloca to memref.alloca (+ memref.cast).
///
/// The op allocates `num_elem` elements of the pointee type with automatic
/// (block-scoped) lifetime and yields a scalar pointer. The pass models a
/// scalar `tile<ptr<T>>` as an unranked `memref<*xT>`, so we allocate a ranked
/// `memref<num_elem x T>` on the stack and cast it to the unranked result type.
///
/// Attribute mapping:
///   - `num_elem`  -> the (single) static dimension of the ranked memref.
///   - `alignment` -> memref.alloca's `alignment` (a non-zero power of two,
///                    guaranteed by the source verifier and required as such by
///                    memref.alloca).
///   - `global`    -> marks the address as shareable across tile threads. The
///                    unranked memref pointer model carries no memory space
///                    able to express that sharing, so the conversion bails
///                    when it is set.
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

    // The source verifier guarantees alignment is a non-zero power of two,
    // which is exactly what memref.alloca requires.
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

/// Materialize the value that feeds a broadcast into a ranked pointer tile
/// (`vector<...xindex>` of per-lane offsets).
///
/// A scalar pointer converts to an unranked memref, which carries no per-lane
/// offset; the load/store lowerings recover the base pointer from it directly,
/// so the lane offsets start at 0. Any other source already is the lane value.
static Value getPointerTileLaneSource(OpBuilder &builder, Location loc,
                                      Value source) {
  if (isa<UnrankedMemRefType>(source.getType()))
    return arith::ConstantIndexOp::create(builder, loc, 0);
  return source;
}

/// Convert cuda_tile.broadcast to vector.broadcast.
///
/// Both ops expand size-1 dimensions by duplicating data along them while
/// preserving the rank.  cuda_tile.broadcast requires same rank for source
/// and result and only stretches dimensions of size 1.  vector.broadcast has
/// the same "dim-1 stretching" semantics for trailing dimensions when the
/// source and result have equal rank, so the lowering is a direct 1:1 map.
struct ConvertBroadcast : public OpConversionPattern<cuda_tile::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "cannot convert result type");

    Value source = adaptor.getSource();
    if (auto dstVecTy = dyn_cast<VectorType>(resultTy)) {
      // Handles both data tiles (vector<NxMxelemTy>) and ranked pointer tiles
      // (vector<NxMxindex>), whose scalar-pointer source needs a lane offset.
      rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
          op, dstVecTy,
          getPointerTileLaneSource(rewriter, op.getLoc(), source));
      return success();
    }
    if (source.getType() == resultTy) {
      rewriter.replaceOp(op, source);
      return success();
    }
    return rewriter.notifyMatchFailure(op, "unsupported broadcast result type");
  }
};

/// Convert cuda_tile.cat to vector.insert_strided_slice.
///
///   1. Create a poison/undef vector of the result type (ub.poison) — its
///      elements will be fully overwritten by the two inserts.
///   2. vector.insert_strided_slice lhs into the result at all-zero offsets.
///   3. vector.insert_strided_slice rhs into the result at offset
///      [0,...,lhs.shape[d],...,0] (only the concat-dim offset is non-zero).
///
///   All sizes and offsets are statically known from the tile types, which is
///   exactly what vector.insert_strided_slice requires (I64ArrayAttr offsets,
///   unit strides).
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

    // Start with an undefined result vector (all elements will be written).
    Value dest = ub::PoisonOp::create(rewriter, loc, dstVecTy);

    // Offsets for lhs: all zeros.
    SmallVector<int64_t> lhsOffsets(rank, 0);
    // Strides: all ones (required by vector.insert_strided_slice).
    SmallVector<int64_t> strides(rank, 1);

    Value withLhs = vector::InsertStridedSliceOp::create(
        rewriter, loc, lhs, dest, lhsOffsets, strides);

    // Offsets for rhs: zero everywhere except concatDim = lhs.shape[concatDim].
    SmallVector<int64_t> rhsOffsets(rank, 0);
    rhsOffsets[concatDim] = lhsVecTy.getDimSize(concatDim);

    rewriter.replaceOpWithNewOp<vector::InsertStridedSliceOp>(
        op, rhs, withLhs, rhsOffsets, strides);
    return success();
  }
};

/// Map a cuda_tile comparison predicate to the arith.cmpf predicate with the
/// matching ordering (`cuda_tile::ComparisonOrdering` is either ordered or
/// unordered).
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

/// Convert cuda_tile.constant to an arith.constant of the converted type:
/// a scalar attribute for rank-0 tiles, a DenseElementsAttr of the target
/// vector type otherwise.
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

/// Convert cuda_tile.divf to arith.divf.
///
/// `rounding<approx>` is represented by the `arcp` (allow reciprocal) FastMath
/// flag. The default `rounding<nearest_even>` is represented by arith.divf's
/// default semantics. Other rounding modes are preserved as
/// `tir-dropped-rounding`.
/// `flush_to_zero` has no arith equivalent and is preserved on the result as
/// `tir-dropped-flush-to-zero`.
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

/// Convert cuda_tile.entry to gpu.func (gpu target) or func.func (cpu target).
///
/// The function signature is derived by applying the type converter to each
/// entry argument type (e.g. `tile<ptr<T>>` -> `memref<*xT>`, `tile<i32>` ->
/// `i32`). The entry body is signature-converted in place and then moved into
/// the new function body.
///
/// When `append-grid-args=true`, the function additionally receives six
/// trailing `i32` arguments carrying launch coordinates: the three tile block
/// ids (x, y, z) followed by the three grid dimensions (x, y, z). These follow
/// the converted entry arguments.
///
/// `optimization_hints`, when present, is preserved on the produced function as
/// the discardable attribute `tir-dropped-optimization-hints`.
///
/// When `known-block-size` provides three values, the produced gpu.func carries
/// them as the `known_block_size` attribute.
struct ConvertEntry : public OptionsPattern<cuda_tile::EntryOp> {
  using OptionsPattern::OptionsPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::EntryOp entryOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = entryOp.getContext();
    Location loc = entryOp.getLoc();
    Block *entryBlock = &entryOp.getBody().front();
    unsigned numArgs = entryBlock->getNumArguments();

    // Compute the function arg types and prepare a signature conversion for
    // the entry block.
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

    // Optionally append launch coordinates as six trailing i32 arguments
    // (block id x/y/z then grid dim x/y/z), used by dim-query lowerings.
    if (options.appendGridArgs) {
      SmallVector<Type> launchArgTypes(AppendedGridArgLayout::kNumArgs,
                                       IntegerType::get(ctx, 32));
      funcArgTypes.append(launchArgTypes.begin(), launchArgTypes.end());
      sigConv.addInputs(launchArgTypes);
    }

    auto funcType = FunctionType::get(ctx, funcArgTypes, {});

    // Convert the entry block's arg types; this replaces the block with a
    // new one having the converted signature and rewires uses via source
    // materializations.
    FailureOr<Block *> convertedBlock =
        rewriter.convertRegionTypes(&entryOp.getBody(), *tc, &sigConv);
    if (failed(convertedBlock))
      return failure();

    if (options.target == TileIRTarget::GPU) {
      // GPU: lower to a gpu.func kernel and merge the converted body into its
      // (auto-created) entry block.
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
      // CPU: lower to a plain func.func and move the converted body region in.
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
///
/// `flush_to_zero` is not representable in math FastMath flags and is preserved
/// on the result as `tir-dropped-flush-to-zero` when set.
using ConvertExp2 = ConvertUnaryFlushToZeroOp<cuda_tile::Exp2Op, math::Exp2Op>;

/// Convert cuda_tile.extract to vector.shape_cast + vector.transpose +
/// vector.extract.
///
/// Source semantics (from Ops.td):
///   For `extract %t[%i_0, ..., %i_{n-1}] : tile<D_0 x ... x D_{n-1} x T>
///                                       -> tile<R_0 x ... x R_{n-1} x T>`,
///   each R_k evenly divides D_k.  With S_k = D_k / R_k slices per axis,
///       result[a_0, ..., a_{n-1}] = source[i_0*R_0 + a_0, ...,
///                                          i_{n-1}*R_{n-1} + a_{n-1}].
///   The $indices are interpreted as unsigned i32; OOB is UB.
///
/// Lowering (dynamic indices preclude vector.extract_strided_slice which
/// requires static offsets):
///   1. shape_cast <D_0 x ... x D_{n-1}>
///                 -> <S_0 x R_0 x S_1 x R_1 x ... x S_{n-1} x R_{n-1}>.
///      Row-major linearization gives position [s_0,r_0,...,s_k,r_k] the same
///      linear index as source[s_0*R_0 + r_0, ..., s_k*R_k + r_k] because
///      D_k = S_k * R_k.
///   2. transpose with permutation [0,2,...,2(n-1), 1,3,...,2(n-1)+1] to
///      group slice-index dims first:
///          <S_0 x ... x S_{n-1} x R_0 x ... x R_{n-1}>.
///   3. vector.extract at [i_0, ..., i_{n-1}] yields the <R_0 x ... x R_{n-1}>
///      subvector matching the source semantics.
///
/// Special cases:
///   - rank-1 source: interleaved shape <S_0, R_0> already has the slice dim
///     leading, the permutation is the identity, so the transpose is skipped.
///   - source type == result type (scalar tile or all S_k == 1): forward the
///     source directly. This also covers the scalar tile<T> case where the
///     converted type is not a VectorType.
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

    // Trivial case: source and result types coincide (all S_k == 1, or a
    // scalar tile<T> that converts to a non-vector type).
    if (source.getType() == resultTy) {
      rewriter.replaceOp(op, source);
      return success();
    }

    auto srcVecTy = cast<VectorType>(source.getType());
    auto dstVecTy = cast<VectorType>(resultTy);
    int64_t rank = srcVecTy.getRank();

    // Step 1: Build interleaved reshape <S_0, R_0, S_1, R_1, ...>.
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

    // Step 2: Transpose slice dims to the front.  For rank 1 the permutation
    // is [0,1] (identity), so we skip the transpose.
    Value extractSource = reshaped;
    if (rank > 1) {
      SmallVector<int64_t> perm;
      perm.reserve(2 * rank);
      for (int64_t k = 0; k < rank; ++k)
        perm.push_back(2 * k); // slice dims first
      for (int64_t k = 0; k < rank; ++k)
        perm.push_back(2 * k + 1); // result dims trailing
      extractSource =
          vector::TransposeOp::create(rewriter, loc, reshaped, perm);
    }

    // Step 3: Cast the unsigned i32 slice indices to `index` (the spec
    // declares $indices as unsigned, so use index_castui) and emit
    // vector.extract.
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

/// Convert cuda_tile.fma to math.fma.
///
/// math.fma rounds once to nearest even. Other rounding modes, and all modes
/// under `drop-rounding-modes`, are preserved as `tir-dropped-rounding`;
/// `flush_to_zero` is preserved as `tir-dropped-flush-to-zero`.
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

    // Convert region types
    if (failed(
            rewriter.convertRegionTypes(&op.getRegion(), *getTypeConverter())))
      return failure();

    // Merge old body into new body
    Block *oldBody = op.getBody();
    Block *newBody = newForOp.getBody();

    // Remove auto-generated yield in new body
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

/// Convert cuda_tile.ftof to arith.extf / arith.truncf / arith.convertf.
///
///   - Widening uses arith.extf. It is exact, so the rounding mode does not
///     matter and is only preserved under `drop-rounding-modes`.
///   - Narrowing uses arith.truncf and conversions between formats of the same
///     width use arith.convertf. Both carry a rounding-mode attribute, so the
///     source rounding mode is mapped onto it when possible and preserved as
///     `tir-dropped-rounding` otherwise.
///
/// Works for both scalar float and vector<float> types.
struct ConvertFToF : public OptionsPattern<cuda_tile::FToFOp> {
  using OptionsPattern::OptionsPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::FToFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getConvertedResultTypeOrFail(
        op, getTypeConverter(), rewriter, "cannot convert ftof result type");
    if (failed(resultTy))
      return failure();

    // After type conversion the source and result types may coincide -- e.g.
    // `ftof f32 -> tf32` on the CPU target, where tf32 lowers to f32.  Such a
    // cast is a no-op, so forward the converted source value (the rounding mode
    // is irrelevant).
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

/// Convert cuda_tile.get_global to memref.get_global (+ memref.cast).
///
///   1. Resolve the referenced global symbol, accepting either cuda_tile.global
///      or an already-converted memref.global.
///   2. Emit memref.get_global with the ranked memref type derived from that
///      global initializer.
///   3. Cast to the converted result type (typically memref<*xT>) so this
///      pass's pointer model remains uniform (tile<ptr<T>> -> memref<*xT>).
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

/// Convert cuda_tile.get_index_space_shape.
///
/// For a tile view with tile dims mapped to tensor dims via dim_map,
///   index_space_shape[i] = ceildiv(tensor_shape[dimMap[i]], viewStrides[i]),
/// where viewStrides[i] is the per-dim base advance: tile_shape[i] for
/// partition_view, traversal_strides[i] for strided_view. Partial edge tiles
/// are included in the count, which the ceildiv naturally accounts for.
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

    // For each tile dimension i:
    // - The corresponding tensor_view dimension is dimMap[i]
    // - index_space_dim_i = ceildiv(memref.dim(dimMap[i]), viewStrides[i])
    // When the memref dimension is statically known, fold to a constant.
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
        // Static dimension: compute ceildiv at compile time.
        int64_t numTiles = (dimSize + stride - 1) / stride;
        Value cst = arith::ConstantIndexOp::create(rewriter, loc, numTiles);
        castedResult = castValueToType(rewriter, loc, cst, resultTy);
      } else {
        // Dynamic dimension: emit memref.dim + ceildivui.
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

/// Convert cuda_tile.get_tensor_shape.
///
/// For a converted tensor_view memref, each result is the extent of the
/// corresponding memref dimension. Static extents are folded to constants;
/// dynamic extents are queried via memref.dim.
///
/// Source semantics specify that these values are interpreted as unsigned
/// integers. When the target result type is integer, use index_castui.
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

/// Convert cuda_tile.global to memref.global.
///
///   1. Derive a ranked static memref type from the initializer shape/element.
///   2. Emit memref.global with the same symbol name and initial value.
///   3. Preserve alignment when non-zero; omit it otherwise.
///
/// Note: cuda_tile.global is mutable, so we do not set memref.global
/// `constant`.
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

/// Convert cuda_tile.iota to vector.step + arith.index_castui.
///
///   1. Emit vector.step : vector<nxindex> to materialize [0..n-1].
///   2. Convert lanes to the destination integer element type with
///      arith.index_castui to preserve unsigned interpretation.
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

/// Convert cuda_tile.load_view_tko to vector.transfer_read.
///
/// The indices, permutation map and in-bounds flags come from
/// buildTransferViewAccessPlan. Out-of-bounds lanes read the view's
/// `padding_value`, or poison when it has none.
///
/// Restrictions (return notifyMatchFailure on violation):
///   - Only `weak` memory_ordering_semantics is supported.
///   - `memory_scope` is not supported.
///
/// `optimization_hints`, when present, is preserved on the produced
/// vector.transfer_read as the discardable attribute
/// `tir-dropped-optimization-hints`.
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

/// Recover the runtime offset (the descriptor's offset field) carried by an
/// unranked converted pointer value (`memref<*xT>`).
///
/// `memref.reinterpret_cast` expresses an offset that is *absolute* to the
/// underlying buffer, so any pattern that needs to advance such a pointer
/// (chained scalar `offset` ops, or a `make_tensor_view` whose base was
/// pre-shifted) must read the base's current offset and add to it rather than
/// overwrite it. The unranked base is cast to a ranked memref with a dynamic
/// offset / unit stride, then its offset field is read via
/// memref.extract_strided_metadata.
///
/// Kernel-pointer function arguments are a special case: by the calling
/// convention of the converted kernels (see Passes.td), they are bare pointers
/// with a zero descriptor offset, so there is nothing to recover. We
/// short-circuit them to a static `0` and emit *no* IR. This also keeps every
/// use of the argument a reinterpret_cast with a static zero offset, which
/// ConvertMemrefArgsToRankedMemref requires to promote the argument.
///
/// The short-circuit is restricted to entry-block arguments of function-like
/// ops. A pointer carried as a loop/region iter-arg (e.g. an `scf.for` body
/// argument) may hold a non-zero, pre-shifted offset, so it must go through the
/// metadata path instead of being assumed zero.
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

/// Reinterpret an unranked converted pointer (`memref<*xT>`) as a rank-0 memref
/// the scalar load/store/atomic patterns can address, *preserving* the
/// descriptor's absolute offset.
///
/// `memref.reinterpret_cast` offsets are absolute to the underlying buffer, so
/// reinterpreting with a literal offset of 0 would reset a pre-shifted pointer
/// (e.g. the result of a scalar `offset` op) back to the buffer start and read
/// the wrong element. We recover the base's current offset and re-apply it,
/// mirroring how `make_tensor_view` and scalar `offset` preserve offsets.
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

/// Convert cuda_tile.make_tensor_view to memref.reinterpret_cast.
///
/// The base operand is a scalar `tile<ptr<T>>`, which the type converter maps
/// to `memref<*xT>`. The tensor_view result type maps to a ranked memref.
/// Static shape and stride entries come from the tensor_view type.
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

    // make_tensor_view reshapes the buffer at the base pointer's current
    // location, so it must preserve whatever absolute offset the base memref
    // descriptor carries. Recover it unconditionally rather than matching only
    // a specific producer shape (e.g. direct scalar `offset`).
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

/// Convert cuda_tile.mmaf to vector.contract (matmul-style contraction).
///
///   1. Convert result tile type to a vector type
///   2. Build affine indexing maps and iterator types depending on the rank.
///      - Unbatched (3 iterators, d0=m, d1=n, d2=k):
///          lhs (d0,d2), rhs (d2,d1), acc (d0,d1); iters [par,par,red].
///      - Batched (4 iterators, d0=b, d1=m, d2=n, d3=k):
///          lhs (d0,d1,d3), rhs (d0,d3,d2), acc (d0,d1,d2);
///          iters [par,par,par,red].
///   3. Replace with vector.contract(lhs, rhs, acc) with combining kind = add.
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

    // Explicit combining kind = add (mmaf is multiply-accumulate).
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        op, adaptor.getLhs(), adaptor.getRhs(), adaptor.getAcc(),
        rewriter.getAffineMapArrayAttr({spec->mapA, spec->mapB, spec->mapC}),
        rewriter.getArrayAttr(spec->iterTypes), vector::CombiningKind::ADD);
    return success();
  }
};

/// Convert cuda_tile.mmai to vector.contract (matmul-style contraction).
///
/// Lowering mirrors mmaf and uses the same indexing-map / iterator builder.
/// vector.contract promotes narrower integer operands by sign extension, so
/// operands are first extended to the accumulator element type according to
/// their signedness.
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

/// Convert cuda_tile.module by moving its body contents.
///
/// For the GPU target the body is moved into a new gpu.module of the same name.
/// For the CPU target the cuda_tile.module is dissolved: its contents are
/// inlined into the enclosing module (the builtin.module the pass runs on) and
/// the cuda_tile.module wrapper is erased.
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

      // Move ops from cuda_tile.module body into gpu.module body, inserting
      // before the gpu.module_end terminator if present.
      if (newBody->mightHaveTerminator())
        rewriter.inlineBlockBefore(oldBody, newBody->getTerminator());
      else
        rewriter.inlineBlockBefore(oldBody, newBody, newBody->end());
    } else {
      // Dissolve the module into the enclosing module by inlining its body
      // ops right before the cuda_tile.module op in its parent block.
      rewriter.inlineBlockBefore(&tileirMod.getBody().front(), tileirMod);
    }

    rewriter.eraseOp(tileirMod);
    return success();
  }
};

using ConvertMulF = ConvertBinaryFloatOp<cuda_tile::MulFOp, arith::MulFOp>;

/// Convert cuda_tile.mulhii by taking the high part of mului_extended.
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

/// Convert scalar cuda_tile.offset on pointer tiles to a memref view.
///
/// Pointer model in this pass: tile<ptr<T>> -> memref<*xT>. For
/// `offset(ptr, off)` with a scalar `off`, build a rank-1 memref view with
/// dynamic offset and unit size/stride, then cast back to memref<*xT>. The
/// result's descriptor carries the accumulated offset, which consumers recover
/// with recoverUnrankedPtrOffset.
struct ConvertOffsetScalarPtr
    : public OpConversionPattern<cuda_tile::OffsetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::OffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto offTy = dyn_cast<cuda_tile::TileType>(op.getOffset().getType());
    if (!offTy || !offTy.getShape().empty())
      return rewriter.notifyMatchFailure(
          op, "only scalar pointer offsets are supported");

    // Restrict to the pass's pointer model: tile<ptr<T>> -> memref<*xT>.
    // Lowering through a ranked source memref would be unsafe because
    // memref.reinterpret_cast's offset is absolute to the underlying buffer
    // and would silently discard any pre-existing offset / strided layout on
    // the source view.
    auto srcUnranked = dyn_cast<UnrankedMemRefType>(adaptor.getPtr().getType());
    auto dstUnranked = dyn_cast_or_null<UnrankedMemRefType>(
        getTypeConverter()->convertType(op.getType()));
    if (!srcUnranked || !dstUnranked)
      return rewriter.notifyMatchFailure(
          op, "expected unranked memref pointer model on both source and "
              "result");

    Type elemTy = srcUnranked.getElementType();
    Attribute memSpace = srcUnranked.getMemorySpace();
    if (dstUnranked.getElementType() != elemTy)
      return rewriter.notifyMatchFailure(
          op, "source and result element types must match");
    if (dstUnranked.getMemorySpace() != memSpace)
      return rewriter.notifyMatchFailure(
          op, "source and result memory spaces must match");

    // Offset element type must be an integer; reject pointer/float scalars.
    if (!isa<IntegerType>(offTy.getElementType()))
      return rewriter.notifyMatchFailure(op,
                                         "offset element type must be integer");

    Value offIdx = castValueToType(rewriter, op.getLoc(), adaptor.getOffset(),
                                   rewriter.getIndexType());
    if (!offIdx)
      return rewriter.notifyMatchFailure(
          op, "offset addend could not be converted to index");

    // reinterpret_cast's offset is absolute to the underlying buffer. Always
    // accumulate the source pointer's current descriptor offset so semantically
    // equivalent sources (direct offset, ptr_to_ptr chain, block arg, etc.)
    // are handled uniformly. A raw kernel-pointer argument carries a static
    // zero offset, in which case the addend alone is the absolute offset.
    OpFoldResult srcOff =
        recoverUnrankedPtrOffset(rewriter, op.getLoc(), adaptor.getPtr());
    OpFoldResult totalOff = offIdx;
    if (!isZeroInteger(srcOff))
      totalOff = arith::AddIOp::create(rewriter, op.getLoc(),
                                       getValueOrCreateConstantIndexOp(
                                           rewriter, op.getLoc(), srcOff),
                                       offIdx)
                     .getResult();

    auto rank1ViewTy =
        get1DDynamicOffsetMemRefType(elemTy, /*size=*/1, memSpace);

    auto rc = memref::ReinterpretCastOp::create(
        rewriter, op.getLoc(), rank1ViewTy, adaptor.getPtr(), totalOff,
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(1)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(1)});

    Value result = rc.getResult();
    if (result.getType() != dstUnranked)
      result =
          memref::CastOp::create(rewriter, op.getLoc(), dstUnranked, result);

    rewriter.replaceOp(op, result);
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

/// Convert scalar (rank-0) cuda_tile.load_ptr_tko on a `tile<ptr<T>>` to a
/// `memref.reinterpret_cast` + `memref.load`. The source `tile<ptr<T>>` is
/// converted to `memref<*xT>`; a rank-0 reinterpret_cast that keeps the
/// descriptor offset recovers the scalar memref the load reads from.
///
/// `optimization_hints`, when present, is preserved on the produced memref.load
/// as the discardable attribute `tir-dropped-optimization-hints`.
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

/// Convert scalar (rank-0) cuda_tile.store_ptr_tko on a `tile<ptr<T>>` to a
/// `memref.reinterpret_cast` + `memref.store`. Mirrors ConvertLoadPtrTkoScalar.
///
/// `optimization_hints`, when present, is preserved on the produced
/// memref.store as the discardable attribute `tir-dropped-optimization-hints`.
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

/// Walk the original cuda_tile ptr value backward through offset/broadcast/
/// reshape/assume to find the scalar base pointer (tile<ptr<T>>).
static Value findOriginalBasePtr(Value ptrTile) {
  while (ptrTile) {
    if (auto ty = dyn_cast<cuda_tile::TileType>(ptrTile.getType())) {
      if (ty.getShape().empty())
        break;
    }
    if (auto assume = ptrTile.getDefiningOp<cuda_tile::AssumeOp>()) {
      ptrTile = assume.getValue();
      continue;
    }
    if (auto bcast = ptrTile.getDefiningOp<cuda_tile::BroadcastOp>()) {
      ptrTile = bcast.getSource();
      continue;
    }
    if (auto rs = ptrTile.getDefiningOp<cuda_tile::ReshapeOp>()) {
      ptrTile = rs.getSource();
      continue;
    }
    if (auto off = ptrTile.getDefiningOp<cuda_tile::OffsetOp>()) {
      ptrTile = off.getPtr();
      continue;
    }
    break;
  }
  return ptrTile;
}

/// Convert ranked (non-scalar) cuda_tile.offset on pointer tiles.
///
/// After type conversion, the pointer operand is vector<...xindex> (per-element
/// byte offsets from buffer start) and the integer offset is vector<...xiN>.
/// The result is: element-wise (ptr_offsets + index_cast(int_offsets)).
struct ConvertOffsetRanked : public OpConversionPattern<cuda_tile::OffsetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::OffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTileTy = cast<cuda_tile::TileType>(op.getType());
    if (resTileTy.getShape().empty())
      return rewriter.notifyMatchFailure(op, "scalar offset handled elsewhere");

    Value ptrVec = adaptor.getPtr();
    Value offVec = adaptor.getOffset();

    auto ptrVecTy = dyn_cast<VectorType>(ptrVec.getType());
    if (!ptrVecTy || !isa<IndexType>(ptrVecTy.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "expected vector<...xindex> for ranked pointer");

    // Cast offset to index type.
    auto offVecTy = cast<VectorType>(offVec.getType());
    if (!isa<IndexType>(offVecTy.getElementType())) {
      auto idxVecTy =
          VectorType::get(offVecTy.getShape(), rewriter.getIndexType());
      offVec =
          arith::IndexCastOp::create(rewriter, op.getLoc(), idxVecTy, offVec);
    }

    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, ptrVec, offVec);
    return success();
  }
};

/// Derive the flat base memref and the mask of a pointer tile access.
static LogicalResult deriveAccessMemRefAndMask(
    Operation *op, Value origPtr, Value cvtPtr, Value origMask, Value cvtMask,
    Type elemTy, ArrayRef<int64_t> shape, ConversionPatternRewriter &rewriter,
    Value &baseMemref, Value &mask) {
  Location loc = op->getLoc();

  auto ptrVecTy = dyn_cast<VectorType>(cvtPtr.getType());
  if (!ptrVecTy || !isa<IndexType>(ptrVecTy.getElementType()))
    return rewriter.notifyMatchFailure(
        op, "expected vector<...xindex> for ranked pointer");

  // Find the base memref by tracing the original pointer chain.
  Value origBase = findOriginalBasePtr(origPtr);
  if (!origBase || !isa<cuda_tile::TileType>(origBase.getType()))
    return rewriter.notifyMatchFailure(op, "cannot find scalar base pointer");
  auto baseTileTy = cast<cuda_tile::TileType>(origBase.getType());
  if (!baseTileTy.getShape().empty())
    return rewriter.notifyMatchFailure(op, "base is not scalar");

  // Get the converted base value.
  Value scalarBase = rewriter.getRemappedValue(origBase);
  if (!scalarBase)
    return rewriter.notifyMatchFailure(op, "cannot find converted base memref");
  if (!isa<BaseMemRefType>(scalarBase.getType()))
    return rewriter.notifyMatchFailure(op, "base is not a memref");

  // Cast to memref<?xelemTy> for gather/scatter. The flat type must keep a
  // dynamic offset: the scalar base may be a `memref.reinterpret_cast` that
  // carries a non-zero offset (e.g. the per-row `h_in*W` of a pooling window).
  // A plain `memref<?xelemTy>` has a *static* offset of 0, which would make the
  // gather/scatter address computation (getStridedElementPtr) ignore the
  // descriptor's offset field and drop the row stride entirely.
  auto flatMemTy =
      get1DDynamicOffsetMemRefType(elemTy, /*size=*/ShapedType::kDynamic);
  baseMemref = memref::CastOp::create(rewriter, loc, flatMemTy, scalarBase);

  // Mask.
  if (origMask) {
    mask = cvtMask;
  } else {
    auto maskTy = VectorType::get(shape, rewriter.getI1Type());
    Value trueVal = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
    mask = vector::BroadcastOp::create(rewriter, loc, maskTy, trueVal);
  }

  return success();
}

/// Convert ranked cuda_tile.load_ptr_tko to vector.gather.
///
/// The pointer tile (vector<...xindex>) holds per-element offsets from the
/// buffer base. `optimization_hints`, when present, is preserved on the gather
/// as the discardable attribute `tir-dropped-optimization-hints`.
struct ConvertLoadPtrTkoRanked
    : public TokenDroppingPattern<cuda_tile::LoadPtrTkoOp> {
  using TokenDroppingPattern::TokenDroppingPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::LoadPtrTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileTy = cast<cuda_tile::TileType>(op.getResult().getType());
    if (tileTy.getShape().empty())
      return rewriter.notifyMatchFailure(op, "scalar load handled elsewhere");

    if (failed(checkCommonTkoGuards(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    auto resultVecTy =
        cast<VectorType>(getTypeConverter()->convertType(tileTy));

    Value baseMemref, mask;
    if (failed(deriveAccessMemRefAndMask(
            op, op.getSource(), adaptor.getSource(), op.getMask(),
            adaptor.getMask(), resultVecTy.getElementType(), tileTy.getShape(),
            rewriter, baseMemref, mask)))
      return failure();

    // Passthrough.
    Value passThru;
    if (op.getPaddingValue()) {
      passThru = adaptor.getPaddingValue();
    } else {
      auto zeroAttr = rewriter.getZeroAttr(resultVecTy);
      passThru =
          arith::ConstantOp::create(rewriter, loc, resultVecTy, zeroAttr);
    }

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto gatherOp = vector::GatherOp::create(
        rewriter, loc, resultVecTy, baseMemref, ValueRange{c0},
        adaptor.getSource(), mask, passThru);
    preserveDroppedOptHints(op, gatherOp);
    rewriter.replaceOp(op, {gatherOp.getResult(), Value()});
    return success();
  }
};

/// Convert ranked cuda_tile.store_ptr_tko to vector.scatter.
///
/// `optimization_hints`, when present, is preserved on the produced
/// vector.scatter as the discardable attribute
/// `tir-dropped-optimization-hints`.
struct ConvertStorePtrTkoRanked
    : public TokenDroppingPattern<cuda_tile::StorePtrTkoOp> {
  using TokenDroppingPattern::TokenDroppingPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::StorePtrTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto valTileTy = cast<cuda_tile::TileType>(op.getValue().getType());
    if (valTileTy.getShape().empty())
      return rewriter.notifyMatchFailure(op, "scalar store handled elsewhere");

    if (failed(checkCommonTkoGuards(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    Value valVec = adaptor.getValue();

    Value baseMemref, mask;
    if (failed(deriveAccessMemRefAndMask(
            op, op.getDestination(), adaptor.getDestination(), op.getMask(),
            adaptor.getMask(),
            cast<VectorType>(valVec.getType()).getElementType(),
            valTileTy.getShape(), rewriter, baseMemref, mask)))
      return failure();

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto scatterOp = vector::ScatterOp::create(
        rewriter, loc, /*resultType=*/Type(), baseMemref, ValueRange{c0},
        adaptor.getDestination(), mask, valVec);
    preserveDroppedOptHints(op, scatterOp);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Map a cuda_tile.atomic_rmw mode to the equivalent arith atomic_rmw kind.
///
/// MAX/MIN are the signed integer variants; UMAX/UMIN are the unsigned ones.
/// XCHG (unconditional swap) maps to `assign`. Every cuda_tile mode has an
/// arith equivalent, so this mapping is total.
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

/// Convert scalar (rank-0) cuda_tile.atomic_rmw_tko on a `tile<ptr<T>>` to a
/// `memref.reinterpret_cast` + `memref.atomic_rmw`. Mirrors
/// ConvertLoadPtrTkoScalar: the source `tile<ptr<T>>` is converted to
/// `memref<*xT>`; a rank-0 reinterpret_cast that keeps the descriptor offset
/// recovers the scalar memref the atomic operates on.
///
/// Both ops return the value read at the location before the update, so the
/// result maps directly. The `memory_ordering_semantics` and `memory_scope`
/// attributes have no representation on memref.atomic_rmw (which lowers to an
/// acq_rel LLVM atomicrmw with no scope). Because acq_rel is at least as strong
/// as any requested ordering, dropping the request is conservatively safe (it
/// can only over-synchronize, never under-synchronize); the original values are
/// preserved on the result as the discardable attributes
/// `tir-dropped-memory-ordering` and `tir-dropped-memory-scope`. This is why,
/// unlike the non-atomic load/store_tko lowerings (see checkCommonTkoGuards),
/// atomic_rmw_tko accepts any ordering rather than rejecting non-`weak`.
///
/// Higher-rank atomics are not lowered in this pass.
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
      // Only a statically-true scalar mask can be dropped here; a dynamic or
      // possibly-false mask would require a predicated atomic we cannot
      // represent. A rank-0 mask constant is always splat, so reading the
      // splat value is safe once we know it is a scalar constant.
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
///
/// Both ops reorder the dimensions of an N-D tensor/vector according to a
/// permutation array.  The only mechanical difference is the attribute type:
///   cuda_tile.permute uses DenseI32ArrayAttr,
///   vector.transpose  uses DenseI64ArrayAttr.
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

/// Convert cuda_tile.ptr_to_ptr using memref.cast when representable.
///
/// Pointer model in this pass: tile<ptr<T>> -> memref<*xT>. The pattern fails
/// if the conversion cannot be represented as a memref.cast.
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

/// Convert cuda_tile.reduce to vector.reduction (1D->scalar) or
/// vector.multi_reduction (ND->(N-1)D).
///
/// Only supports single-operand reductions where the body contains exactly
/// one recognized combining op.
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

      // Multi-dim case: vector.multi_reduction
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

      // 1D -> scalar case: vector.reduction
      Value acc =
          arith::ConstantOp::create(rewriter, loc, resultTy, identityAttr);
      rewriter.replaceOpWithNewOp<vector::ReductionOp>(op, kind, source, acc);
    }
    return success();
  }
};

/// Convert cuda_tile.reshape to vector.shape_cast / vector.broadcast /
/// vector.extract depending on source/result ranks.
//
///   - vector -> vector: vector.shape_cast
///   - scalar -> vector: vector.broadcast (scalar to single-element vector)
///   - vector -> scalar: vector.extract at [0,...,0]
///   - scalar -> scalar: identity
struct ConvertReshape : public OpConversionPattern<cuda_tile::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "cannot convert result type");

    Value source = adaptor.getSource();
    auto srcVecTy = dyn_cast<VectorType>(source.getType());
    auto dstVecTy = dyn_cast<VectorType>(resultTy);

    if (srcVecTy && dstVecTy) {
      rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, dstVecTy, source);
    } else if (!srcVecTy && dstVecTy) {
      // Also covers a scalar pointer reshaped to a ranked pointer tile
      // (e.g. tile<ptr<T>> -> tile<1x1xptr<T>> -> vector<1x1xindex>).
      rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
          op, dstVecTy,
          getPointerTileLaneSource(rewriter, op.getLoc(), source));
    } else if (srcVecTy && !dstVecTy) {
      SmallVector<int64_t> indices(srcVecTy.getRank(), 0);
      rewriter.replaceOpWithNewOp<vector::ExtractOp>(op, source, indices);
    } else {
      rewriter.replaceOp(op, source);
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
///
/// `flush_to_zero` is not representable in math FastMath flags and is
/// preserved on the result as `tir-dropped-flush-to-zero` when set.
using ConvertRsqrt =
    ConvertUnaryFlushToZeroOp<cuda_tile::RsqrtOp, math::RsqrtOp>;

/// Convert cuda_tile.scan to vector.scan.
///
/// Only supports single-operand scans where the body contains exactly one
/// recognized combining op. cuda_tile.scan semantics are inclusive (result[j]
/// = f(result[j-1], X[j]) starting with the identity), so we lower with
/// `inclusive = true`. The `reverse = true` case is not representable in
/// vector.scan and is rejected.
struct ConvertScan : public OpConversionPattern<cuda_tile::ScanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // vector.scan has no reverse mode.
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

    // Build the initial_value: an (n-1)-D vector splat with the identity
    // (dim `dim` of the source removed).
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

/// Convert cuda_tile.store_view_tko to vector.transfer_write.
///
/// The indices, permutation map and in-bounds flags come from
/// buildTransferViewAccessPlan. vector.transfer_write skips out-of-bounds
/// lanes, matching the view semantics ("Out-of-bounds tile elements are masked
/// during stores").
///
/// Restrictions / guards:
///   - Only `weak` memory_ordering_semantics is supported.
///   - `memory_scope` is not supported.
///
/// `optimization_hints`, when present, is preserved on the produced
/// vector.transfer_write as the discardable attribute
/// `tir-dropped-optimization-hints`.
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

/// Convert cuda_tile.trunci to arith.trunci while preserving overflow flags.
///
/// Overflow mapping:
///   - none            -> no flags
///   - no_signed_wrap  -> nsw
///   - no_unsigned_wrap-> nuw
///   - no_wrap         -> nsw,nuw
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

/// Populate type-conversion rules for cuda_tile -> gpu/vector lowering.
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
  // cuda_tile.tile<elemTy> (scalar, rank 0):
  //   - ints        -> preserved scalar integer type
  //   - float       -> preserved scalar type
  //   - ptr<T>      -> memref<*xT> (unranked memref backing the pointer)
  converter.addConversion(
      [ctx, &converter](cuda_tile::TileType tileTy) -> Type {
        auto shape = tileTy.getShape();
        Type elemTy = tileTy.getElementType();

        if (auto ptrTy = dyn_cast<cuda_tile::PointerType>(elemTy)) {
          if (shape.empty())
            return UnrankedMemRefType::get(
                converter.convertType(ptrTy.getPointeeType()), {});
          // Ranked pointer tiles represent per-element offsets from a base
          // buffer.  Lower to vector<...xindex> so that
          // broadcast/reshape/offset become trivial vector arithmetic and
          // loads/stores lower to vector.gather/scatter.
          return VectorType::get(shape, IndexType::get(ctx));
        }
        elemTy = converter.convertType(elemTy);
        if (!elemTy || !elemTy.isIntOrFloat())
          return Type();
        if (shape.empty())
          return elemTy;
        return VectorType::get(shape, elemTy);
      });

  // tensor_view / partition_view -> ranked memref describing the same buffer.
  // (partition_view inherits its memref layout from the underlying tensor_view;
  // tile_shape / dim_map / padding_value are read off the source op's type at
  // each use site.)
  converter.addConversion([&converter](cuda_tile::TensorViewType tvTy) -> Type {
    return tensorViewToMemRefType(tvTy, converter);
  });
  converter.addConversion(
      [&converter](cuda_tile::PartitionViewType pvTy) -> Type {
        return tensorViewToMemRefType(pvTy.getTensorView(), converter);
      });
  // strided_view / gather_scatter_view likewise alias the underlying
  // tensor_view buffer; tile_shape / traversal_strides / dim_map / sparse_dim /
  // padding_value are read off the view type at each consumer use site.
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

/// Register all cuda_tile -> gpu/vector conversion patterns.
static void populateTileIRToMLIRConversionPatterns(
    TypeConverter &converter, RewritePatternSet &patterns,
    const ConvertTileIRToMLIRPassOptions &options) {
  MLIRContext *ctx = patterns.getContext();
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

/// cuda_tile.for is an automatic allocation scope but scf.for is not: wrap the
/// body of every lowered loop that allocates in a memref.alloca_scope so that
/// its stack allocations are released at the end of each iteration.
static void scopeLoopAllocations(ModuleOp module) {
  llvm::SetVector<scf::ForOp> loops;
  module.walk([&](memref::AllocaOp alloca) {
    Operation *scope = alloca->getParentOp();
    while (scope && !isa<scf::ForOp>(scope) &&
           !scope->hasTrait<OpTrait::AutomaticAllocationScope>())
      scope = scope->getParentOp();
    if (auto loop = dyn_cast_or_null<scf::ForOp>(scope))
      loops.insert(loop);
  });
  for (scf::ForOp loop : loops) {
    Block *body = loop.getBody();
    auto yield = cast<scf::YieldOp>(body->getTerminator());
    OpBuilder builder(yield);
    auto scope = memref::AllocaScopeOp::create(builder, loop.getLoc(),
                                               yield.getOperandTypes());
    Block *scopeBody = builder.createBlock(&scope.getBodyRegion());
    scopeBody->getOperations().splice(scopeBody->end(), body->getOperations(),
                                      body->begin(), scope->getIterator());
    builder.setInsertionPointToEnd(scopeBody);
    memref::AllocaScopeReturnOp::create(builder, loop.getLoc(),
                                        yield.getOperands());
    yield->setOperands(scope.getResults());
  }
}

/// Pass driver for lowering Tile IR to GPU/vector/scf/arith/memref.
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
    scopeLoopAllocations(module);

    RewritePatternSet postConversionPatterns(ctx);
    tileir::populatePostConversionPatterns(postConversionPatterns);
    walkAndApplyPatterns(module, std::move(postConversionPatterns));

    // Mark the module as a GPU container module when targeting the GPU. For the
    // CPU target the GPU container-module marker is intentionally omitted.
    if (target == TileIRTarget::GPU)
      module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                      UnitAttr::get(ctx));
  }
};

} // namespace
