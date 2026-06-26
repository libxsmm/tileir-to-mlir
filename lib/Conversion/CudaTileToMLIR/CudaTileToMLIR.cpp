//===- TileIRToMLIR.cpp - CudaTile IR to MLIR conversion --------*- C++ -*-===//
//
// Conversion pass from CudaTile IR to GPU/vector/scf/arith/memref ops.
//
//===----------------------------------------------------------------------===//

// cuda_tile ops with registered conversion patterns in this pass:
// cuda_tile::AbsFOp
// cuda_tile::AbsIOp
// cuda_tile::AddFOp
// cuda_tile::AddIOp
// cuda_tile::AllocOp
// cuda_tile::AndIOp
// cuda_tile::AssumeOp
// cuda_tile::Atan2Op
// cuda_tile::AtomicRMWTkoOp (scalar/rank-0 only; higher-rank requires
// --tileir-ptr-to-view)
// cuda_tile::BitcastOp
// cuda_tile::BroadcastOp
// cuda_tile::CatOp
// cuda_tile::CeilOp
// cuda_tile::CmpFOp
// cuda_tile::CmpIOp
// cuda_tile::ConstantOp
// cuda_tile::ContinueOp
// cuda_tile::CosOp
// cuda_tile::CosHOp
// cuda_tile::DivFOp
// cuda_tile::DivIOp
// cuda_tile::EntryOp
// cuda_tile::ExpOp
// cuda_tile::Exp2Op
// cuda_tile::ExtIOp
// cuda_tile::ExtractOp
// cuda_tile::FloorOp
// cuda_tile::FmaOp
// cuda_tile::ForOp
// cuda_tile::FToFOp
// cuda_tile::FToIOp
// cuda_tile::GetGlobalOp
// cuda_tile::GetTensorShapeOp
// cuda_tile::GetIndexSpaceShapeOp
// cuda_tile::GetNumTileBlocksOp
// cuda_tile::GetTileBlockIdOp
// cuda_tile::GlobalOp
// cuda_tile::IToFOp
// cuda_tile::IfOp
// cuda_tile::IotaOp
// cuda_tile::JoinTokensOp
// cuda_tile::LoadPtrTkoOp (scalar/rank-0 only; higher-rank requires
// --tileir-ptr-to-view) cuda_tile::LoadViewTkoOp cuda_tile::LogOp
// cuda_tile::Log2Op
// cuda_tile::MakePartitionViewOp
// cuda_tile::MakeStridedViewOp
// cuda_tile::MakeTensorViewOp
// cuda_tile::MakeTokenOp
// cuda_tile::MaxFOp
// cuda_tile::MaxIOp
// cuda_tile::MinFOp
// cuda_tile::MinIOp
// cuda_tile::MmaFOp
// cuda_tile::MmaIOp
// cuda_tile::ModuleOp
// cuda_tile::MulFOp
// cuda_tile::MulhiIOp
// cuda_tile::MulIOp
// cuda_tile::NegFOp
// cuda_tile::NegIOp
// cuda_tile::OrIOp
// cuda_tile::PackOp
// cuda_tile::PermuteOp
// cuda_tile::PowOp
// cuda_tile::PtrToPtrOp
// cuda_tile::ReduceOp
// cuda_tile::RemFOp
// cuda_tile::RemIOp
// cuda_tile::ReshapeOp
// cuda_tile::ReturnOp
// cuda_tile::RsqrtOp
// cuda_tile::ScanOp
// cuda_tile::SelectOp
// cuda_tile::ShLIOp
// cuda_tile::ShRIOp
// cuda_tile::SinOp
// cuda_tile::SinHOp
// cuda_tile::SqrtOp
// cuda_tile::StorePtrTkoOp (scalar/rank-0 only; higher-rank requires
// --tileir-ptr-to-view) cuda_tile::StoreViewTkoOp cuda_tile::SubFOp
// cuda_tile::SubIOp
// cuda_tile::TanOp
// cuda_tile::TanHOp
// cuda_tile::TruncIOp
// cuda_tile::UnpackOp
// cuda_tile::XOrIOp
// cuda_tile::YieldOp
//
// cuda_tile ops without a registered conversion pattern in this pass:
// cuda_tile::AssertOp
// cuda_tile::AtomicCASTkoOp
// cuda_tile::AtomicRedViewTkoOp
// cuda_tile::BreakOp
// cuda_tile::IntToPtrOp
// cuda_tile::LoopOp
// cuda_tile::MakeGatherScatterViewOp
// cuda_tile::MmafScaledOp
// cuda_tile::OffsetOp
// cuda_tile::PrintTkoOp
// cuda_tile::PtrToIntOp

// Notes:
// MmafScaledOp: ff no direct lowering (like to Xe) exists, a fallback would be
//   1. vector.broadcast + vector.shape_cast the scale tile to the operand's K
//   extent (infer V = K / scaleK, bail if not exact),
//   2. arith.scaling_extf %operand, %broadcastScale : <…lowp…>, <…f8E8M0FNU…>
//   to <…f32…>, then feed the two f32 results plus acc into the existing
//   buildMmaContractionSpec + vector.contract path used by ConvertMmaF.
//
// make_gather_scatter_view
// This cannot be expressed as a single vector.transfer_read/write. Along
// sparse_dim, each of the N rows of the result tile comes from an independent
// base index supplied by the 1-D index tile — i.e. a gather (load) / scatter
// (store), not a contiguous slice. Type converter + make-pattern: same as
// strided — map GatherScatterViewType → memref of tensor_view and forward the
// memref. (Cheap.) A dedicated lowering for load_view_tko / store_view_tko when
// the view is gather/scatter. The transfer-plan abstraction does not fit.
// Realistic options: Loop over the sparse dimension: for each row r, compute
// base = sparseIndex[r] * tensorStride[sparse_dim] plus the scalar offsets of
// the other dims, do a vector.transfer_read/write of the remaining (dense)
// sub-tile, and vector.insert/extract it into/out of the result tile. Masking
// for OOB rows uses the existing padding-value logic.
// vector.gather/vector.scatter with a computed index vector: build a
// per-element index vector from the 1-D sparse index broadcast across the dense
// dims plus an iota for the dense dims; this mirrors how the pointer-tile load
// (ConvertLoadPtrTko...) is handled. More compact but the index-vector
// construction is fiddly. get_index_space_shape: the sparse dim's extent is
// driven by the gather-index count / tile shape rather than the tensor dim; the
// dense dims use the partition formula. Needs its own branch. verifyIndices
// shape: the converter must accept a 1-D index tile at sparse_dim (the current
// code assumes scalar indices everywhere) and thread it through to the
// gather/scatter. This is substantially more work than strided and warrants its
// own pattern (e.g. ConvertGatherScatterLoad/Store) rather than being folded
// into the shared transfer plan.

#include "mlir/Conversion/CudaTileToMLIR/CudaTileToMLIR.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/TypeSwitch.h"

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTILEIRTOMLIRPASS
#include "mlir/Conversion/CudaTileToMLIR/Passes.h.inc"
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
static MemRefType tensorViewToMemRefType(cuda_tile::TensorViewType tvTy) {
  auto shape = tvTy.getShape();
  auto strides = tvTy.getStrides();
  Type elemTy = tvTy.getElementType();

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
  unsigned tensorViewRank;     // Rank of the underlying tensor_view
  // Optional padding value attribute from the view type; null if the view does
  // not specify one (i.e. OOB loads yield unspecified values).
  cuda_tile::PaddingValueAttr paddingValue;
};

/// Extract tile-view layout info from `view`'s type and pair it with the
/// already type-converted memref `convertedView`. Returns failure (with a
/// match-failure note) for view kinds that the transfer-based lowering cannot
/// model (e.g. gather_scatter_view, whose sparse dimension requires gather /
/// scatter rather than a contiguous transfer).
static FailureOr<ViewInfo> getViewInfo(Operation *op, Value view,
                                       Value convertedView,
                                       ConversionPatternRewriter &rewriter) {
  ViewInfo info;
  info.memref = convertedView;

  auto fillIdentityDimMapIfEmpty = [&]() {
    if (info.dimMap.empty())
      for (unsigned d = 0, e = info.tileShape.size(); d < e; ++d)
        info.dimMap.push_back(static_cast<int32_t>(d));
  };

  if (auto pvType = dyn_cast<cuda_tile::PartitionViewType>(view.getType())) {
    for (auto v : pvType.getTileShape().asArrayRef())
      info.tileShape.push_back(v);
    // partition_view tiles tile the tensor exactly: advance == tile extent.
    info.viewStrides.assign(info.tileShape.begin(), info.tileShape.end());
    info.dimMap.assign(pvType.getDimMap().begin(), pvType.getDimMap().end());
    fillIdentityDimMapIfEmpty();
    info.tensorViewRank = pvType.getTensorView().getShape().size();
    info.paddingValue = pvType.getPaddingValue();
    return info;
  }

  if (auto svType = dyn_cast<cuda_tile::StridedViewType>(view.getType())) {
    for (auto v : svType.getTileShape().asArrayRef())
      info.tileShape.push_back(v);
    // strided_view advances the tile base by the traversal stride.
    for (auto v : svType.getTraversalStrides().asArrayRef())
      info.viewStrides.push_back(v);
    info.dimMap.assign(svType.getDimMap().begin(), svType.getDimMap().end());
    fillIdentityDimMapIfEmpty();
    info.tensorViewRank = svType.getTensorView().getShape().size();
    info.paddingValue = svType.getPaddingValue();
    return info;
  }

  return rewriter.notifyMatchFailure(
      op, "view kind is not supported by the transfer-based lowering");
}

/// Validate the semantic invariants that transfer and index-space queries rely
/// on for a tile view.
static LogicalResult validateViewInfo(Operation *op, const ViewInfo &info,
                                      ConversionPatternRewriter &rewriter) {
  auto memrefTy = dyn_cast<MemRefType>(info.memref.getType());
  if (!memrefTy)
    return rewriter.notifyMatchFailure(
        op, "view source did not convert to a ranked memref");

  if (memrefTy.getRank() != static_cast<int64_t>(info.tensorViewRank))
    return rewriter.notifyMatchFailure(
        op, "converted view memref rank does not match tensor_view rank");

  if (info.dimMap.size() != info.tileShape.size())
    return rewriter.notifyMatchFailure(
        op, "view dim_map rank does not match tile_shape rank");

  if (info.viewStrides.size() != info.tileShape.size())
    return rewriter.notifyMatchFailure(
        op, "view stride rank does not match tile_shape rank");

  llvm::SmallBitVector seenDims(info.tensorViewRank);
  for (auto [tileDim, tensorDim] : llvm::enumerate(info.dimMap)) {
    if (tensorDim < 0 ||
        static_cast<unsigned>(tensorDim) >= info.tensorViewRank)
      return rewriter.notifyMatchFailure(
          op, "view dim_map references an out-of-range tensor dimension");
    if (seenDims.test(tensorDim))
      return rewriter.notifyMatchFailure(
          op, "view dim_map must be a permutation without duplicates");
    seenDims.set(tensorDim);
    if (info.tileShape[tileDim] <= 0)
      return rewriter.notifyMatchFailure(
          op, "view tile dimensions must be strictly positive");
    if (info.viewStrides[tileDim] <= 0)
      return rewriter.notifyMatchFailure(
          op, "view traversal strides must be strictly positive");
  }

  return success();
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

/// Attach the discardable `tir-dropped-flush-to-zero` unit attribute to `newOp`
/// when `flushToZero` is set. flush_to_zero has no arith/math equivalent, so it
/// is preserved as an annotation.
static void preserveDroppedFlushToZero(OpBuilder &builder, bool flushToZero,
                                       Operation *newOp) {
  if (flushToZero)
    newOp->setAttr("tir-dropped-flush-to-zero", builder.getUnitAttr());
}

/// Attach the discardable `tir-dropped-rounding` string attribute to `newOp`,
/// recording a source rounding mode that the target op cannot represent.
static void preserveDroppedRounding(OpBuilder &builder,
                                    cuda_tile::RoundingMode rounding,
                                    Operation *newOp) {
  newOp->setAttr(
      "tir-dropped-rounding",
      builder.getStringAttr(cuda_tile::stringifyRoundingMode(rounding)));
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
static Value castValueToType(OpBuilder &builder, Location loc, Value value,
                             Type targetType) {
  if (value.getType() == targetType)
    return value;
  if ((isa<IndexType>(value.getType()) && isa<IntegerType>(targetType)) ||
      (isa<IntegerType>(value.getType()) && isa<IndexType>(targetType)))
    return arith::IndexCastOp::create(builder, loc, targetType, value);
  return Value();
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

template <typename OpT>
static FailureOr<VectorType>
getConvertedVectorResultTypeOrFail(OpT op, Type sourceType,
                                   ConversionPatternRewriter &rewriter,
                                   StringRef reason) {
  auto vecTy = dyn_cast<VectorType>(sourceType);
  if (!vecTy) {
    (void)rewriter.notifyMatchFailure(op, reason);
    return failure();
  }
  return vecTy;
}

static SmallVector<int64_t> getReducedVectorShape(VectorType sourceType,
                                                  uint32_t dim) {
  SmallVector<int64_t> shape(sourceType.getShape().begin(),
                             sourceType.getShape().end());
  shape.erase(shape.begin() + dim);
  return shape;
}

/// Build the ranked memref type corresponding to a cuda_tile.global
/// definition.
///
/// cuda_tile.global stores a DenseElementsAttr payload and semantically
/// materializes a static allocation initialized at module load time. We lower
/// that allocation to memref.global with a ranked static memref type matching
/// the payload shape/element type.
static FailureOr<MemRefType>
getGlobalMemRefTypeOrFail(cuda_tile::GlobalOp globalOp,
                          ConversionPatternRewriter &rewriter,
                          Operation *diagnosticOp) {
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

  return MemRefType::get(initTy.getShape(), initTy.getElementType());
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

/// Convert unary source-based ops by forwarding the converted source operand.
template <typename SrcOp, typename DstOp>
struct ConvertUnarySourceOp : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.template replaceOpWithNewOp<DstOp>(op, adaptor.getSource());
    return success();
  }
};

/// Convert a rank-1 whole-tile bit reinterpretation (pack / unpack) to
/// vector.bitcast.
template <typename SrcOp>
struct ConvertVectorBitcastOp : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy =
        getConvertedResultTypeOrFail(op, this->getTypeConverter(), rewriter,
                                     "cannot convert pack/unpack result type");
    if (failed(resultTy))
      return failure();
    auto vecTy = dyn_cast<VectorType>(resultTy.value());
    if (!vecTy)
      return rewriter.notifyMatchFailure(
          op, "pack/unpack result did not convert to a vector type");
    rewriter.template replaceOpWithNewOp<vector::BitCastOp>(
        op, vecTy, adaptor.getSource());
    return success();
  }
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

/// Convert a unary source-based op to a math op, mapping `rounding<approx>` to
/// the `afn` (allow approximate functions) FastMath flag. The source rounding
/// mode is always preserved as `tir-dropped-rounding`; when `PreserveFtz` is
/// set, `flush_to_zero` is preserved as `tir-dropped-flush-to-zero`.
template <typename SrcOp, typename DstOp, bool PreserveFtz>
struct ConvertUnaryApproxMathOp : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rounding = op.getRoundingMode();
    auto fmf = (rounding == cuda_tile::RoundingMode::APPROX)
                   ? arith::FastMathFlags::afn
                   : arith::FastMathFlags::none;
    auto newOp = rewriter.template replaceOpWithNewOp<DstOp>(
        op, adaptor.getSource(),
        arith::FastMathFlagsAttr::get(rewriter.getContext(), fmf));
    preserveDroppedRounding(rewriter, rounding, newOp);
    if constexpr (PreserveFtz)
      preserveDroppedFlushToZero(rewriter, op.getFlushToZero(), newOp);
    return success();
  }
};

/// Convert float binary ops to arith float ops.
///
/// The cuda_tile rounding mode is mapped onto the arith op's rounding-mode
/// attribute when a direct equivalent exists; the pattern bails when it does
/// not. `flush_to_zero` has no arith equivalent and is preserved on the result
/// op as the discardable attribute `tir-dropped-flush-to-zero`.
template <typename SrcOp, typename DstOp>
struct ConvertBinaryFloatOp : public OpConversionPattern<SrcOp> {
  ConvertBinaryFloatOp(const TypeConverter &tc, MLIRContext *ctx,
                       CudaTileTarget target)
      : OpConversionPattern<SrcOp>(tc, ctx), target(target) {}

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    arith::RoundingModeAttr roundingAttr;
    if (target == CudaTileTarget::GPU) {
      auto arithRounding = mapRoundingModeToArith(op.getRoundingMode());
      if (!arithRounding)
        return rewriter.notifyMatchFailure(
            op, "rounding mode has no arith equivalent");
      roundingAttr =
          arith::RoundingModeAttr::get(rewriter.getContext(), *arithRounding);
    }
    bool ftz = op.getFlushToZero();
    auto fmAttr = arith::FastMathFlagsAttr::get(rewriter.getContext(),
                                                arith::FastMathFlags::none);
    // Older LLVM: arith float binary ops have no rounding-mode attribute.
    // auto newOp = rewriter.template replaceOpWithNewOp<DstOp>(
    //     op, adaptor.getLhs(), adaptor.getRhs(), fmAttr, roundingAttr);
    (void)roundingAttr;
    auto newOp = rewriter.template replaceOpWithNewOp<DstOp>(
        op, adaptor.getLhs(), adaptor.getRhs(), fmAttr);
    if (target == CudaTileTarget::CPU)
      preserveDroppedRounding(rewriter, op.getRoundingMode(), newOp);
    preserveDroppedFlushToZero(rewriter, ftz, newOp);
    return success();
  }

  CudaTileTarget target;
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

/// Convert binary lhs/rhs source-based ops by forwarding both operands.
template <typename SrcOp, typename DstOp>
struct ConvertBinaryLhsRhsOp : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.template replaceOpWithNewOp<DstOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs());
    return success();
  }
};

/// Convert signedness-directed casts that also require an exact rounding mode.
///
/// Used by:
///   - cuda_tile.ftoi  -> arith.fptosi / arith.fptoui
///   - cuda_tile.itof  -> arith.sitofp / arith.uitofp
///
///   1. Require the source op rounding mode to match `ExpectedRounding`.
///   2. Convert the destination tile type (`to`) via the type converter.
///   3. Dispatch by signedness to the signed/unsigned arith destination op.
template <typename SrcOp, typename SignedDstOp, typename UnsignedDstOp,
          cuda_tile::RoundingMode ExpectedRounding>
struct ConvertFromToSignednessCastWithRoundingOp
    : public OpConversionPattern<SrcOp> {
  ConvertFromToSignednessCastWithRoundingOp(const TypeConverter &tc,
                                            MLIRContext *ctx,
                                            CudaTileTarget target)
      : OpConversionPattern<SrcOp>(tc, ctx), target(target) {}

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (target == CudaTileTarget::GPU &&
        op.getRoundingMode() != ExpectedRounding)
      return rewriter.notifyMatchFailure(op,
                                         "unsupported rounding mode for cast");

    auto resultTy =
        getConvertedResultTypeOrFail(op, this->getTypeConverter(), rewriter,
                                     "cannot convert cast result type");
    if (failed(resultTy))
      return failure();

    Operation *newOp = nullptr;
    if (op.getSignedness() == cuda_tile::Signedness::Unsigned)
      newOp = rewriter
                  .template replaceOpWithNewOp<UnsignedDstOp>(
                      op, resultTy.value(), adaptor.getFrom())
                  .getOperation();
    else
      newOp = rewriter
                  .template replaceOpWithNewOp<SignedDstOp>(
                      op, resultTy.value(), adaptor.getFrom())
                  .getOperation();

    if (target == CudaTileTarget::CPU)
      preserveDroppedRounding(rewriter, op.getRoundingMode(), newOp);
    return success();
  }

  CudaTileTarget target;
};

/// Convert cuda_tile terminators (continue / yield) to scf.yield.
template <typename SrcOp>
struct ConvertToScfYield : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.template replaceOpWithNewOp<scf::YieldOp>(op,
                                                       adaptor.getOperands());
    return success();
  }
};

/// Layout of the six trailing launch-coordinate arguments that the CPU target
/// appends to every lowered entry function (see ConvertEntry). They appear in
/// the order: tile block id x/y/z, then grid dim x/y/z. The GPU target instead
/// obtains these from gpu.block_id / gpu.grid_dim, so it adds no extra
/// arguments.
struct CpuLaunchArgLayout {
  /// Total number of trailing launch-coordinate arguments.
  static constexpr unsigned kNumArgs = 6;
  /// Offset of the tile-block-id triple from the start of the launch
  /// coordinates.
  static constexpr unsigned kBlockIdBase = 0;
  /// Offset of the grid-dimension triple from the start of the launch
  /// coordinates.
  static constexpr unsigned kGridDimBase = 3;

  /// Look up the CPU function-argument index carrying the value for the query
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
///   - CPU: the matching trailing function arguments, whose indices are looked
///     up via CpuLaunchArgLayout starting at `CpuArgBase`.
///
/// Each result is cast to the converted result type as needed.
template <typename SrcOp, typename GpuDimOp, unsigned CpuArgBase>
struct ConvertDimQueryOp : public OpConversionPattern<SrcOp> {
  ConvertDimQueryOp(const TypeConverter &tc, MLIRContext *ctx,
                    CudaTileTarget target)
      : OpConversionPattern<SrcOp>(tc, ctx), target(target) {}

  LogicalResult
  matchAndRewrite(SrcOp op, typename OpConversionPattern<SrcOp>::OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultTy =
        this->getTypeConverter()->convertType(op.getResult(0).getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "cannot convert result type");

    // On CPU the launch coordinates are passed as trailing function arguments;
    // resolve the enclosing function once.
    func::FuncOp func;
    if (target == CudaTileTarget::CPU) {
      func = op->template getParentOfType<func::FuncOp>();
      if (!func || func.getNumArguments() < CpuLaunchArgLayout::kNumArgs)
        return rewriter.notifyMatchFailure(
            op,
            "expected enclosing func.func with launch-coordinate arguments");
    }

    SmallVector<Value, 3> results;
    for (gpu::Dimension dim :
         {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z}) {
      Value raw = target == CudaTileTarget::GPU
                      ? Value(GpuDimOp::create(rewriter, loc, dim))
                      : Value(func.getArgument(CpuLaunchArgLayout::argIndex(
                            func.getNumArguments(), CpuArgBase, dim)));
      Value casted = castValueToType(rewriter, loc, raw, resultTy);
      if (!casted)
        return rewriter.notifyMatchFailure(
            op, "cannot cast dim query result to target type");
      results.push_back(casted);
    }
    rewriter.replaceOp(op, results);
    return success();
  }

  CudaTileTarget target;
};

/// Convert cuda_tile.maxf/minf based on propagate_nan.
///
/// flush_to_zero has no equivalent in the arith FastMath flags; it is dropped.
/// propagate_nan dispatches to arith.maximumf/minimumf (NaN propagating) vs
/// arith.maxnumf/minnumf (NaN suppressing).
template <typename SrcOp, bool IsMax>
struct ConvertMinMaxFOp : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // flush_to_zero has no equivalent in arith max/min FastMath flags; drop it.
    bool ftz = op.getFlushToZero();
    Operation *newOp;
    if (op.getPropagateNan()) {
      if constexpr (IsMax) {
        newOp = rewriter
                    .template replaceOpWithNewOp<arith::MaximumFOp>(
                        op, adaptor.getLhs(), adaptor.getRhs())
                    .getOperation();
      } else {
        newOp = rewriter
                    .template replaceOpWithNewOp<arith::MinimumFOp>(
                        op, adaptor.getLhs(), adaptor.getRhs())
                    .getOperation();
      }
    } else {
      if constexpr (IsMax) {
        newOp = rewriter
                    .template replaceOpWithNewOp<arith::MaxNumFOp>(
                        op, adaptor.getLhs(), adaptor.getRhs())
                    .getOperation();
      } else {
        newOp = rewriter
                    .template replaceOpWithNewOp<arith::MinNumFOp>(
                        op, adaptor.getLhs(), adaptor.getRhs())
                    .getOperation();
      }
    }
    preserveDroppedFlushToZero(rewriter, ftz, newOp);
    return success();
  }
};

/// Convert cuda_tile.maxi/mini using signed or unsigned arith variants.
template <typename SrcOp, bool IsMax>
struct ConvertMinMaxIOp : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool isUnsigned = op.getSignedness() == cuda_tile::Signedness::Unsigned;
    if constexpr (IsMax) {
      if (isUnsigned)
        rewriter.template replaceOpWithNewOp<arith::MaxUIOp>(
            op, adaptor.getLhs(), adaptor.getRhs());
      else
        rewriter.template replaceOpWithNewOp<arith::MaxSIOp>(
            op, adaptor.getLhs(), adaptor.getRhs());
    } else {
      if (isUnsigned)
        rewriter.template replaceOpWithNewOp<arith::MinUIOp>(
            op, adaptor.getLhs(), adaptor.getRhs());
      else
        rewriter.template replaceOpWithNewOp<arith::MinSIOp>(
            op, adaptor.getLhs(), adaptor.getRhs());
    }
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

  Block &block = op.getBody().front();
  if (block.getNumArguments() != 2)
    return rewriter.notifyMatchFailure(
        op, "cannot determine combining kind from body");

  auto yieldOp = dyn_cast<cuda_tile::YieldOp>(block.getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return rewriter.notifyMatchFailure(
        op, "cannot determine combining kind from body");

  Operation *combiningOp = nullptr;
  for (Operation &bodyOp : block.without_terminator()) {
    if (combiningOp)
      return rewriter.notifyMatchFailure(
          op, "cannot determine combining kind from body");
    combiningOp = &bodyOp;
  }
  if (!combiningOp)
    return rewriter.notifyMatchFailure(
        op, "cannot determine combining kind from body");

  if (combiningOp->getNumOperands() != 2 || combiningOp->getNumResults() != 1 ||
      yieldOp.getOperand(0) != combiningOp->getResult(0))
    return rewriter.notifyMatchFailure(
        op, "cannot determine combining kind from body");

  auto lhsArg = dyn_cast<BlockArgument>(combiningOp->getOperand(0));
  auto rhsArg = dyn_cast<BlockArgument>(combiningOp->getOperand(1));
  if (!lhsArg || !rhsArg || lhsArg.getOwner() != &block ||
      rhsArg.getOwner() != &block ||
      lhsArg.getArgNumber() == rhsArg.getArgNumber())
    return rewriter.notifyMatchFailure(
        op, "cannot determine combining kind from body");

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
    return rewriter.notifyMatchFailure(
        op, "cannot determine combining kind from body");

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

/// Convert integer binary ops that dispatch on signedness (remi, shri).
template <typename SrcOp, typename SignedDstOp, typename UnsignedDstOp>
struct ConvertBinaryLhsRhsWithSignednessOp : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getSignedness() == cuda_tile::Signedness::Unsigned)
      rewriter.template replaceOpWithNewOp<UnsignedDstOp>(op, adaptor.getLhs(),
                                                          adaptor.getRhs());
    else
      rewriter.template replaceOpWithNewOp<SignedDstOp>(op, adaptor.getLhs(),
                                                        adaptor.getRhs());
    return success();
  }
};

/// Validate shared load/store_tko constraints before lowering.
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
  if (!op.getResultToken().use_empty())
    return rewriter.notifyMatchFailure(
        op, "result_token has live uses; this lowering drops the token");
  return success();
}

/// Preserve an `optimization_hints` attribute that this lowering otherwise
/// drops. The attribute's inner builtin `DictionaryAttr` is attached to the
/// produced target op as the discardable attribute
/// `tir-dropped-optimization-hints`.
template <typename TkoOp>
static void preserveDroppedOptHints(TkoOp op, Operation *newOp) {
  if (auto hints = op.getOptimizationHintsAttr())
    newOp->setAttr("tir-dropped-optimization-hints", hints.getValue());
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
                            Value view, Value convertedView, VectorType vecTy,
                            ValueRange convertedIndices) {
  auto viewInfoOr = getViewInfo(op, view, convertedView, rewriter);
  if (failed(viewInfoOr))
    return failure();
  ViewInfo viewInfo = std::move(*viewInfoOr);
  if (failed(validateViewInfo(op, viewInfo, rewriter)))
    return failure();

  unsigned tileRank = viewInfo.tileShape.size();
  unsigned tensorRank = viewInfo.tensorViewRank;

  // The TileView interface is the authoritative source for how many tile-space
  // indices the view expects; cross-check the extracted layout against it so a
  // mismatch is reported against the view contract rather than silently relied
  // upon downstream.
  if (cast<cuda_tile::TileView>(view.getType()).getViewIndexRank() != tileRank)
    return rewriter.notifyMatchFailure(
        op, "view index rank does not match tile_shape rank");
  if ((unsigned)vecTy.getRank() != tileRank)
    return rewriter.notifyMatchFailure(
        op, "converted tile rank does not match view tile_shape rank");
  if (convertedIndices.size() != tileRank)
    return rewriter.notifyMatchFailure(
        op, "view index rank does not match tile_shape rank");

  Location loc = op->getLoc();
  auto *ctx = rewriter.getContext();

  // Build memref indices in tensor-dimension order.
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> memrefIndices(tensorRank, zero);
  for (unsigned i = 0; i < tileRank; ++i) {
    unsigned tensorDim = viewInfo.dimMap[i];
    int64_t stride = viewInfo.viewStrides[i];
    Value tileIndex = castValueToType(rewriter, loc, convertedIndices[i],
                                      rewriter.getIndexType());
    if (!tileIndex)
      return rewriter.notifyMatchFailure(
          op, "view index could not be converted to index");
    Value strideVal = arith::ConstantIndexOp::create(rewriter, loc, stride);
    auto nswFlag = arith::IntegerOverflowFlagsAttr::get(
        rewriter.getContext(), arith::IntegerOverflowFlags::nsw);
    Value elemOffset =
        arith::MulIOp::create(rewriter, loc, tileIndex, strideVal, nswFlag);
    memrefIndices[tensorDim] = elemOffset;
  }

  SmallVector<AffineExpr> permExprs;
  permExprs.reserve(tileRank);
  for (int32_t td : viewInfo.dimMap)
    permExprs.push_back(getAffineDimExpr(td, ctx));
  auto permutationMap = AffineMap::get(tensorRank, 0, permExprs, ctx);

  auto memrefTy = cast<MemRefType>(viewInfo.memref.getType());
  auto memrefShape = memrefTy.getShape();
  SmallVector<bool> inBounds(tileRank, false);
  for (unsigned i = 0; i < tileRank; ++i) {
    int64_t ext = memrefShape[viewInfo.dimMap[i]];
    int64_t stride = viewInfo.viewStrides[i];
    int64_t tile = viewInfo.tileShape[i];
    if (ext == ShapedType::kDynamic || stride <= 0 || tile <= 0) {
      inBounds[i] = false;
      continue;
    }
    // Number of in-bounds tile bases along this dimension (partial edge tiles
    // are included), then check whether the trailing tile fits entirely.
    int64_t numTiles = (ext + stride - 1) / stride;
    int64_t lastBase = numTiles > 0 ? (numTiles - 1) * stride : 0;
    inBounds[i] = (lastBase + tile <= ext);
  }

  return TransferViewAccessPlan{std::move(viewInfo), std::move(memrefIndices),
                                permutationMap, std::move(inBounds)};
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//
using ConvertAbsF = ConvertUnarySourceOp<cuda_tile::AbsFOp, math::AbsFOp>;

using ConvertAbsI = ConvertUnarySourceOp<cuda_tile::AbsIOp, math::AbsIOp>;

using ConvertAddF = ConvertBinaryFloatOp<cuda_tile::AddFOp, arith::AddFOp>;

using ConvertAddI =
    ConvertBinaryLhsRhsWithOverflowOp<cuda_tile::AddIOp, arith::AddIOp>;

using ConvertAndI = ConvertBinaryLhsRhsOp<cuda_tile::AndIOp, arith::AndIOp>;

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

/// Convert cuda_tile.assume to a pass-through (just forward the source value).
struct ConvertAssume : public OpConversionPattern<cuda_tile::AssumeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::AssumeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getValue());
    return success();
  }
};

/// Convert cuda_tile.atan2 to math.atan2.
struct ConvertAtan2 : public OpConversionPattern<cuda_tile::Atan2Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::Atan2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<math::Atan2Op>(op, adaptor.getX(),
                                               adaptor.getY());
    return success();
  }
};

/// Convert cuda_tile.bitcast to arith.bitcast.
struct ConvertBitcast : public OpConversionPattern<cuda_tile::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getConvertedResultTypeOrFail(
        op, getTypeConverter(), rewriter, "cannot convert bitcast result type");
    if (failed(resultTy))
      return failure();
    rewriter.replaceOpWithNewOp<arith::BitcastOp>(op, resultTy.value(),
                                                  adaptor.getSource());
    return success();
  }
};

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
      // (vector<NxMxindex>). For pointer tiles, the source may be an unranked
      // memref (scalar ptr) that needs to be turned into an index first.
      if (isa<UnrankedMemRefType>(source.getType())) {
        // Scalar pointer being broadcast to ranked pointer tile.
        // We broadcast an index of 0 (since the base pointer is extracted later
        // directly from the source by the load/store lowerings).
        Value zero = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
        rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, dstVecTy, zero);
      } else {
        rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, dstVecTy, source);
      }
    } else if (adaptor.getSource().getType() == resultTy) {
      rewriter.replaceOp(op, adaptor.getSource());
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported broadcast result type");
    }

    return success();
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

using ConvertCeil = ConvertUnarySourceOp<cuda_tile::CeilOp, math::CeilOp>;

/// Convert cuda_tile.cmpf to arith.cmpf.
struct ConvertCmpF : public OpConversionPattern<cuda_tile::CmpFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<arith::CmpFPredicate> pred = failure();
    using CP = cuda_tile::ComparisonPredicate;
    using CO = cuda_tile::ComparisonOrdering;
    if (op.getComparisonOrdering() == CO::ORDERED) {
      switch (op.getComparisonPredicate()) {
      case CP::EQUAL:
        pred = arith::CmpFPredicate::OEQ;
        break;
      case CP::NOT_EQUAL:
        pred = arith::CmpFPredicate::ONE;
        break;
      case CP::LESS_THAN:
        pred = arith::CmpFPredicate::OLT;
        break;
      case CP::LESS_THAN_OR_EQUAL:
        pred = arith::CmpFPredicate::OLE;
        break;
      case CP::GREATER_THAN:
        pred = arith::CmpFPredicate::OGT;
        break;
      case CP::GREATER_THAN_OR_EQUAL:
        pred = arith::CmpFPredicate::OGE;
        break;
      }
    } else if (op.getComparisonOrdering() == CO::UNORDERED) {
      switch (op.getComparisonPredicate()) {
      case CP::EQUAL:
        pred = arith::CmpFPredicate::UEQ;
        break;
      case CP::NOT_EQUAL:
        pred = arith::CmpFPredicate::UNE;
        break;
      case CP::LESS_THAN:
        pred = arith::CmpFPredicate::ULT;
        break;
      case CP::LESS_THAN_OR_EQUAL:
        pred = arith::CmpFPredicate::ULE;
        break;
      case CP::GREATER_THAN:
        pred = arith::CmpFPredicate::UGT;
        break;
      case CP::GREATER_THAN_OR_EQUAL:
        pred = arith::CmpFPredicate::UGE;
        break;
      }
    }
    if (failed(pred))
      return rewriter.notifyMatchFailure(op,
                                         "unsupported cmpf predicate/ordering");
    rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, *pred, adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};

/// Convert cuda_tile.cmpi to arith.cmpi.
struct ConvertCmpI : public OpConversionPattern<cuda_tile::CmpIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<arith::CmpIPredicate> pred = failure();
    using CP = cuda_tile::ComparisonPredicate;
    bool isUnsigned = op.getSignedness() == cuda_tile::Signedness::Unsigned;
    switch (op.getComparisonPredicate()) {
    case CP::EQUAL:
      pred = arith::CmpIPredicate::eq;
      break;
    case CP::NOT_EQUAL:
      pred = arith::CmpIPredicate::ne;
      break;
    case CP::LESS_THAN:
      pred = isUnsigned ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt;
      break;
    case CP::LESS_THAN_OR_EQUAL:
      pred = isUnsigned ? arith::CmpIPredicate::ule : arith::CmpIPredicate::sle;
      break;
    case CP::GREATER_THAN:
      pred = isUnsigned ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::sgt;
      break;
    case CP::GREATER_THAN_OR_EQUAL:
      pred = isUnsigned ? arith::CmpIPredicate::uge : arith::CmpIPredicate::sge;
      break;
    }
    if (failed(pred))
      return rewriter.notifyMatchFailure(
          op, "unsupported cmpi predicate/signedness");
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, *pred, adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};

/// Convert cuda_tile.constant to arith/vector constants.
///
///   - Scalar tiles (rank 0): Convert to scalar arith ops
///   - Ranked tiles: Convert to an arith.constant with a DenseElementsAttr of
///     the target vector type (splat values use the splat form, e.g.
///     `arith.constant dense<7> : vector<1x1xi32>`).
///   - Scalar integer conversion preserves the integer type; casts to `index`
///     are inserted at the ops that require them.
///   - Scalar pointer tiles are forwarded unchanged; the type converter maps
///     them to `memref<*xT>`.
struct ConvertConstant : public OpConversionPattern<cuda_tile::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileType = op.getType();
    Type resultType = getTypeConverter()->convertType(tileType);
    if (!resultType)
      return failure();

    auto denseVal = op.getValue();
    Location loc = op.getLoc();

    if (tileType.getShape().empty()) {
      // Scalar tile -> scalar constant
      auto elemTy = tileType.getElementType();
      if (isa<IntegerType>(elemTy)) {
        auto splat = denseVal.getSplatValue<APInt>();
        rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, resultType,
                                                          splat.getSExtValue());
      } else if (isa<FloatType>(elemTy)) {
        auto splat = denseVal.getSplatValue<APFloat>();
        rewriter.replaceOpWithNewOp<arith::ConstantFloatOp>(
            op, cast<FloatType>(resultType), splat);
      } else {
        return failure();
      }
    } else {
      // Tile -> vector constant (splat or dense), emitted directly as
      // arith.constant with a DenseElementsAttr of the target vector type.
      auto vecTy = cast<VectorType>(resultType);
      DenseElementsAttr vecAttr;
      if (denseVal.isSplat()) {
        vecAttr =
            DenseElementsAttr::get(vecTy, denseVal.getSplatValue<Attribute>());
      } else {
        SmallVector<Attribute> attrs(denseVal.getValues<Attribute>().begin(),
                                     denseVal.getValues<Attribute>().end());
        vecAttr = DenseElementsAttr::get(vecTy, attrs);
      }
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, vecTy, vecAttr);
    }
    return success();
  }
};

using ConvertContinue = ConvertToScfYield<cuda_tile::ContinueOp>;

using ConvertCos = ConvertUnarySourceOp<cuda_tile::CosOp, math::CosOp>;

using ConvertCosH = ConvertUnarySourceOp<cuda_tile::CosHOp, math::CoshOp>;

/// Convert cuda_tile.divf to arith.divf.
///
/// The cuda_tile rounding mode is mapped onto arith.divf's rounding-mode
/// attribute when a direct equivalent exists. `rounding<approx>` has no
/// rounding-mode equivalent but maps to the `arcp` (allow reciprocal) FastMath
/// flag. Any other unmapped rounding mode causes the pattern to bail.
/// `flush_to_zero` has no arith equivalent and is preserved on the result as
/// `tir-dropped-flush-to-zero`.
struct ConvertDivF : public OpConversionPattern<cuda_tile::DivFOp> {
  ConvertDivF(const TypeConverter &tc, MLIRContext *ctx, CudaTileTarget target)
      : OpConversionPattern(tc, ctx), target(target) {}

  LogicalResult
  matchAndRewrite(cuda_tile::DivFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rounding = op.getRoundingMode();
    bool ftz = op.getFlushToZero();

    arith::RoundingModeAttr roundingAttr;
    arith::FastMathFlags fmf = arith::FastMathFlags::none;
    if (rounding == cuda_tile::RoundingMode::APPROX) {
      // approx has no rounding-mode equivalent; map to the arcp FastMath flag.
      fmf = arith::FastMathFlags::arcp;
    } else if (auto arithRounding = mapRoundingModeToArith(rounding)) {
      if (target == CudaTileTarget::GPU)
        roundingAttr =
            arith::RoundingModeAttr::get(rewriter.getContext(), *arithRounding);
    } else if (target == CudaTileTarget::GPU) {
      return rewriter.notifyMatchFailure(
          op, "rounding mode has no arith equivalent");
    }

    // Older LLVM: arith.divf has no rounding-mode attribute.
    // auto newOp = rewriter.replaceOpWithNewOp<arith::DivFOp>(
    //     op, adaptor.getLhs(), adaptor.getRhs(),
    //     arith::FastMathFlagsAttr::get(rewriter.getContext(), fmf),
    //     roundingAttr);
    (void)roundingAttr;
    auto newOp = rewriter.replaceOpWithNewOp<arith::DivFOp>(
        op, adaptor.getLhs(), adaptor.getRhs(),
        arith::FastMathFlagsAttr::get(rewriter.getContext(), fmf));
    if (target == CudaTileTarget::CPU)
      preserveDroppedRounding(rewriter, rounding, newOp);
    preserveDroppedFlushToZero(rewriter, ftz, newOp);
    return success();
  }

  CudaTileTarget target;
};

/// Convert cuda_tile.divi to arith.divsi/divui; rejects non-ZERO rounding.
struct ConvertDivI : public OpConversionPattern<cuda_tile::DivIOp> {
  ConvertDivI(const TypeConverter &tc, MLIRContext *ctx, CudaTileTarget target)
      : OpConversionPattern(tc, ctx), target(target) {}

  LogicalResult
  matchAndRewrite(cuda_tile::DivIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (target == CudaTileTarget::GPU &&
        op.getRounding() != cuda_tile::RoundingMode::ZERO)
      return rewriter.notifyMatchFailure(
          op, "only rounding<zero> (truncating) division is supported");
    Operation *newOp = nullptr;
    if (op.getSignedness() == cuda_tile::Signedness::Unsigned)
      newOp = rewriter
                  .replaceOpWithNewOp<arith::DivUIOp>(op, adaptor.getLhs(),
                                                      adaptor.getRhs())
                  .getOperation();
    else
      newOp = rewriter
                  .replaceOpWithNewOp<arith::DivSIOp>(op, adaptor.getLhs(),
                                                      adaptor.getRhs())
                  .getOperation();
    if (target == CudaTileTarget::CPU)
      preserveDroppedRounding(rewriter, op.getRounding(), newOp);
    return success();
  }

  CudaTileTarget target;
};

/// Convert cuda_tile.entry to gpu.func (gpu target) or func.func (cpu target).
///
/// The function signature is derived by applying the type converter to each
/// entry argument type (e.g. `tile<ptr<T>>` -> `memref<*xT>`, `tile<i32>` ->
/// `i32`). The entry body is signature-converted in place and then moved into
/// the new function body.
///
/// For the CPU target the function additionally receives six trailing `i32`
/// arguments carrying the launch coordinates that the GPU target obtains from
/// dimension-query ops: the three tile block ids (x, y, z) followed by the
/// three grid dimensions (x, y, z). These follow the converted entry
/// arguments.
///
/// `optimization_hints`, when present, is preserved on the produced function as
/// the discardable attribute `tir-dropped-optimization-hints`.
struct ConvertEntry : public OpConversionPattern<cuda_tile::EntryOp> {
  ConvertEntry(const TypeConverter &tc, MLIRContext *ctx, CudaTileTarget target)
      : OpConversionPattern(tc, ctx), target(target) {}

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

    funcArgTypes.reserve(funcArgTypes.size() + numArgs);
    for (unsigned i = 0; i < numArgs; ++i) {
      Type origTy = entryBlock->getArgument(i).getType();
      Type converted = tc->convertType(origTy);
      if (!converted)
        return rewriter.notifyMatchFailure(entryOp,
                                           "cannot convert entry arg type");
      funcArgTypes.push_back(converted);
      sigConv.addInputs(i, converted);
    }

    // CPU launch coordinates are passed in as six trailing i32 arguments since
    // gpu dimension-query ops are unavailable on that target.
    if (target == CudaTileTarget::CPU) {
      SmallVector<Type> launchArgTypes(CpuLaunchArgLayout::kNumArgs,
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

    if (target == CudaTileTarget::GPU) {
      // GPU: lower to a gpu.func kernel and merge the converted body into its
      // (auto-created) entry block.
      auto gpuFunc =
          gpu::GPUFuncOp::create(rewriter, loc, entryOp.getSymName(), funcType);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       rewriter.getUnitAttr());
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

  CudaTileTarget target;
};

using ConvertExp = ConvertUnarySourceOp<cuda_tile::ExpOp, math::ExpOp>;

/// Convert cuda_tile.exp2 to math.exp2.
///
/// `flush_to_zero` is not representable in math FastMath flags and is preserved
/// on the result as `tir-dropped-flush-to-zero` when set.
using ConvertExp2 = ConvertUnaryFlushToZeroOp<cuda_tile::Exp2Op, math::Exp2Op>;

/// Convert cuda_tile.exti to arith.extsi / arith.extui.
///
///   1. Convert the destination tile type (`to`) via the type converter.
///   2. Dispatch to arith.extui for `signedness = unsigned`, otherwise to
///      arith.extsi.
struct ConvertExtI : public OpConversionPattern<cuda_tile::ExtIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ExtIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy =
        getConvertedResultTypeOrFail(op, this->getTypeConverter(), rewriter,
                                     "cannot convert cast result type");
    if (failed(resultTy))
      return failure();

    if (op.getSignedness() == cuda_tile::Signedness::Unsigned)
      rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, resultTy.value(),
                                                  adaptor.getFrom());
    else
      rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, resultTy.value(),
                                                  adaptor.getFrom());
    return success();
  }
};

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

using ConvertFloor = ConvertUnarySourceOp<cuda_tile::FloorOp, math::FloorOp>;

/// Convert cuda_tile.fma to math.fma.
///
/// `rounding_mode` and `flush_to_zero` are not representable in math FastMath
/// flags and are preserved on the result as `tir-dropped-rounding` and
/// `tir-dropped-flush-to-zero`.
struct ConvertFma : public OpConversionPattern<cuda_tile::FmaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::FmaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rounding = op.getRoundingMode();
    bool ftz = op.getFlushToZero();
    auto newOp = rewriter.replaceOpWithNewOp<math::FmaOp>(
        op, adaptor.getLhs(), adaptor.getRhs(), adaptor.getAcc());
    preserveDroppedRounding(rewriter, rounding, newOp);
    preserveDroppedFlushToZero(rewriter, ftz, newOp);
    return success();
  }
};

/// Convert cuda_tile.for to scf.for.
///
///   - Create scf.ForOp with bounds cast to `index` and initial values
///     forwarded.
///   - If the induction variable indexes a `partition_view` along a statically
///     known tile-shape axis (directly, or as the value-preserving
///     `divi(muli(iv, N), N)`), rescale `lb`/`ub`/`step` from tile space to
///     element space by that tile size and replace body uses of the IV with
///     `divui(new_iv, tile_size)` so the tile-space index is recovered.
///   - Convert the region types and merge the original body into the new one,
///     replacing the induction-variable and iter-arg block arguments.
///
/// `unsignedCmp` (use unsigned comparison for loop termination) has no
/// equivalent on scf.for and is otherwise dropped; it is preserved on the
/// produced scf.for as the discardable attribute `tir-dropped-unsigned-cmp`.
struct ConvertFor : public OpConversionPattern<cuda_tile::ForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lb = castValueToType(rewriter, loc, adaptor.getLowerBound(),
                               rewriter.getIndexType());
    Value ub = castValueToType(rewriter, loc, adaptor.getUpperBound(),
                               rewriter.getIndexType());
    Value step = castValueToType(rewriter, loc, adaptor.getStep(),
                                 rewriter.getIndexType());
    if (!lb || !ub || !step)
      return rewriter.notifyMatchFailure(
          op, "for bounds could not be converted to index");

    // Infer the loop-axis tile size from the partition_view tile shapes at the
    // load_view_tko/store_view_tko indices that are semantically the induction
    // variable. The same tile shape is what buildTransferViewAccessPlan
    // multiplies by when forming element-space memref indices, so deriving the
    // loop step from it keeps step and per-tile index scaling consistent.
    //
    // Frontends sometimes materialize that tile-space index as
    // `divi(muli(iv, N), N)` rather than a direct `%iv`. For round-to-zero
    // division `(iv * N) / N == iv` exactly, so such indices are normalized to
    // a direct IV use; that lets the merged body see the recovered tile index
    // `divui(new_iv, N)` directly and keeps the transfer lowering on the same
    // path as a plain `%iv` index. Detection is read-only; the normalization is
    // applied through the rewriter only once we commit to rescaling.
    Value origIV = op.getInductionVar();
    auto stripAssume = [](Value value) {
      while (auto assume = value.getDefiningOp<cuda_tile::AssumeOp>())
        value = assume.getValue();
      return value;
    };
    auto matchScalarI32Constant = [&](Value value) -> std::optional<int64_t> {
      value = stripAssume(value);
      DenseIntElementsAttr ints;
      if (!matchPattern(value, m_Constant(&ints)) || !ints.isSplat())
        return std::nullopt;
      return ints.getSplatValue<APInt>().getSExtValue();
    };
    // Returns true when `idx` is semantically `origIV` as a tile index of the
    // given size: either a direct use, or `divi(muli(iv, N), N)`
    // (round-to-zero).
    auto matchesLoopTileIndex = [&](Value idx, int64_t expectedTileSize) {
      idx = stripAssume(idx);
      if (idx == origIV)
        return true;
      auto div = idx.getDefiningOp<cuda_tile::DivIOp>();
      if (!div || div.getRounding() != cuda_tile::RoundingMode::ZERO)
        return false;
      if (matchScalarI32Constant(div.getRhs()) != expectedTileSize)
        return false;
      auto mul = stripAssume(div.getLhs()).getDefiningOp<cuda_tile::MulIOp>();
      if (!mul)
        return false;
      for (auto [maybeIV, maybeCst] :
           {std::pair<Value, Value>(mul.getLhs(), mul.getRhs()),
            std::pair<Value, Value>(mul.getRhs(), mul.getLhs())})
        if (stripAssume(maybeIV) == origIV &&
            matchScalarI32Constant(maybeCst) == expectedTileSize)
          return true;
      return false;
    };

    // Read-only scan: derive the tile size and remember which (op, operand)
    // index positions need normalizing to a direct IV use.
    int64_t tileSize = 0;
    SmallVector<std::pair<Operation *, unsigned>> normalizeTargets;
    auto inspectViewUse = [&](Operation *owner, Value view,
                              OperandRange indices) -> bool {
      auto pvTy = dyn_cast<cuda_tile::PartitionViewType>(view.getType());
      if (!pvTy)
        return true;
      auto shape = pvTy.getTileShape().asArrayRef();
      for (auto [pos, idx] : llvm::enumerate(indices)) {
        if (pos >= shape.size())
          return false;
        int64_t sz = shape[pos];
        if (sz <= 0)
          return false;
        if (!matchesLoopTileIndex(idx, sz))
          continue;
        if (tileSize != 0 && tileSize != sz)
          return false;
        tileSize = sz;
        if (idx != origIV)
          normalizeTargets.emplace_back(owner, pos);
      }
      return true;
    };
    WalkResult walkResult = op.getBody()->walk([&](Operation *bodyOp) {
      bool ok = true;
      if (auto ld = dyn_cast<cuda_tile::LoadViewTkoOp>(bodyOp))
        ok = inspectViewUse(ld, ld.getView(), ld.getIndex());
      else if (auto st = dyn_cast<cuda_tile::StoreViewTkoOp>(bodyOp))
        ok = inspectViewUse(st, st.getView(), st.getIndex());
      return ok ? WalkResult::advance() : WalkResult::interrupt();
    });
    if (walkResult.wasInterrupted())
      tileSize = 0;

    if (tileSize > 0) {
      // Normalize the matched `divi(muli(iv, N), N)` indices to a direct IV use
      // (value-preserving) so the merged body indexes with `divui(new_iv, N)`.
      for (auto [owner, pos] : normalizeTargets) {
        unsigned idxPos = pos;
        MutableOperandRange indices =
            isa<cuda_tile::LoadViewTkoOp>(owner)
                ? cast<cuda_tile::LoadViewTkoOp>(owner).getIndexMutable()
                : cast<cuda_tile::StoreViewTkoOp>(owner).getIndexMutable();
        rewriter.modifyOpInPlace(
            owner, [&] { indices.slice(idxPos, 1).assign(origIV); });
      }
      Value tileSizeVal =
          arith::ConstantIndexOp::create(rewriter, loc, tileSize);
      lb = arith::MulIOp::create(rewriter, loc, lb, tileSizeVal);
      ub = arith::MulIOp::create(rewriter, loc, ub, tileSizeVal);
      step = arith::MulIOp::create(rewriter, loc, step, tileSizeVal);
    }

    auto newForOp = scf::ForOp::create(rewriter, loc, lb, ub, step,
                                       adaptor.getInitValues());
    if (op.getUnsignedCmp())
      newForOp->setAttr("tir-dropped-unsigned-cmp", rewriter.getUnitAttr());

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

    Value iv;
    if (tileSize > 0) {
      // The loop iterates in element space; recover the tile-space index
      // that body ops expect with divui(iv, tileSize).
      Value tileSizeVal =
          arith::ConstantIndexOp::create(rewriter, loc, tileSize);
      Value tileIdx = arith::DivUIOp::create(
          rewriter, loc, newForOp.getInductionVar(), tileSizeVal);
      iv = castValueToType(rewriter, loc, tileIdx,
                           oldBody->getArgument(0).getType());
    } else {
      iv = castValueToType(rewriter, loc, newForOp.getInductionVar(),
                           oldBody->getArgument(0).getType());
    }
    if (!iv)
      return rewriter.notifyMatchFailure(
          op, "for induction variable could not be converted to body type");
    replacingValues.push_back(iv);
    for (auto arg : newForOp.getRegionIterArgs())
      replacingValues.push_back(arg);

    rewriter.mergeBlocks(oldBody, newBody, replacingValues);
    rewriter.replaceOp(op, newForOp.getResults());
    return success();
  }
};

/// Convert cuda_tile.ftof to arith.extf / arith.truncf.
///
///   - Source and destination element widths must differ.
///   - Widening uses arith.extf, which has no rounding-mode attribute (float
///     widening is exact); the source rounding mode is preserved on the result
///     as the discardable attribute `tir-dropped-rounding`.
///   - Narrowing uses arith.truncf, which carries a rounding-mode attribute, so
///     the source rounding mode is mapped onto it; the pattern bails when the
///     cuda_tile rounding mode has no direct arith equivalent.
///
/// Works for both scalar float and vector<float> types.
struct ConvertFToF : public OpConversionPattern<cuda_tile::FToFOp> {
  ConvertFToF(const TypeConverter &tc, MLIRContext *ctx, CudaTileTarget target)
      : OpConversionPattern(tc, ctx), target(target) {}

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
      // arith.extf has no rounding-mode attribute; preserve the source rounding
      // mode as a discardable annotation.
      auto extOp = rewriter.replaceOpWithNewOp<arith::ExtFOp>(
          op, resultTy.value(), adaptor.getFrom());
      preserveDroppedRounding(rewriter, op.getRoundingMode(), extOp);
      return success();
    }
    if (srcWidth > dstWidth) {
      // arith.truncf carries a rounding-mode attribute; map it directly and
      // bail when there is no equivalent.
      auto arithRounding = mapRoundingModeToArith(op.getRoundingMode());
      if (target == CudaTileTarget::GPU && !arithRounding)
        return rewriter.notifyMatchFailure(
            op, "ftof rounding mode has no arith.truncf equivalent");
      arith::RoundingModeAttr roundingAttr;
      if (target == CudaTileTarget::GPU && arithRounding)
        roundingAttr =
            arith::RoundingModeAttr::get(rewriter.getContext(), *arithRounding);
      auto truncOp = rewriter.replaceOpWithNewOp<arith::TruncFOp>(
          op, resultTy.value(), adaptor.getFrom(), roundingAttr,
          /*fastmath=*/arith::FastMathFlagsAttr{});
      if (target == CudaTileTarget::CPU)
        preserveDroppedRounding(rewriter, op.getRoundingMode(), truncOp);
      return success();
    }

    return rewriter.notifyMatchFailure(op,
                                       "ftof source/result widths must differ");
  }

  CudaTileTarget target;
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
    if (auto cudaGlobal = dyn_cast<cuda_tile::GlobalOp>(symbolOp)) {
      rankedMemRefTy = getGlobalMemRefTypeOrFail(cudaGlobal, rewriter, op);
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
    if (failed(validateViewInfo(op, viewInfo, rewriter)))
      return failure();

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
                      CpuLaunchArgLayout::kGridDimBase>;

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
                      CpuLaunchArgLayout::kBlockIdBase>;

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
    auto memrefTy = getGlobalMemRefTypeOrFail(op, rewriter, op);
    if (failed(memrefTy))
      return failure();

    auto initAttr = dyn_cast<ElementsAttr>(op.getValue());
    if (!initAttr)
      return rewriter.notifyMatchFailure(
          op, "global initializer must be an elements attribute");

    auto initTy = dyn_cast<ShapedType>(initAttr.getType());
    if (!initTy || !initTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "global initializer must be a statically shaped elements "
              "attribute");

    auto tensorTy =
        RankedTensorType::get(initTy.getShape(), initTy.getElementType());
    ElementsAttr normalizedInitAttr = initAttr;
    if (initAttr.getType() != tensorTy) {
      auto denseAttr = dyn_cast<DenseElementsAttr>(initAttr);
      if (!denseAttr)
        return rewriter.notifyMatchFailure(
            op, "global initializer must be a dense elements attribute when "
                "retyping is required");

      if (denseAttr.isSplat()) {
        normalizedInitAttr = ElementsAttr(DenseElementsAttr::get(
            tensorTy, denseAttr.getSplatValue<Attribute>()));
      } else {
        SmallVector<Attribute> values(denseAttr.getValues<Attribute>().begin(),
                                      denseAttr.getValues<Attribute>().end());
        normalizedInitAttr =
            ElementsAttr(DenseElementsAttr::get(tensorTy, values));
      }
    }

    IntegerAttr alignmentAttr;
    if (op.getAlignment() != 0)
      alignmentAttr = rewriter.getI64IntegerAttr(op.getAlignment());

    rewriter.replaceOpWithNewOp<memref::GlobalOp>(
        op, op.getSymName(), /*sym_visibility=*/StringAttr(), *memrefTy,
        normalizedInitAttr, /*constant=*/false, alignmentAttr);
    return success();
  }
};

/// Convert cuda_tile.if to scf.if.
struct ConvertIf : public OpConversionPattern<cuda_tile::IfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "cannot convert if result types");

    bool hasElse = !op.getElseRegion().empty();
    auto newIfOp = scf::IfOp::create(rewriter, op.getLoc(), resultTypes,
                                     adaptor.getCondition(), hasElse);

    // Move then region.
    {
      Block *oldBlock = op.getThenBlock();
      Block *newBlock = newIfOp.thenBlock();
      if (newBlock->mightHaveTerminator())
        rewriter.eraseOp(newBlock->getTerminator());
      rewriter.mergeBlocks(oldBlock, newBlock, {});
    }

    // Move else region (if present).
    if (hasElse) {
      Block *oldBlock = op.getElseBlock();
      Block *newBlock = newIfOp.elseBlock();
      if (newBlock->mightHaveTerminator())
        rewriter.eraseOp(newBlock->getTerminator());
      rewriter.mergeBlocks(oldBlock, newBlock, {});
    }

    rewriter.replaceOp(op, newIfOp.getResults());
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
/// Mapping summary:
///   - View indices (tile-level) are multiplied by the tile shape and stored
///     into the memref index slot dim_map[i]. Non-covered memref dims are 0.
///   - The result `vector` keeps the tile's logical (tile-dim) shape. A
///     `permutation_map` on the transfer_read maps memref dim `dim_map[i]`
///     to vector dim `i`, so no post-load transpose is needed.
///   - `inBounds[i]` is set true only when the tensor extent along the
///     corresponding memref dim is static and exactly divisible by the tile
///     extent. Otherwise the transfer_read masks and substitutes the
///     `padding` value, which is taken from `partition_view.padding_value`.
///
/// Restrictions (return notifyMatchFailure on violation):
///   - Only `weak` memory_ordering_semantics is supported.
///   - `memory_scope` is not supported.
///   - `result_token` must have no live uses; this lowering drops the token.
///
/// `optimization_hints`, when present, is preserved on the produced
/// vector.transfer_read as the discardable attribute
/// `tir-dropped-optimization-hints`.
struct ConvertLoadViewTko
    : public OpConversionPattern<cuda_tile::LoadViewTkoOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::LoadViewTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (failed(checkCommonTkoGuards(op, rewriter)))
      return failure();

    auto vecTy = getConvertedVectorResultTypeOrFail(
        op, getTypeConverter()->convertType(op.getTile().getType()), rewriter,
        "load_view_tko tile must convert to a vector type");
    if (failed(vecTy))
      return failure();

    auto plan = buildTransferViewAccessPlan(rewriter, op, op.getView(),
                                            adaptor.getView(), *vecTy,
                                            adaptor.getIndex());
    if (failed(plan))
      return failure();

    Value padding;
    if (!plan->viewInfo.paddingValue) {
      padding = ub::PoisonOp::create(rewriter, loc, vecTy->getElementType());
    } else if (auto fty = dyn_cast<FloatType>(vecTy->getElementType())) {
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
                                             vecTy->getElementType(), 0);
    }
    auto readOp = vector::TransferReadOp::create(
        rewriter, loc, *vecTy, plan->viewInfo.memref, plan->memrefIndices,
        AffineMapAttr::get(plan->permutationMap), padding,
        /*mask=*/Value(), rewriter.getBoolArrayAttr(plan->inBounds));
    preserveDroppedOptHints(op, readOp);

    rewriter.replaceOp(op, {readOp.getResult(), Value()});
    return success();
  }
};

using ConvertLog = ConvertUnarySourceOp<cuda_tile::LogOp, math::LogOp>;

using ConvertLog2 = ConvertUnarySourceOp<cuda_tile::Log2Op, math::Log2Op>;

/// Convert cuda_tile.make_partition_view
///
/// The partition_view type maps to the same ranked memref as its underlying
/// tensor_view, so this pattern just forwards the already-converted memref.
struct ConvertMakePartitionView
    : public OpConversionPattern<cuda_tile::MakePartitionViewOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::MakePartitionViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getTensorView());
    return success();
  }
};

/// Convert cuda_tile.make_strided_view
///
/// Like partition_view, a strided_view is backed by the same ranked memref as
/// its underlying tensor_view; tile_shape / traversal_strides / dim_map /
/// padding_value are read off the result type at each consumer use site. So
/// this pattern just forwards the already-converted tensor_view memref.
struct ConvertMakeStridedView
    : public OpConversionPattern<cuda_tile::MakeStridedViewOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::MakeStridedViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getTensorView());
    return success();
  }
};

/// Convert cuda_tile.make_token by erasing it.
///
/// TKO tokens are currently ignored during lowering; this pattern erases the
/// generator op. It assumes other patterns (load_tko etc.) have already
/// dropped their dependency on this token.
struct ConvertMakeToken : public OpConversionPattern<cuda_tile::MakeTokenOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::MakeTokenOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert cuda_tile.join_tokens by erasing it.
struct ConvertJoinTokens : public OpConversionPattern<cuda_tile::JoinTokensOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::JoinTokensOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert cuda_tile.make_gather_scatter_view
///
/// A gather_scatter_view is backed by the same ranked memref as its underlying
/// tensor_view, so the view value itself forwards the converted memref. Note
/// that consuming a gather_scatter_view through load_view_tko / store_view_tko
/// requires gather/scatter semantics along the sparse dimension, which the
/// transfer-based consumer lowering does not yet implement; those consumers
/// will report a match failure for this view kind.
struct ConvertMakeGatherScatterView
    : public OpConversionPattern<cuda_tile::MakeGatherScatterViewOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::MakeGatherScatterViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getTensorView());
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
/// Kernel-pointer function arguments are a special case: they enter as raw
/// pointers with a statically-zero descriptor offset, so there is nothing to
/// recover. We short-circuit them to a static `0` and emit *no* IR. This is not
/// just an optimization: ConvertMemrefArgsToPtrArgs only promotes an unranked
/// argument to `!llvm.ptr` when every use of it is one and the same cast.
/// Emitting an extra `memref.cast` of the argument here would create a second,
/// divergent use and defeat that promotion, leaving the argument unranked and
/// the enclosing `func.func` unconvertible under the bare-pointer calling
/// convention.
static OpFoldResult
recoverUnrankedPtrOffset(ConversionPatternRewriter &rewriter, Location loc,
                         Value unrankedBase) {
  if (isa<BlockArgument>(unrankedBase))
    return rewriter.getIndexAttr(0);
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
/// Dynamic shape/stride operands of the source op are scalar `tile<i32>`, so
/// this lowering inserts explicit casts to `index` before building the
/// memref.reinterpret_cast operands; static dims come from the tensor_view
/// type.
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

    auto tvType = cast<cuda_tile::TensorViewType>(op.getType());
    auto tvShape = tvType.getShape();
    auto tvStrides = tvType.getStrides();
    unsigned rank = tvShape.size();

    // Variadic operands hold only the values for the dynamic dims, in
    // dimension order.
    auto dynShape = adaptor.getDynamicShape();
    auto dynStrides = adaptor.getDynamicStrides();
    unsigned dynShapeIdx = 0, dynStrideIdx = 0;

    SmallVector<OpFoldResult> sizes;
    sizes.reserve(rank);
    for (unsigned d = 0; d < rank; ++d) {
      if (tvShape[d] == ShapedType::kDynamic) {
        Value size =
            castValueToType(rewriter, op.getLoc(), dynShape[dynShapeIdx++],
                            rewriter.getIndexType());
        if (!size)
          return rewriter.notifyMatchFailure(
              op, "dynamic tensor_view shape could not be converted to index");
        sizes.push_back(size);
      } else
        sizes.push_back(rewriter.getIndexAttr(tvShape[d]));
    }

    SmallVector<OpFoldResult> strides;
    strides.reserve(rank);
    for (unsigned d = 0; d < rank; ++d) {
      if (tvStrides[d] == ShapedType::kDynamic) {
        Value stride =
            castValueToType(rewriter, op.getLoc(), dynStrides[dynStrideIdx++],
                            rewriter.getIndexType());
        if (!stride)
          return rewriter.notifyMatchFailure(
              op, "dynamic tensor_view stride could not be converted to index");
        strides.push_back(stride);
      } else
        strides.push_back(rewriter.getIndexAttr(tvStrides[d]));
    }

    // make_tensor_view reshapes the buffer at the base pointer's current
    // location, so it must preserve whatever absolute offset the base memref
    // descriptor carries. Recover it unconditionally rather than matching only
    // a specific producer shape (e.g. direct scalar `offset`).
    OpFoldResult offset =
        recoverUnrankedPtrOffset(rewriter, op.getLoc(), adaptor.getBase());

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, resultTy, adaptor.getBase(), offset, sizes, strides);
    return success();
  }
};

using ConvertMaxF = ConvertMinMaxFOp<cuda_tile::MaxFOp, /*IsMax=*/true>;

using ConvertMaxI = ConvertMinMaxIOp<cuda_tile::MaxIOp, /*IsMax=*/true>;

using ConvertMinF = ConvertMinMaxFOp<cuda_tile::MinFOp, /*IsMax=*/false>;

using ConvertMinI = ConvertMinMaxIOp<cuda_tile::MinIOp, /*IsMax=*/false>;

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
/// The only semantic difference here is element type domain (integer).
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

    auto signLhs = op.getSignednessLhs();
    auto signRhs = op.getSignednessRhs();
    // Explicit combining kind = add (mmai is integer multiply-accumulate).
    auto newOp = rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        op, adaptor.getLhs(), adaptor.getRhs(), adaptor.getAcc(),
        rewriter.getAffineMapArrayAttr({spec->mapA, spec->mapB, spec->mapC}),
        rewriter.getArrayAttr(spec->iterTypes), vector::CombiningKind::ADD);
    newOp->setAttr(
        "tir-dropped-signedness-lhs",
        rewriter.getStringAttr(cuda_tile::stringifySignedness(signLhs)));
    newOp->setAttr(
        "tir-dropped-signedness-rhs",
        rewriter.getStringAttr(cuda_tile::stringifySignedness(signRhs)));
    return success();
  }
};

/// Convert cuda_tile.module by moving its body contents.
///
/// For the GPU target the body is moved into a new gpu.module of the same name.
/// For the CPU target the cuda_tile.module is dissolved: its contents are
/// inlined into the enclosing module (the builtin.module the pass runs on) and
/// the cuda_tile.module wrapper is erased.
struct ConvertModule : public OpConversionPattern<cuda_tile::ModuleOp> {
  ConvertModule(const TypeConverter &tc, MLIRContext *ctx,
                CudaTileTarget target)
      : OpConversionPattern(tc, ctx), target(target) {}

  LogicalResult
  matchAndRewrite(cuda_tile::ModuleOp cudaMod, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (target == CudaTileTarget::GPU) {
      auto gpuMod = gpu::GPUModuleOp::create(rewriter, cudaMod.getLoc(),
                                             cudaMod.getSymName());
      Block *oldBody = &cudaMod.getBody().front();
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
      rewriter.inlineBlockBefore(&cudaMod.getBody().front(), cudaMod);
    }

    rewriter.eraseOp(cudaMod);
    return success();
  }

  CudaTileTarget target;
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
/// unranked cast carries the buffer pointer at the offset position, so a
/// downstream make_tensor_view's reinterpret_cast (offset 0) correctly starts
/// at the shifted location.
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
    OpFoldResult totalOff;
    if (isa<Attribute>(srcOff)) {
      totalOff = OpFoldResult(offIdx);
    } else {
      totalOff = OpFoldResult(arith::AddIOp::create(rewriter, op.getLoc(),
                                                    cast<Value>(srcOff), offIdx)
                                  .getResult());
    }

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

using ConvertNegF = ConvertUnarySourceOp<cuda_tile::NegFOp, arith::NegFOp>;

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
    newOp->setAttr(
        "tir-dropped-overflow",
        rewriter.getStringAttr(cuda_tile::stringifyIntegerOverflow(overflow)));
    return success();
  }
};

/// Convert scalar (rank-0) cuda_tile.load_ptr_tko on a `tile<ptr<T>>` to a
/// `memref.reinterpret_cast` + `memref.load`. The source `tile<ptr<T>>` is
/// converted to `memref<*xT>`; a rank-0 reinterpret_cast (offset 0) recovers
/// the scalar memref the load reads from.
///
/// Higher-rank pointer loads must first be lifted by `--tileir-ptr-to-view`.
///
/// `optimization_hints`, when present, is preserved on the produced memref.load
/// as the discardable attribute `tir-dropped-optimization-hints`.
struct ConvertLoadPtrTkoScalar
    : public OpConversionPattern<cuda_tile::LoadPtrTkoOp> {
  using OpConversionPattern::OpConversionPattern;

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

    auto srcUnranked =
        dyn_cast<UnrankedMemRefType>(adaptor.getSource().getType());
    if (!srcUnranked)
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
    : public OpConversionPattern<cuda_tile::StorePtrTkoOp> {
  using OpConversionPattern::OpConversionPattern;

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

    auto dstUnranked =
        dyn_cast<UnrankedMemRefType>(adaptor.getDestination().getType());
    if (!dstUnranked)
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

/// Helper for vector.gather/scatter lowering. Derives the common 1D base
/// memref and the flattened (rank-1) mask and index vectors.
///
/// vector.gather/scatter only lower to LLVM for rank-1 vectors, so the index
/// and mask are flattened here with vector.shape_cast (a pure row-major
/// reshape). The base is already a flat 1-D memref and the whole address lives
/// in the per-element index vector, so flattening preserves the
/// lane<->index<->value correspondence exactly. Callers flatten the
/// value/result and shape_cast back to the tile shape.
static LogicalResult deriveGatherScatterMemRefAndMask(
    Operation *op, Value origPtr, Value cvtPtr, Value origMask, Value cvtMask,
    Type elemTy, ArrayRef<int64_t> shape, ConversionPatternRewriter &rewriter,
    Value &baseMemref, Value &mask, Value &indexVec) {
  Location loc = op->getLoc();

  indexVec = cvtPtr;
  auto ptrVecTy = dyn_cast<VectorType>(indexVec.getType());
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
  Value scalarBase;
  if (auto blockArg = dyn_cast<BlockArgument>(origBase)) {
    scalarBase = rewriter.getRemappedValue(blockArg);
  } else {
    scalarBase = rewriter.getRemappedValue(origBase);
  }
  if (!scalarBase)
    return rewriter.notifyMatchFailure(op, "cannot find converted base memref");

  // Cast to memref<?xelemTy> for gather/scatter. The flat type must keep a
  // dynamic offset: the scalar base may be a `memref.reinterpret_cast` that
  // carries a non-zero offset (e.g. the per-row `h_in*W` of a pooling window).
  // A plain `memref<?xelemTy>` has a *static* offset of 0, which would make the
  // gather/scatter address computation (getStridedElementPtr) ignore the
  // descriptor's offset field and drop the row stride entirely.
  auto flatMemTy =
      get1DDynamicOffsetMemRefType(elemTy, /*size=*/ShapedType::kDynamic);
  if (!isa<MemRefType>(scalarBase.getType()) &&
      !isa<UnrankedMemRefType>(scalarBase.getType()))
    return rewriter.notifyMatchFailure(op, "base is not a memref");
  baseMemref = memref::CastOp::create(rewriter, loc, flatMemTy, scalarBase);

  // Mask.
  auto maskTy = VectorType::get(shape, rewriter.getI1Type());
  if (origMask) {
    mask = cvtMask;
  } else {
    Value trueVal = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
    mask = vector::BroadcastOp::create(rewriter, loc, maskTy, trueVal);
  }

  // Flatten the index and mask to rank-1 so the gather/scatter is legal for
  // the Vector->LLVM lowering (which only supports rank-1).
  if (ptrVecTy.getRank() != 1) {
    int64_t numElts = ptrVecTy.getNumElements();
    auto flatIdxTy = VectorType::get({numElts}, rewriter.getIndexType());
    indexVec = vector::ShapeCastOp::create(rewriter, loc, flatIdxTy, indexVec);
    auto flatMaskTy = VectorType::get({numElts}, rewriter.getI1Type());
    mask = vector::ShapeCastOp::create(rewriter, loc, flatMaskTy, mask);
  }

  return success();
}

/// Convert ranked cuda_tile.load_ptr_tko to vector.gather.
///
/// The pointer tile (vector<...xindex>) holds per-element offsets from the
/// buffer base. We trace the original IR to find the scalar base pointer
/// (converted to memref<*xT>), cast it to memref<?xT>, and emit a 1-D
/// vector.gather, reshaping back to the tile shape.
///
/// `optimization_hints`, when present, is preserved on the produced
/// vector.gather as the discardable attribute `tir-dropped-optimization-hints`.
struct ConvertLoadPtrTkoRanked
    : public OpConversionPattern<cuda_tile::LoadPtrTkoOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::LoadPtrTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileTy = cast<cuda_tile::TileType>(op.getResult().getType());
    if (tileTy.getShape().empty())
      return rewriter.notifyMatchFailure(op, "scalar load handled elsewhere");

    if (failed(checkCommonTkoGuards(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    auto shape = tileTy.getShape();
    Type elemTy = tileTy.getElementType();

    Value baseMemref, mask, indexVec;
    if (failed(deriveGatherScatterMemRefAndMask(
            op, op.getSource(), adaptor.getSource(), op.getMask(),
            adaptor.getMask(), elemTy, shape, rewriter, baseMemref, mask,
            indexVec)))
      return failure();

    // Passthrough.
    auto resultVecTy = VectorType::get(shape, elemTy);
    Value passThru;
    if (op.getPaddingValue()) {
      passThru = adaptor.getPaddingValue();
    } else {
      auto zeroAttr = rewriter.getZeroAttr(resultVecTy);
      passThru =
          arith::ConstantOp::create(rewriter, loc, resultVecTy, zeroAttr);
    }

    // vector.gather lowers to LLVM only for rank-1 vectors. The index/mask were
    // flattened to rank-1 in the helper; flatten the passthrough to match,
    // emit a 1-D gather, then shape_cast the result back to the tile shape.
    auto flatResTy = VectorType::get({resultVecTy.getNumElements()}, elemTy);
    if (resultVecTy.getRank() != 1)
      passThru =
          vector::ShapeCastOp::create(rewriter, loc, flatResTy, passThru);

    // Emit vector.gather.
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto gatherOp =
        vector::GatherOp::create(rewriter, loc, flatResTy, baseMemref,
                                 ValueRange{c0}, indexVec, mask, passThru);
    preserveDroppedOptHints(op, gatherOp);

    Value result = gatherOp.getResult();
    if (resultVecTy.getRank() != 1)
      result = vector::ShapeCastOp::create(rewriter, loc, resultVecTy, result);

    rewriter.replaceOp(op, {result, Value()});
    return success();
  }
};

/// Convert ranked cuda_tile.store_ptr_tko to vector.scatter.
///
/// `optimization_hints`, when present, is preserved on the produced
/// vector.scatter as the discardable attribute
/// `tir-dropped-optimization-hints`.
struct ConvertStorePtrTkoRanked
    : public OpConversionPattern<cuda_tile::StorePtrTkoOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::StorePtrTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto valTileTy = cast<cuda_tile::TileType>(op.getValue().getType());
    if (valTileTy.getShape().empty())
      return rewriter.notifyMatchFailure(op, "scalar store handled elsewhere");

    if (failed(checkCommonTkoGuards(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    auto shape = valTileTy.getShape();
    Type elemTy = valTileTy.getElementType();

    Value baseMemref, mask, indexVec;
    if (failed(deriveGatherScatterMemRefAndMask(
            op, op.getDestination(), adaptor.getDestination(), op.getMask(),
            adaptor.getMask(), elemTy, shape, rewriter, baseMemref, mask,
            indexVec)))
      return failure();

    Value valVec = adaptor.getValue();

    // vector.scatter lowers to LLVM only for rank-1 vectors. The index/mask
    // were flattened to rank-1 in the helper; flatten the value to match.
    auto valVecTy = cast<VectorType>(valVec.getType());
    if (valVecTy.getRank() != 1) {
      auto flatValTy = VectorType::get({valVecTy.getNumElements()}, elemTy);
      valVec = vector::ShapeCastOp::create(rewriter, loc, flatValTy, valVec);
    }

    // Emit vector.scatter.
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto scatterOp = vector::ScatterOp::create(
        rewriter, loc, /*resultType=*/Type(), baseMemref, ValueRange{c0},
        indexVec, mask, valVec);
    preserveDroppedOptHints(op, scatterOp);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Map a cuda_tile.atomic_rmw mode to the equivalent arith atomic_rmw kind.
///
/// MAX/MIN are the signed integer variants; UMAX/UMIN are the unsigned ones.
/// XCHG (unconditional swap) maps to `assign`. Returns nullopt for modes with
/// no clean arith equivalent.
static std::optional<arith::AtomicRMWKind>
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
  return std::nullopt;
}

/// Convert scalar (rank-0) cuda_tile.atomic_rmw_tko on a `tile<ptr<T>>` to a
/// `memref.reinterpret_cast` + `memref.atomic_rmw`. Mirrors
/// ConvertLoadPtrTkoScalar: the source `tile<ptr<T>>` is converted to
/// `memref<*xT>`; a rank-0 reinterpret_cast (offset 0) recovers the scalar
/// memref the atomic operates on.
///
/// Both ops return the value read at the location before the update, so the
/// result maps directly. The `memory_ordering_semantics` and `memory_scope`
/// attributes have no representation on memref.atomic_rmw (which lowers to an
/// acq_rel LLVM atomicrmw with no scope); they are preserved on the result as
/// the discardable attributes `tir-dropped-memory-ordering` and
/// `tir-dropped-memory-scope`.
///
/// Higher-rank atomics are not lowered in this pass.
struct ConvertAtomicRMWTko
    : public OpConversionPattern<cuda_tile::AtomicRMWTkoOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::AtomicRMWTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileTy = cast<cuda_tile::TileType>(op.getResult().getType());
    if (!tileTy.getShape().empty())
      return rewriter.notifyMatchFailure(
          op, "only scalar (rank-0) atomic_rmw_tko is supported here");
    if (auto mask = op.getMask())
      if (!mask.getType().getShape().empty() ||
          !mask.getDefiningOp<cuda_tile::ConstantOp>() ||
          !mask.getDefiningOp<cuda_tile::ConstantOp>()
               .getValue()
               .getValues<llvm::APInt>()[0]
               .getBoolValue())
        return rewriter.notifyMatchFailure(
            op, "masked scalar atomic_rmw_tko is not supported");
    if (!op.getResultToken().use_empty())
      return rewriter.notifyMatchFailure(
          op, "result_token has live uses; this lowering drops the token");

    std::optional<arith::AtomicRMWKind> kind = mapAtomicRMWMode(op.getMode());
    if (!kind)
      return rewriter.notifyMatchFailure(op, "unsupported atomic_rmw mode");

    auto srcUnranked =
        dyn_cast<UnrankedMemRefType>(adaptor.getPointers().getType());
    if (!srcUnranked)
      return rewriter.notifyMatchFailure(
          op, "expected unranked memref pointer source");

    Location loc = op.getLoc();
    Value rc = reinterpretScalarPtrPreservingOffset(rewriter, loc,
                                                    adaptor.getPointers());
    auto rmw =
        memref::AtomicRMWOp::create(rewriter, loc, *kind, adaptor.getArg(), rc,
                                    /*indices=*/ValueRange{});
    rmw->setAttr(
        "tir-dropped-memory-ordering",
        rewriter.getStringAttr(cuda_tile::stringifyMemoryOrderingSemantics(
            op.getMemoryOrderingSemantics())));
    rmw->setAttr("tir-dropped-memory-scope",
                 rewriter.getStringAttr(
                     cuda_tile::stringifyMemoryScope(op.getMemoryScope())));
    rewriter.replaceOp(op, {rmw.getResult(), Value()});
    return success();
  }
};

using ConvertOrI = ConvertBinaryLhsRhsOp<cuda_tile::OrIOp, arith::OrIOp>;

/// Convert cuda_tile.pack to vector.bitcast (rank-1 whole-tile bit
/// reinterpret).
using ConvertPack = ConvertVectorBitcastOp<cuda_tile::PackOp>;

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

/// Convert cuda_tile.pow to math.powf.
struct ConvertPow : public OpConversionPattern<cuda_tile::PowOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::PowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<math::PowFOp>(op, adaptor.getSource(),
                                              adaptor.getExponent());
    return success();
  }
};

/// Convert cuda_tile.ptr_to_ptr using memref.cast when representable.
///
/// Pointer model in this pass: tile<ptr<T>> -> memref<*xT>
///
///   1. Convert the result pointer tile type to a memref type.
///   2. Recover the converted memref source when the adaptor provides a
///      temporary unrealized_conversion_cast wrapper.
///   3. Require memref.cast compatibility and emit memref.cast.
///   4. Fail the pattern if ptr_to_ptr cannot be represented as memref.cast.
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
    // During conversion, ptr_to_ptr often receives a target-materialized
    // operand (unrealized_conversion_cast) to the still-illegal source type.
    // Peel that wrapper to recover the already-converted memref input.
    if (auto materialize = source.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (materialize.getInputs().size() == 1 &&
          isa<BaseMemRefType>(materialize.getInputs()[0].getType()))
        source = materialize.getInputs()[0];
    }

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

using ConvertRemF = ConvertBinaryLhsRhsOp<cuda_tile::RemFOp, arith::RemFOp>;

using ConvertRemI =
    ConvertBinaryLhsRhsWithSignednessOp<cuda_tile::RemIOp, arith::RemSIOp,
                                        arith::RemUIOp>;

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
      if (isa<UnrankedMemRefType>(source.getType())) {
        // Scalar pointer being reshaped to ranked pointer tile
        // (e.g. tile<ptr<T>> → tile<1x1xptr<T>>). The ranked ptr type is now
        // vector<...xindex>. We broadcast an index of 0 (since the base pointer
        // is extracted later directly from the source by the load/store
        // lowerings).
        Value zero = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
        rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, dstVecTy, zero);
      } else {
        rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, dstVecTy, source);
      }
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
struct ConvertReturn : public OpConversionPattern<cuda_tile::ReturnOp> {
  ConvertReturn(const TypeConverter &tc, MLIRContext *ctx,
                CudaTileTarget target)
      : OpConversionPattern(tc, ctx), target(target) {}

  LogicalResult
  matchAndRewrite(cuda_tile::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (target == CudaTileTarget::GPU)
      rewriter.replaceOpWithNewOp<gpu::ReturnOp>(op);
    else
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }

  CudaTileTarget target;
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

/// Convert cuda_tile.select to arith.select.
///
/// cuda_tile.select is element-wise: result[i] = cond[i] ? val_if_true[i]
/// : val_if_false[i]. All three operands have the same shape and the
/// condition is i1 (scalar or vector of i1). arith.select natively supports
/// both scalar i1 and vector<...xi1> conditions with matching shapes, so the
/// lowering is a direct one-to-one mapping.
struct ConvertSelect : public OpConversionPattern<cuda_tile::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::SelectOp>(
        op, adaptor.getCond(), adaptor.getValIfTrue(), adaptor.getValIfFalse());
    return success();
  }
};

using ConvertShLI =
    ConvertBinaryLhsRhsWithOverflowOp<cuda_tile::ShLIOp, arith::ShLIOp>;

using ConvertShRI =
    ConvertBinaryLhsRhsWithSignednessOp<cuda_tile::ShRIOp, arith::ShRSIOp,
                                        arith::ShRUIOp>;

using ConvertSin = ConvertUnarySourceOp<cuda_tile::SinOp, math::SinOp>;

using ConvertSinH = ConvertUnarySourceOp<cuda_tile::SinHOp, math::SinhOp>;

/// Convert cuda_tile.sqrt to math.sqrt.
///
/// `rounding<approx>` maps to the `afn` (allow approximate functions) FastMath
/// flag. All other rounding modes and `flush_to_zero` are not representable in
/// math FastMath flags and are preserved on the result as
/// `tir-dropped-rounding` and `tir-dropped-flush-to-zero`.
using ConvertSqrt = ConvertUnaryApproxMathOp<cuda_tile::SqrtOp, math::SqrtOp,
                                             /*PreserveFtz=*/true>;

/// Convert cuda_tile.store_view_tko to vector.transfer_write.
///
/// Mapping summary (mirror of ConvertLoadViewTko):
///   - Memref indices: memrefIndices[dim_map[i]] = tileIndex[i] * tileShape[i].
///   - The stored vector keeps tile-dim order; a `permutation_map` makes
///     vector dim i write into memref dim dim_map[i] (no pre-store transpose).
///   - `inBounds[i]` is true only when the tensor extent along memref dim
///     dim_map[i] is static and divisible by tile dim i. Otherwise
///     vector.transfer_write masks the OOB lanes, matching the
///     partition_view spec ("Out-of-bounds tile elements are masked during
///     stores").
///
/// Restrictions / guards:
///   - Only `weak` memory_ordering_semantics is supported.
///   - `memory_scope` is not supported.
///   - `result_token` must be unused (we drop the token).
///
/// `optimization_hints`, when present, is preserved on the produced
/// vector.transfer_write as the discardable attribute
/// `tir-dropped-optimization-hints`.
struct ConvertStoreViewTko
    : public OpConversionPattern<cuda_tile::StoreViewTkoOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::StoreViewTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (failed(checkCommonTkoGuards(op, rewriter)))
      return failure();

    auto vecTy = getConvertedVectorResultTypeOrFail(
        op, getTypeConverter()->convertType(op.getTile().getType()), rewriter,
        "store_view_tko tile must convert to a vector type");
    if (failed(vecTy))
      return failure();

    auto plan = buildTransferViewAccessPlan(rewriter, op, op.getView(),
                                            adaptor.getView(), *vecTy,
                                            adaptor.getIndex());
    if (failed(plan))
      return failure();

    auto writeOp = vector::TransferWriteOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{}, adaptor.getTile(),
        plan->viewInfo.memref, plan->memrefIndices,
        AffineMapAttr::get(plan->permutationMap),
        /*mask=*/Value(), rewriter.getBoolArrayAttr(plan->inBounds));
    preserveDroppedOptHints(op, writeOp);

    rewriter.eraseOp(op);
    return success();
  }
};

using ConvertSubF = ConvertBinaryFloatOp<cuda_tile::SubFOp, arith::SubFOp>;

using ConvertSubI =
    ConvertBinaryLhsRhsWithOverflowOp<cuda_tile::SubIOp, arith::SubIOp>;

using ConvertTan = ConvertUnarySourceOp<cuda_tile::TanOp, math::TanOp>;

/// Convert cuda_tile.tanh to math.tanh.
///
/// rounding<approx> maps to the `afn` (allow approximate functions) FastMath
/// flag on math.tanh. The `full` rounding mode has no equivalent in FastMath
/// flags; both modes lower to math.tanh.
using ConvertTanH = ConvertUnaryApproxMathOp<cuda_tile::TanHOp, math::TanhOp,
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

/// Convert cuda_tile.unpack to vector.bitcast (rank-1 whole-tile bit
/// reinterpret).
using ConvertUnpack = ConvertVectorBitcastOp<cuda_tile::UnpackOp>;

using ConvertXOrI = ConvertBinaryLhsRhsOp<cuda_tile::XOrIOp, arith::XOrIOp>;

using ConvertYield = ConvertToScfYield<cuda_tile::YieldOp>;

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Type converter and conversion pattern population
//===----------------------------------------------------------------------===//

/// Populate type-conversion rules for cuda_tile -> gpu/vector lowering.
static void populateTileIRToMLIRTypeConverter(TypeConverter &converter,
                                              MLIRContext *ctx,
                                              CudaTileTarget target) {
  // Fallback: keep types unchanged.
  converter.addConversion([](Type type) { return type; });

  // cuda_tile.tile<MxNxelemTy> -> vector<MxNxelemTy> (ranked tiles)
  // cuda_tile.tile<elemTy> (scalar, rank 0):
  //   - ints        -> preserved scalar integer type
  //   - float       -> preserved scalar type
  //   - ptr<T>      -> memref<*xT> (unranked memref backing the pointer)
  converter.addConversion([ctx, target](cuda_tile::TileType tileTy) -> Type {
    auto shape = tileTy.getShape();
    auto elemTy = tileTy.getElementType();

    // CPU has no tf32 representation; lower tf32 tiles to f32 so the resulting
    // vector/arith ops are valid on the host target.
    if (target == CudaTileTarget::CPU && isa<FloatTF32Type>(elemTy))
      elemTy = Float32Type::get(ctx);

    if (shape.empty()) {
      if (isa<IntegerType>(elemTy)) {
        return elemTy;
      }

      if (isa<FloatType>(elemTy))
        return elemTy;
      if (auto ptrTy = dyn_cast<cuda_tile::PointerType>(elemTy))
        return UnrankedMemRefType::get(ptrTy.getPointeeType(), {});
      return Type();
    }
    if (auto ptrTy = dyn_cast<cuda_tile::PointerType>(elemTy)) {
      // Ranked pointer tiles represent per-element offsets from a base
      // buffer.  Lower to vector<...xindex> so that broadcast/reshape/offset
      // become trivial vector arithmetic and loads/stores lower to
      // vector.gather/scatter.
      return VectorType::get(shape, IndexType::get(ctx));
    }
    return VectorType::get(shape, elemTy);
  });

  // tensor_view / partition_view -> ranked memref describing the same buffer.
  // (partition_view inherits its memref layout from the underlying tensor_view;
  // tile_shape / dim_map / padding_value are read off the source op's type at
  // each use site.)
  converter.addConversion([](cuda_tile::TensorViewType tvTy) -> Type {
    return tensorViewToMemRefType(tvTy);
  });
  converter.addConversion([](cuda_tile::PartitionViewType pvTy) -> Type {
    return tensorViewToMemRefType(pvTy.getTensorView());
  });
  // strided_view / gather_scatter_view likewise alias the underlying
  // tensor_view buffer; tile_shape / traversal_strides / dim_map / sparse_dim /
  // padding_value are read off the view type at each consumer use site.
  converter.addConversion([](cuda_tile::StridedViewType svTy) -> Type {
    return tensorViewToMemRefType(svTy.getTensorView());
  });
  converter.addConversion([](cuda_tile::GatherScatterViewType gsTy) -> Type {
    return tensorViewToMemRefType(gsTy.getTensorView());
  });
  converter.addConversion(
      [](cuda_tile::TokenType tokTy) -> Type { return tokTy; });

  // Source/target materialization: both insert an unrealized_conversion_cast.
  auto materializeCast = [](OpBuilder &builder, Type resultType,
                            ValueRange inputs, Location loc) -> Value {
    return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
        .getResult(0);
  };
  converter.addSourceMaterialization(materializeCast);
  converter.addTargetMaterialization(materializeCast);
}

/// Register all cuda_tile -> gpu/vector conversion patterns.
static void populateTileIRToMLIRConversionPatterns(TypeConverter &converter,
                                                   RewritePatternSet &patterns,
                                                   CudaTileTarget target) {
  MLIRContext *ctx = patterns.getContext();
  // Target-dependent patterns select gpu vs func ops (module / entry / return)
  // and gpu dim-query ops vs leading function arguments (block id / grid dim).
  patterns.add<ConvertEntry, ConvertModule, ConvertReturn,
               ConvertGetNumTileBlocks, ConvertGetTileBlockId>(converter, ctx,
                                                               target);
  patterns.add<ConvertAddF, ConvertDivF, ConvertDivI, ConvertFToF, ConvertFToI,
               ConvertIToF, ConvertMulF, ConvertSubF>(converter, ctx, target);
  patterns.add<
      ConvertAbsF, ConvertAbsI, ConvertAddI, ConvertAlloca, ConvertAndI,
      ConvertAssume, ConvertAtan2, ConvertBitcast, ConvertBroadcast, ConvertCat,
      ConvertCeil, ConvertCmpF, ConvertCmpI, ConvertConstant, ConvertContinue,
      ConvertCos, ConvertCosH, ConvertAtomicRMWTko, ConvertExp, ConvertExp2,
      ConvertExtI, ConvertExtract, ConvertFloor, ConvertFma, ConvertFor,
      ConvertGetGlobal, ConvertGetIndexSpaceShape, ConvertGetTensorShape,
      ConvertGlobal, ConvertIf, ConvertIota, ConvertJoinTokens,
      ConvertLoadPtrTkoRanked, ConvertLoadPtrTkoScalar, ConvertLoadViewTko,
      ConvertLog, ConvertLog2, ConvertMakeGatherScatterView,
      ConvertMakePartitionView, ConvertMakeStridedView, ConvertMakeTensorView,
      ConvertMakeToken, ConvertMaxF, ConvertMaxI, ConvertMinF, ConvertMinI,
      ConvertMmaF, ConvertMmaI, ConvertMulhiI, ConvertMulI, ConvertOffsetRanked,
      ConvertOffsetScalarPtr, ConvertNegF, ConvertNegI, ConvertOrI, ConvertPack,
      ConvertPermute, ConvertPow, ConvertPtrToPtrCastOrFail, ConvertReduce,
      ConvertRemF, ConvertRemI, ConvertReshape, ConvertRsqrt, ConvertScan,
      ConvertSelect, ConvertShLI, ConvertShRI, ConvertSin, ConvertSinH,
      ConvertSqrt, ConvertStorePtrTkoRanked, ConvertStorePtrTkoScalar,
      ConvertStoreViewTko, ConvertSubI, ConvertTan, ConvertTanH, ConvertTruncI,
      ConvertUnpack, ConvertXOrI, ConvertYield>(converter, ctx);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

/// Pass driver for lowering CudaTile IR to GPU/vector/scf/arith/memref.
struct ConvertTileIRToMLIRPass
    : public impl::ConvertTileIRToMLIRPassBase<ConvertTileIRToMLIRPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    TypeConverter typeConverter;
    populateTileIRToMLIRTypeConverter(typeConverter, ctx, target);

    RewritePatternSet patterns(ctx);
    populateTileIRToMLIRConversionPatterns(typeConverter, patterns, target);

    ConversionTarget conversionTarget(*ctx);

    // GPU/vector/arith/scf/memref/ub/func ops are legal.
    if (target == CudaTileTarget::GPU)
      conversionTarget.addLegalDialect<gpu::GPUDialect>();
    else
      conversionTarget.addLegalDialect<func::FuncDialect>();
    conversionTarget.addLegalDialect<arith::ArithDialect, math::MathDialect,
                                     memref::MemRefDialect, scf::SCFDialect,
                                     ub::UBDialect, vector::VectorDialect>();
    conversionTarget.addLegalOp<UnrealizedConversionCastOp>();

    // CudaTile ops are illegal (target of conversion).
    conversionTarget.addIllegalDialect<cuda_tile::CudaTileDialect>();

    if (failed(applyPartialConversion(module, conversionTarget,
                                      std::move(patterns))))
      return signalPassFailure();

    // Fold muli(divui(x, c), c) -> x. These patterns are produced when
    // ConvertFor rescales loop bounds and inserts a divui in the body; once
    // a transfer op then multiplies the recovered tile-space index by the
    // same tile size, the round-trip collapses. The divui result may be
    // wrapped in index_cast ops, and the two `c` operands may be distinct
    // constant ops with the same value.
    module.walk([](arith::MulIOp op) {
      for (auto [mulOperand, otherOperand] :
           {std::pair(op.getLhs(), op.getRhs()),
            std::pair(op.getRhs(), op.getLhs())}) {
        Value v = mulOperand;
        while (auto cast = v.getDefiningOp<arith::IndexCastOp>())
          v = cast.getIn();
        auto divOp = v.getDefiningOp<arith::DivUIOp>();
        if (!divOp)
          continue;
        if (divOp.getRhs() == otherOperand) {
          op.replaceAllUsesWith(divOp.getLhs());
          op->erase();
          return;
        }
        auto divCst = divOp.getRhs().getDefiningOp<arith::ConstantIndexOp>();
        auto mulCst = otherOperand.getDefiningOp<arith::ConstantIndexOp>();
        if (divCst && mulCst && divCst.value() == mulCst.value()) {
          op.replaceAllUsesWith(divOp.getLhs());
          op->erase();
          return;
        }
      }
    });

    // Erase unrealized_conversion_casts left dead by pattern application,
    // to a fixed point.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> toErase;
      module.walk([&](UnrealizedConversionCastOp op) {
        if (op->use_empty())
          toErase.push_back(op);
      });
      for (auto *op : toErase) {
        op->erase();
        changed = true;
      }
    }

    // Mark the module as a GPU container module when targeting the GPU. For the
    // CPU target the GPU container-module marker is intentionally omitted.
    if (target == CudaTileTarget::GPU)
      module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                      UnitAttr::get(ctx));
  }
};

} // namespace
