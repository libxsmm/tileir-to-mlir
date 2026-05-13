//===- TileIRToGPU.cpp - CudaTile IR to GPU conversion ----------*- C++ -*-===//
//
// Conversion pass from CudaTile IR to GPU/vector/scf/arith/memref ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/CudaTileToGPU/CudaTileToGPU.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Metadata populated during preprocessing
//===----------------------------------------------------------------------===//

namespace {

/// Metadata for a tensor_view, populated during analysis before any IR mutation.
struct TensorViewMetadata {
  Value baseValue; // Root base pointer SSA value (may or may not be a
                   // BlockArgument)
  MemRefType memrefTy; // The memref type derived from tensor_view
                       // shape/strides
  Value memref;        // The gpu.func memref argument (set during entry
                       // conversion)
};

/// Map from make_tensor_view Operation* to its metadata.
using TensorViewMap = DenseMap<Operation *, TensorViewMetadata>;

/// Information about a partition view extracted from the IR.
struct PartitionViewInfo {
  Value memref;                   // The memref value in the converted IR
  SmallVector<int64_t> tileShape; // Tile dimensions
  SmallVector<int32_t> dimMap;    // Mapping from tile dims to tensor_view dims
  unsigned tensorViewRank;        // Rank of the underlying tensor_view
  // Optional padding value attribute from the partition_view type; null if the
  // partition view does not specify one (i.e. OOB loads yield unspecified).
  cuda_tile::PaddingValueAttr paddingValue;
};

/// Trace through assume ops to find the defining op.
static Value traceToDefThroughAssumes(Value val) {
  while (val) {
    if (auto assumeOp =
            dyn_cast_or_null<cuda_tile::AssumeOp>(val.getDefiningOp())) {
      val = assumeOp.getValue();
      continue;
    }
    break;
  }
  return val;
}

/// Resolve a partition_view SSA value to a PartitionViewInfo:
/// walk back from `partitionView` to its defining `make_partition_view` and
/// the underlying `make_tensor_view`, look up the latter in `tvMap`, and
/// collect the tile shape, dim_map, tensor-view rank, and the converted
/// memref.
static FailureOr<PartitionViewInfo>
getPartitionViewInfo(ConversionPatternRewriter &rewriter, Operation *op,
                     Value partitionView, const TensorViewMap &tvMap) {
  auto pvOp = dyn_cast_or_null<cuda_tile::MakePartitionViewOp>(
      partitionView.getDefiningOp());
  if (!pvOp)
  return rewriter.notifyMatchFailure(
    op,
        "partition view operand is not defined by a make_partition_view op");

  auto tvOp = dyn_cast_or_null<cuda_tile::MakeTensorViewOp>(
      pvOp.getTensorView().getDefiningOp());
  if (!tvOp)
  return rewriter.notifyMatchFailure(
    op,
        "underlying tensor view is not defined by a make_tensor_view op");

  auto it = tvMap.find(tvOp.getOperation());
  if (it == tvMap.end())
  return rewriter.notifyMatchFailure(
    op,
        "make_tensor_view not present in TensorViewMap (analysis missed it)");

  if (!it->second.memref)
  return rewriter.notifyMatchFailure(
    op,
        "tensor view has no associated memref yet "
        "(cuda_tile.entry not yet converted to gpu.func)");

  auto pvType = cast<cuda_tile::PartitionViewType>(pvOp.getType());
  auto tvType = pvType.getTensorView();

  PartitionViewInfo info;
  auto tileShapeAttr = pvType.getTileShape();
  for (auto v : tileShapeAttr.asArrayRef())
    info.tileShape.push_back(v);
  info.dimMap.assign(pvType.getDimMap().begin(), pvType.getDimMap().end());
  info.tensorViewRank = tvType.getShape().size();
  info.memref = it->second.memref;
  info.paddingValue = pvType.getPaddingValue();

  return info;
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//
// Conversion patterns for MakeTensorView and MakePartitionView are not needed.
// These ops are kept legal during dialect conversion and cleaned up afterward.
//===----------------------------------------------------------------------===//

/// Convert cuda_tile.constant to arith/vector constants.
///   - Scalar tiles (rank 0): Convert to scalar arith ops
///   - Ranked tiles: Convert to vector constants
///     - Splat values -> vector.broadcast
///     - Dense values -> arith.constant with DenseElementsAttr
///   - Scalar integer conversion must check if target type is IndexType, as MLIR
///     uses IndexType for loop bounds and array subscripts.
///   - Pointer types in scalar tiles are intermediate and will become dead after
///     make_tensor_view ops are erased; conversion preserves them as-is.
///   - Dense non-splat vectors require extracting all attributes and
///     reconstructing.
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
        if (isa<IndexType>(resultType)) {
          rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(
              op, splat.getSExtValue());
        } else {
          rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(
              op, resultType, splat.getSExtValue());
        }
      } else if (isa<FloatType>(elemTy)) {
        auto splat = denseVal.getSplatValue<APFloat>();
        rewriter.replaceOpWithNewOp<arith::ConstantFloatOp>(
            op, cast<FloatType>(resultType), splat);
      } else {
        return failure();
      }
    } else {
      // Tile -> vector splat or dense constant
      auto vecTy = cast<VectorType>(resultType);
      if (denseVal.isSplat()) {
        auto elemTy = tileType.getElementType();
        Value splatVal;
        if (isa<IntegerType>(elemTy)) {
          auto splat = denseVal.getSplatValue<APInt>();
          splatVal = arith::ConstantIntOp::create(rewriter, loc,
                                                  elemTy, splat.getSExtValue());
        } else if (isa<FloatType>(elemTy)) {
          auto splat = denseVal.getSplatValue<APFloat>();
          splatVal = arith::ConstantFloatOp::create(
              rewriter, loc, cast<FloatType>(elemTy), splat);
        } else {
          return failure();
        }
        rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, vecTy, splatVal);
      } else {
        // Non-splat dense: create a vector constant directly
        SmallVector<Attribute> attrs(denseVal.getValues<Attribute>().begin(),
                                     denseVal.getValues<Attribute>().end());
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(
            op, vecTy, DenseElementsAttr::get(vecTy, attrs));
      }
    }
    return success();
  }
};

/// Convert cuda_tile.get_tile_block_id to gpu.block_id x/y/z.
///   - Creates three gpu.block_id operations, one per dimension (x, y, z).
struct ConvertGetTileBlockId
    : public OpConversionPattern<cuda_tile::GetTileBlockIdOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::GetTileBlockIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value bx = gpu::BlockIdOp::create(rewriter, loc, gpu::Dimension::x);
    Value by = gpu::BlockIdOp::create(rewriter, loc, gpu::Dimension::y);
    Value bz = gpu::BlockIdOp::create(rewriter, loc, gpu::Dimension::z);
    rewriter.replaceOp(op, {bx, by, bz});
    return success();
  }
};

/// Convert cuda_tile.muli to arith.muli.
struct ConvertMulI : public OpConversionPattern<cuda_tile::MulIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::MulIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return failure();
    rewriter.replaceOpWithNewOp<arith::MulIOp>(op, adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};

/// Convert cuda_tile.for to scf.for.
///   1. Create scf.ForOp with converted bounds and initial values.
///   2. Convert region types in old body using type converter.
///   3. Remove auto-generated yield in new body.
///   4. Merge old body into new body, replacing block args (induction var +
///      iter args).
///   5. Replace old op with new ForOp results.
struct ConvertFor : public OpConversionPattern<cuda_tile::ForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newForOp = scf::ForOp::create(
        rewriter, op.getLoc(), adaptor.getLowerBound(),
        adaptor.getUpperBound(), adaptor.getStep(), adaptor.getInitValues());

    // Convert region types
    if (failed(rewriter.convertRegionTypes(&op.getRegion(),
                                           *getTypeConverter())))
      return failure();

    // Merge old body into new body
    Block *oldBody = op.getBody();
    Block *newBody = newForOp.getBody();

    // Remove auto-generated yield in new body
    if (newBody->mightHaveTerminator())
      rewriter.eraseOp(newBody->getTerminator());

    SmallVector<Value> replacingValues;
    replacingValues.push_back(newForOp.getInductionVar());
    for (auto arg : newForOp.getRegionIterArgs())
      replacingValues.push_back(arg);

    rewriter.mergeBlocks(oldBody, newBody, replacingValues);
    rewriter.replaceOp(op, newForOp.getResults());
    return success();
  }
};

/// Convert cuda_tile.continue to scf.yield.
struct ConvertContinue : public OpConversionPattern<cuda_tile::ContinueOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ContinueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

/// Convert cuda_tile.return to gpu.return.
struct ConvertReturn : public OpConversionPattern<cuda_tile::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<gpu::ReturnOp>(op);
    return success();
  }
};

/// Convert cuda_tile.mmaf to vector.contract (matmul-style contraction).
///
/// Source-op semantics (from cuda_tile.mmaf verifier):
///   Unbatched: lhs[M,K] x rhs[K,N] + acc[M,N] -> result[M,N]
///   Batched:   lhs[B,M,K] x rhs[B,K,N] + acc[B,M,N] -> result[B,M,N]
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
    int64_t rank = vecResultTy.getRank();
    if (rank != 2 && rank != 3)
      return rewriter.notifyMatchFailure(
          op,
          "only 2D or 3D (batched) mmaf is supported");

    auto *ctx = rewriter.getContext();
    bool batched = (rank == 3);

    // Iteration space:
    //   Unbatched (3 dims): d0=m, d1=n, d2=k
    //   Batched   (4 dims): d0=b, d1=m, d2=n, d3=k
    AffineMap mapA, mapB, mapC;
    SmallVector<Attribute> iterTypes;
    auto parAttr =
        vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel);
    auto redAttr =
        vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction);
    auto d0 = getAffineDimExpr(0, ctx);
    auto d1 = getAffineDimExpr(1, ctx);
    auto d2 = getAffineDimExpr(2, ctx);
    if (!batched) {
      mapA = AffineMap::get(3, 0, {d0, d2}, ctx);
      mapB = AffineMap::get(3, 0, {d2, d1}, ctx);
      mapC = AffineMap::get(3, 0, {d0, d1}, ctx);
      iterTypes = {parAttr, parAttr, redAttr};
    } else {
      auto d3 = getAffineDimExpr(3, ctx);
      mapA = AffineMap::get(4, 0, {d0, d1, d3}, ctx);
      mapB = AffineMap::get(4, 0, {d0, d3, d2}, ctx);
      mapC = AffineMap::get(4, 0, {d0, d1, d2}, ctx);
      iterTypes = {parAttr, parAttr, parAttr, redAttr};
    }

    // Explicit combining kind = add (mmaf is multiply-accumulate).
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        op, adaptor.getLhs(), adaptor.getRhs(), adaptor.getAcc(),
        rewriter.getAffineMapArrayAttr({mapA, mapB, mapC}),
        rewriter.getArrayAttr(iterTypes), vector::CombiningKind::ADD);
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

/// Convert cuda_tile.get_index_space_shape.
/// For
/// partition_view<tile=(T0xT1x...), tensor_view<?x?x...>, dim_map=[d0,d1,...]>
/// index_space_shape[i] = ceildiv(tensor_shape[dimMap[i]], tileShape[i]).
struct ConvertGetIndexSpaceShape
    : public OpConversionPattern<cuda_tile::GetIndexSpaceShapeOp> {
  const TensorViewMap &tvMap;

  ConvertGetIndexSpaceShape(TypeConverter &tc, MLIRContext *ctx,
                            const TensorViewMap &tvMap)
      : OpConversionPattern(tc, ctx), tvMap(tvMap) {}

  LogicalResult
  matchAndRewrite(cuda_tile::GetIndexSpaceShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value srcView = op.getSrc();
    auto pvInfo = getPartitionViewInfo(rewriter, op, srcView, tvMap);
    if (failed(pvInfo))
      return failure();

    Location loc = op.getLoc();
    unsigned rank = pvInfo->tileShape.size();

    // For each tile dimension i:
    // - The corresponding tensor_view dimension is dimMap[i]
    // - index_space_dim_i = ceildiv(memref.dim(dimMap[i]), tileShape[i])
    SmallVector<Value> results;
    for (unsigned i = 0; i < rank; ++i) {
      int64_t tileSize = pvInfo->tileShape[i];
      unsigned tensorDim = pvInfo->dimMap[i];
      Value dimVal =
          memref::DimOp::create(rewriter, loc, pvInfo->memref,
                                arith::ConstantIndexOp::create(
                                    rewriter, loc, tensorDim));
      Value tileSizeVal =
          arith::ConstantIndexOp::create(rewriter, loc, tileSize);
      Value divResult =
          arith::CeilDivUIOp::create(rewriter, loc, dimVal, tileSizeVal);
      results.push_back(divResult);
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

/// Materialize the constant value implied by a partition_view padding_value
/// attribute. Falls back to ub.poison when no padding_value was specified;
/// in that case the spec leaves OOB elements unspecified. The special enum
/// cases (neg_zero / nan / pos_inf / neg_inf) are only valid for floating
/// point element types per the dialect verifier.
static Value makePaddingValue(OpBuilder &b, Location loc, Type elemTy,
                              cuda_tile::PaddingValueAttr pvAttr) {
  if (!pvAttr)
    return ub::PoisonOp::create(b, loc, elemTy);

  auto fty = dyn_cast<FloatType>(elemTy);
  if (!fty) {
    // Integer element types only accept `zero`; emit a 0 constant.
    return arith::ConstantIntOp::create(b, loc, elemTy, 0);
  }

  const llvm::fltSemantics &sem = fty.getFloatSemantics();
  APFloat val = APFloat::getZero(sem, /*Negative=*/false);
  switch (pvAttr.getValue()) {
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
  return arith::ConstantFloatOp::create(b, loc, fty, val);
}

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
/// Restrictions / guards (return notifyMatchFailure on violation):
///   - Only `weak` memory_ordering_semantics is supported. Relaxed/acquire
///     have no equivalent in `vector.transfer_read`.
///   - `memory_scope` is not supported.
///   - `result_token` must be unused: this lowering drops the token, so any
///     consumer would be left dangling.
///     (`cuda_tile.assume`/no consumers is the common case after the rest of
///     the conversion runs.)
struct ConvertLoadViewTko
    : public OpConversionPattern<cuda_tile::LoadViewTkoOp> {
  const TensorViewMap &tvMap;

  ConvertLoadViewTko(TypeConverter &tc, MLIRContext *ctx,
                     const TensorViewMap &tvMap)
      : OpConversionPattern(tc, ctx), tvMap(tvMap) {}

  LogicalResult
  matchAndRewrite(cuda_tile::LoadViewTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    // Guard: unsupported memory ordering / scope.
    if (op.getMemoryOrderingSemantics() !=
        cuda_tile::MemoryOrderingSemantics::WEAK)
      return rewriter.notifyMatchFailure(
        op,
          "only `weak` memory_ordering_semantics is supported");
    if (op.getMemoryScope())
      return rewriter.notifyMatchFailure(
        op,
          "memory_scope is not supported by this lowering");

    // Guard: token result must be unused (we cannot materialize a token).
    if (!op.getResultToken().use_empty())
      return rewriter.notifyMatchFailure(
          op,
          "result_token has live uses; this lowering drops the token");

    auto pvInfo = getPartitionViewInfo(rewriter, op, op.getView(), tvMap);
    if (failed(pvInfo))
      return failure();

    auto vecTy = cast<VectorType>(
        getTypeConverter()->convertType(op.getTile().getType()));

    auto convertedIndices = adaptor.getIndex();
    unsigned tileRank = pvInfo->tileShape.size();
    unsigned tensorRank = pvInfo->tensorViewRank;

    if ((unsigned)vecTy.getRank() != tileRank)
      return rewriter.notifyMatchFailure(
          op,
          "converted tile rank does not match partition tile_shape rank");

    // Build memref indices in tensor-dimension order.
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> memrefIndices(tensorRank, zero);
    for (unsigned i = 0; i < tileRank; ++i) {
      unsigned tensorDim = pvInfo->dimMap[i];
      int64_t tileSize = pvInfo->tileShape[i];
      Value tileSizeVal =
          arith::ConstantIndexOp::create(rewriter, loc, tileSize);
        Value elemOffset = arith::MulIOp::create(
          rewriter, loc, convertedIndices[i], tileSizeVal);
      memrefIndices[tensorDim] = elemOffset;
    }

    // permutation_map: (d0,...,d_{tensorRank-1}) -> (d_{dimMap[0]},...,
    // d_{dimMap[tileRank-1]}). This makes vector dim i correspond to memref
    // dim dimMap[i], handling both transposed dim_map and partial-rank tiles
    // (tileRank < tensorRank) without an extra vector.transpose.
    SmallVector<AffineExpr> permExprs;
    permExprs.reserve(tileRank);
    for (int32_t td : pvInfo->dimMap)
      permExprs.push_back(getAffineDimExpr(td, ctx));
    auto permMap = AffineMap::get(tensorRank, 0, permExprs, ctx);

    // inBounds[i] is true only when we can statically prove the tile fits in
    // the corresponding memref dim. Otherwise transfer_read masks and uses
    // the padding value for OOB lanes.
    auto memrefTy = cast<MemRefType>(pvInfo->memref.getType());
    auto memrefShape = memrefTy.getShape();
    SmallVector<bool> inBounds(tileRank, false);
    for (unsigned i = 0; i < tileRank; ++i) {
      int64_t ms = memrefShape[pvInfo->dimMap[i]];
      int64_t ts = pvInfo->tileShape[i];
      inBounds[i] =
          (ms != ShapedType::kDynamic && ts > 0 && (ms % ts) == 0);
    }

    Value padding = makePaddingValue(rewriter, loc, vecTy.getElementType(),
                                     pvInfo->paddingValue);
    auto readOp = vector::TransferReadOp::create(
        rewriter, loc, vecTy, pvInfo->memref, memrefIndices,
        AffineMapAttr::get(permMap), padding,
      /*mask=*/Value(), rewriter.getBoolArrayAttr(inBounds));

    // result_token has no live uses (guarded above); drop it.
    rewriter.replaceOp(op, {readOp.getResult(), Value()});
    return success();
  }
};

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
struct ConvertStoreViewTko
    : public OpConversionPattern<cuda_tile::StoreViewTkoOp> {
  const TensorViewMap &tvMap;

  ConvertStoreViewTko(TypeConverter &tc, MLIRContext *ctx,
                      const TensorViewMap &tvMap)
      : OpConversionPattern(tc, ctx), tvMap(tvMap) {}

  LogicalResult
  matchAndRewrite(cuda_tile::StoreViewTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    if (op.getMemoryOrderingSemantics() !=
        cuda_tile::MemoryOrderingSemantics::WEAK)
      return rewriter.notifyMatchFailure(
        op,
          "only `weak` memory_ordering_semantics is supported");
    if (op.getMemoryScope())
      return rewriter.notifyMatchFailure(
        op,
          "memory_scope is not supported by this lowering");
    if (!op.getResultToken().use_empty())
      return rewriter.notifyMatchFailure(
        op,
          "result_token has live uses; this lowering drops the token");

    auto pvInfo = getPartitionViewInfo(rewriter, op, op.getView(), tvMap);
    if (failed(pvInfo))
      return failure();

    auto vecTy = cast<VectorType>(
        getTypeConverter()->convertType(op.getTile().getType()));

    auto convertedIndices = adaptor.getIndex();
    unsigned tileRank = pvInfo->tileShape.size();
    unsigned tensorRank = pvInfo->tensorViewRank;

    if ((unsigned)vecTy.getRank() != tileRank)
      return rewriter.notifyMatchFailure(
          op,
          "converted tile rank does not match partition tile_shape rank");

    // Build memref indices in tensor-dimension order.
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> memrefIndices(tensorRank, zero);
    for (unsigned i = 0; i < tileRank; ++i) {
      unsigned tensorDim = pvInfo->dimMap[i];
      int64_t tileSize = pvInfo->tileShape[i];
      Value tileSizeVal =
          arith::ConstantIndexOp::create(rewriter, loc, tileSize);
        Value elemOffset = arith::MulIOp::create(
          rewriter, loc, convertedIndices[i], tileSizeVal);
      memrefIndices[tensorDim] = elemOffset;
    }

    // permutation_map: vector dim i corresponds to memref dim dimMap[i].
    SmallVector<AffineExpr> permExprs;
    permExprs.reserve(tileRank);
    for (int32_t td : pvInfo->dimMap)
      permExprs.push_back(getAffineDimExpr(td, ctx));
    auto permMap = AffineMap::get(tensorRank, 0, permExprs, ctx);

    auto memrefTy = cast<MemRefType>(pvInfo->memref.getType());
    auto memrefShape = memrefTy.getShape();
    SmallVector<bool> inBounds(tileRank, false);
    for (unsigned i = 0; i < tileRank; ++i) {
      int64_t ms = memrefShape[pvInfo->dimMap[i]];
      int64_t ts = pvInfo->tileShape[i];
      inBounds[i] =
          (ms != ShapedType::kDynamic && ts > 0 && (ms % ts) == 0);
    }

    auto writeOp = vector::TransferWriteOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{}, adaptor.getTile(),
        pvInfo->memref, memrefIndices, AffineMapAttr::get(permMap),
        /*mask=*/Value(), rewriter.getBoolArrayAttr(inBounds));
    (void)writeOp;

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Preprocessing: cuda_tile.module/entry -> gpu.module/gpu.func
//===----------------------------------------------------------------------===//

/// Analyze all make_tensor_view ops in the module and populate the
/// TensorViewMap with their root base values and derived memref types.
/// The memref value will be set in later conversions when they get instantiated.
/// This is a read-only analysis step that does not mutate the IR.
static void buildTensorViewMap(ModuleOp module, TensorViewMap &tvMap) {
  MLIRContext *ctx = module.getContext();

  module.walk([&](cuda_tile::MakeTensorViewOp tvOp) {
    Value base = traceToDefThroughAssumes(tvOp.getBase());
    // Trace through casts to find the root base value.
    Value rootBase = base;
    while (rootBase) {
      if (isa<BlockArgument>(rootBase))
        break;
      if (auto castOp = dyn_cast_or_null<UnrealizedConversionCastOp>(
              rootBase.getDefiningOp())) {
        if (castOp.getNumOperands() == 1) {
          rootBase = castOp.getOperand(0);
          continue;
        }
      }
      break;
    }

    // Get element type from the base pointer type.
    auto baseTy = cast<cuda_tile::TileType>(tvOp.getBase().getType());
    auto ptrTy = cast<cuda_tile::PointerType>(baseTy.getElementType());
    Type elemTy = ptrTy.getPointeeType();

    auto tvType = cast<cuda_tile::TensorViewType>(tvOp.getType());
    auto tvShape = tvType.getShape();
    auto tvStrides = tvType.getStrides();
    unsigned rank = tvShape.size();

    // Build shape/strides for memref type from tensor_view.
    SmallVector<int64_t> memrefShape(tvShape.begin(), tvShape.end());
    SmallVector<int64_t> memrefStrides(tvStrides.begin(), tvStrides.end());

    // Check if the layout is the default contiguous row-major layout.
    // If so, use a plain memref without explicit strides.
    bool isDefaultLayout = true;
    if (rank > 0) {
      for (unsigned i = 0; i < rank; ++i) {
        if (memrefStrides[i] == ShapedType::kDynamic ||
            memrefShape[i] == ShapedType::kDynamic) {
          isDefaultLayout = false;
          break;
        }
      }
      if (isDefaultLayout) {
        int64_t expected = 1;
        for (int i = rank - 1; i >= 0; --i) {
          if (memrefStrides[i] != expected) {
            isDefaultLayout = false;
            break;
          }
          expected *= memrefShape[i];
        }
      }
    }

    MemRefType memrefTy;
    if (isDefaultLayout) {
      memrefTy = MemRefType::get(memrefShape, elemTy);
    } else {
      auto layout = StridedLayoutAttr::get(ctx, /*offset=*/0, memrefStrides);
      memrefTy = MemRefType::get(memrefShape, elemTy, layout);
    }

    tvMap[tvOp.getOperation()] = TensorViewMetadata{rootBase, memrefTy, {}};
  });
}

/// Convert cuda_tile.entry to gpu.func. For each tensor_view whose base is an
/// entry block argument, a memref function parameter is created; the
/// TensorViewMap's memref field is filled in here. Other patterns that depend
/// on `meta.memref` will retry until this pattern runs.
struct ConvertEntry : public OpConversionPattern<cuda_tile::EntryOp> {
  TensorViewMap &tvMap;

  ConvertEntry(TypeConverter &tc, MLIRContext *ctx, TensorViewMap &tvMap)
      : OpConversionPattern(tc, ctx), tvMap(tvMap) {}

  LogicalResult
  matchAndRewrite(cuda_tile::EntryOp entryOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = entryOp.getContext();
    Location loc = entryOp.getLoc();
    Block *entryBlock = &entryOp.getBody().front();

    // Collect the tensor_view ops in this entry whose base is an entry block
    // argument, and the set of pointer-arg indices they cover.
    SmallVector<Operation *> blockArgTvOps;
    DenseSet<unsigned> mappedArgIndices;
    for (auto &[op, meta] : tvMap) {
      if (op->getParentRegion() != &entryOp.getBody())
        continue;
      if (auto blockArg = dyn_cast<BlockArgument>(meta.baseValue)) {
        if (blockArg.getOwner() == entryBlock) {
          blockArgTvOps.push_back(op);
          mappedArgIndices.insert(blockArg.getArgNumber());
        }
      }
    }

    // Sort by entry arg index for deterministic gpu.func parameter ordering.
    llvm::sort(blockArgTvOps, [&](Operation *a, Operation *b) {
      return cast<BlockArgument>(tvMap[a].baseValue).getArgNumber() <
             cast<BlockArgument>(tvMap[b].baseValue).getArgNumber();
    });

    // Verify that every pointer-typed entry block arg is covered.
    for (unsigned i = 0; i < entryBlock->getNumArguments(); ++i) {
      auto argTy = entryBlock->getArgument(i).getType();
      if (auto tileTy = dyn_cast<cuda_tile::TileType>(argTy)) {
        if (isa<cuda_tile::PointerType>(tileTy.getElementType())) {
          if (!mappedArgIndices.contains(i))
            return rewriter.notifyMatchFailure(
                entryOp, "pointer entry arg has no corresponding "
                         "make_tensor_view");
        }
      }
    }

    // Build gpu.func argument types (one memref per block-arg-based view).
    SmallVector<Type> gpuFuncArgTypes;
    for (auto *op : blockArgTvOps)
      gpuFuncArgTypes.push_back(tvMap[op].memrefTy);

    // Create the gpu.func before the entry op.
    auto gpuFuncType = FunctionType::get(ctx, gpuFuncArgTypes, {});
    auto gpuFunc = gpu::GPUFuncOp::create(rewriter, loc, entryOp.getSymName(),
                                          gpuFuncType);
    gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                     rewriter.getUnitAttr());

    Block *gpuBlock = &gpuFunc.getBody().front();
    if (gpuBlock->getNumArguments() == 0) {
      for (auto ty : gpuFuncArgTypes)
        gpuBlock->addArgument(ty, loc);
    }

    rewriter.setInsertionPointToStart(gpuBlock);

    // Build replacement values for each entry block arg.
    // - For pointer args used by make_tensor_view: cast memref -> tile<ptr<T>>
    //   (deduped so multiple views sharing one ptr arg reuse the cast).
    // - For other args (dims/strides) used by make_tensor_view/assume:
    //   placeholder cast (will become dead after view ops are cleaned up).
    SmallVector<Value> argReplacements(entryBlock->getNumArguments());
    for (unsigned i = 0; i < blockArgTvOps.size(); ++i) {
      auto &meta = tvMap[blockArgTvOps[i]];
      unsigned ptrArgIdx =
          cast<BlockArgument>(meta.baseValue).getArgNumber();
      Value memrefArg = gpuBlock->getArgument(i);

      // Fill in the memref value so the remaining patterns can use it.
      meta.memref = memrefArg;

      if (argReplacements[ptrArgIdx])
        continue; // already produced cast for this ptr arg

      auto elemTy = meta.memrefTy.getElementType();
      auto ptrTy = cuda_tile::PointerType::get(elemTy);
      auto tilePtrTy = cuda_tile::TileType::get({}, ptrTy);
      argReplacements[ptrArgIdx] =
          UnrealizedConversionCastOp::create(rewriter, loc, tilePtrTy,
                                             memrefArg)
              .getResult(0);
    }

    for (unsigned i = 0; i < entryBlock->getNumArguments(); ++i) {
      if (argReplacements[i])
        continue;
      Value arg = entryBlock->getArgument(i);

      // Verify all uses are by view-construction or assume ops (kept legal).
      for (OpOperand &use : arg.getUses()) {
        if (!isa<cuda_tile::MakeTensorViewOp, cuda_tile::AssumeOp>(
                use.getOwner()))
          return rewriter.notifyMatchFailure(
              entryOp, "entry arg has unexpected use");
      }

      argReplacements[i] =
          UnrealizedConversionCastOp::create(rewriter, loc, arg.getType(),
                                             gpuBlock->getArgument(0))
              .getResult(0);
    }

    // Move ops from entry block into gpu.func block, replacing entry block
    // args with the constructed replacements.
    rewriter.inlineBlockBefore(entryBlock, gpuBlock, gpuBlock->end(),
                               argReplacements);

    rewriter.eraseOp(entryOp);
    return success();
  }
};

/// Convert cuda_tile.module to gpu.module by moving its body contents.
struct ConvertModule : public OpConversionPattern<cuda_tile::ModuleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ModuleOp cudaMod, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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

    rewriter.eraseOp(cudaMod);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Type converter and conversion pattern population
//===----------------------------------------------------------------------===//

static void populateTileIRToGPUTypeConverter(TypeConverter &converter,
                                            MLIRContext *ctx) {
  // Fallback: keep types unchanged.
  converter.addConversion([](Type type) { return type; });

  // cuda_tile.tile<MxNxelemTy> -> vector<MxNxelemTy> (ranked tiles)
  // cuda_tile.tile<elemTy> (scalar, rank 0):
  //   - i32 -> index (index-like scalar in tile IR)
  //   - other integers/floats -> preserved scalar type
  // cuda_tile.tile<ptr<elemTy>> (scalar, pointer) -> kept as-is (intermediate)
  converter.addConversion([ctx](cuda_tile::TileType tileTy) -> Type {
    auto shape = tileTy.getShape();
    auto elemTy = tileTy.getElementType();

    if (shape.empty()) {
      // Scalar tile
      if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
        // Keep i32 as index because tile-level loop bounds/indices lower to
        // SCF/memref index arithmetic, but preserve all other integer widths.
        if (intTy.getWidth() == 32)
          return IndexType::get(ctx);
        return elemTy;
      }
      if (isa<FloatType>(elemTy))
        return elemTy;
      // Pointer types in scalar tiles -> keep as-is (will be dead after
      // make_tensor_view is erased)
      if (isa<cuda_tile::PointerType>(elemTy))
        return tileTy;
      return Type();
    }
    // Ranked tile -> vector
    return VectorType::get(shape, elemTy);
  });

  // View types are kept as-is during conversion. The load/store patterns
  // need to inspect the original view types. They'll be dead after conversion.
  converter.addConversion(
      [](cuda_tile::TensorViewType tvTy) -> Type { return tvTy; });
  converter.addConversion(
      [](cuda_tile::PartitionViewType pvTy) -> Type { return pvTy; });
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

static void populateTileIRToGPUConversionPatterns(
    TypeConverter &converter, RewritePatternSet &patterns,
    TensorViewMap &tvMap) {
  MLIRContext *ctx = patterns.getContext();
  // Patterns that don't need the tvMap.
  patterns.add<ConvertModule, ConvertConstant, ConvertGetTileBlockId,
               ConvertMulI, ConvertFor, ConvertContinue, ConvertReturn,
               ConvertMmaF, ConvertAssume>(converter, ctx);
  // Patterns that need the tvMap.
  patterns.add<ConvertEntry, ConvertGetIndexSpaceShape, ConvertLoadViewTko,
               ConvertStoreViewTko>(converter, ctx, tvMap);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct ConvertTileIRToGPUPass
    : public PassWrapper<ConvertTileIRToGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTileIRToGPUPass)

  StringRef getArgument() const override {
    return "convert-cuda-tile-to-gpu";
  }

  StringRef getDescription() const override {
    return "Convert CudaTile IR to GPU/vector/scf/arith ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cuda_tile::CudaTileDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<gpu::GPUDialect>();
    registry.insert<ub::UBDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    // Step 1: Analyze all make_tensor_view ops and build the full map.
    // This is a read-only pass over the IR before any mutations.
    TensorViewMap tvMap;
    buildTensorViewMap(module, tvMap);

    // Step 2: Run dialect conversion. ConvertModule rewrites
    // cuda_tile.module -> gpu.module; ConvertEntry rewrites
    // cuda_tile.entry -> gpu.func and fills in tvMap[].memref;
    // Other patterns that depend on it will retry until then.
    TypeConverter typeConverter;
    populateTileIRToGPUTypeConverter(typeConverter, ctx);

    RewritePatternSet patterns(ctx);
    populateTileIRToGPUConversionPatterns(typeConverter, patterns, tvMap);

    ConversionTarget target(*ctx);

    // GPU/vector/arith/scf/memref/ub ops are legal.
    target.addLegalDialect<arith::ArithDialect, gpu::GPUDialect,
                           memref::MemRefDialect, scf::SCFDialect,
                           ub::UBDialect, vector::VectorDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    // CudaTile ops are illegal (target of conversion).
    target.addIllegalDialect<cuda_tile::CudaTileDialect>();

    // But keep view-construction and assume ops legal during conversion so
    // that load/store patterns can trace through them.
    target.addLegalOp<cuda_tile::MakeTensorViewOp,
                      cuda_tile::MakePartitionViewOp, cuda_tile::AssumeOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return signalPassFailure();

    // Step 3: Remove dead ops (unrealized_conversion_casts, view ops, assumes).
    // Iterate until fixpoint to handle dependencies.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> toErase;
      module.walk([&](Operation *op) {
        if ((isa<UnrealizedConversionCastOp, cuda_tile::MakeTensorViewOp,
                 cuda_tile::MakePartitionViewOp, cuda_tile::AssumeOp>(op)) &&
            op->use_empty()) {
          toErase.push_back(op);
        }
      });
      for (auto *op : toErase) {
        op->erase();
        changed = true;
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertTileIRToGPUPass() {
  return std::make_unique<ConvertTileIRToGPUPass>();
}
