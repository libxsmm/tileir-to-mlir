//===- TileIRToGPU.cpp - CudaTile IR to GPU conversion ----------*- C++ -*-===//
//
// Conversion pass from CudaTile IR to GPU/vector/scf/arith/memref ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/CudaTileToGPU/CudaTileToGPU.h"

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
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"

using namespace mlir;

namespace {

/// Derive the ranked MemRefType that corresponds to a tensor_view type.
/// If the tensor_view describes a default contiguous row-major layout with
/// fully static shape/strides, a plain (layout-free) memref is returned;
/// otherwise a strided-layout memref is returned (which also covers any
/// dynamic shape or stride).
static MemRefType tensorViewToMemRefType(cuda_tile::TensorViewType tvTy) {
  auto shape = tvTy.getShape();
  auto strides = tvTy.getStrides();
  Type elemTy = tvTy.getElementType();
  unsigned rank = shape.size();

  bool isDefaultLayout = true;
  if (rank == 0) {
    isDefaultLayout = true;
  } else {
    for (unsigned i = 0; i < rank; ++i) {
      if (strides[i] == ShapedType::kDynamic ||
          shape[i] == ShapedType::kDynamic) {
        isDefaultLayout = false;
        break;
      }
    }
    if (isDefaultLayout) {
      int64_t expected = 1;
      for (int i = rank - 1; i >= 0; --i) {
        if (strides[i] != expected) {
          isDefaultLayout = false;
          break;
        }
        expected *= shape[i];
      }
    }
  }

  SmallVector<int64_t> memrefShape(shape.begin(), shape.end());
  if (isDefaultLayout)
    return MemRefType::get(memrefShape, elemTy);
  SmallVector<int64_t> memrefStrides(strides.begin(), strides.end());
  auto layout =
      StridedLayoutAttr::get(elemTy.getContext(), /*offset=*/0, memrefStrides);
  return MemRefType::get(memrefShape, elemTy, layout);
}

/// Information extracted from a partition_view operand at a use site.
struct PartitionViewInfo {
  Value memref;                   // Converted memref backing the partition view
  SmallVector<int64_t> tileShape; // Tile dimensions
  SmallVector<int32_t> dimMap;    // Mapping from tile dims to tensor_view dims
  unsigned tensorViewRank;        // Rank of the underlying tensor_view
  // Optional padding value attribute from the partition_view type; null if the
  // partition view does not specify one (i.e. OOB loads yield unspecified).
  cuda_tile::PaddingValueAttr paddingValue;
};

/// Extract partition-view layout info from `view`'s type and pair it with the
/// already type-converted memref `convertedView`.
static PartitionViewInfo getPartitionViewInfo(Value view, Value convertedView) {
  auto pvType = cast<cuda_tile::PartitionViewType>(view.getType());
  PartitionViewInfo info;
  info.memref = convertedView;
  for (auto v : pvType.getTileShape().asArrayRef())
    info.tileShape.push_back(v);
  info.dimMap.assign(pvType.getDimMap().begin(), pvType.getDimMap().end());
  info.tensorViewRank = pvType.getTensorView().getShape().size();
  info.paddingValue = pvType.getPaddingValue();
  return info;
}

static FailureOr<arith::CmpFPredicate>
mapCmpFPredicate(cuda_tile::ComparisonPredicate pred,
                 cuda_tile::ComparisonOrdering ord) {
  using CP = cuda_tile::ComparisonPredicate;
  using CO = cuda_tile::ComparisonOrdering;
  if (ord == CO::ORDERED) {
    switch (pred) {
    case CP::EQUAL:
      return arith::CmpFPredicate::OEQ;
    case CP::NOT_EQUAL:
      return arith::CmpFPredicate::ONE;
    case CP::LESS_THAN:
      return arith::CmpFPredicate::OLT;
    case CP::LESS_THAN_OR_EQUAL:
      return arith::CmpFPredicate::OLE;
    case CP::GREATER_THAN:
      return arith::CmpFPredicate::OGT;
    case CP::GREATER_THAN_OR_EQUAL:
      return arith::CmpFPredicate::OGE;
    }
  }
  if (ord == CO::UNORDERED) {
    switch (pred) {
    case CP::EQUAL:
      return arith::CmpFPredicate::UEQ;
    case CP::NOT_EQUAL:
      return arith::CmpFPredicate::UNE;
    case CP::LESS_THAN:
      return arith::CmpFPredicate::ULT;
    case CP::LESS_THAN_OR_EQUAL:
      return arith::CmpFPredicate::ULE;
    case CP::GREATER_THAN:
      return arith::CmpFPredicate::UGT;
    case CP::GREATER_THAN_OR_EQUAL:
      return arith::CmpFPredicate::UGE;
    }
  }
  return failure();
}

static FailureOr<arith::CmpIPredicate>
mapCmpIPredicate(cuda_tile::ComparisonPredicate pred,
                 cuda_tile::Signedness signedness) {
  using CP = cuda_tile::ComparisonPredicate;
  bool isUnsigned = signedness == cuda_tile::Signedness::Unsigned;
  switch (pred) {
  case CP::EQUAL:
    return arith::CmpIPredicate::eq;
  case CP::NOT_EQUAL:
    return arith::CmpIPredicate::ne;
  case CP::LESS_THAN:
    return isUnsigned ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt;
  case CP::LESS_THAN_OR_EQUAL:
    return isUnsigned ? arith::CmpIPredicate::ule : arith::CmpIPredicate::sle;
  case CP::GREATER_THAN:
    return isUnsigned ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::sgt;
  case CP::GREATER_THAN_OR_EQUAL:
    return isUnsigned ? arith::CmpIPredicate::uge : arith::CmpIPredicate::sge;
  }
  return failure();
}

static Value castValueToType(OpBuilder &builder, Location loc, Value value,
                             Type targetType) {
  if (value.getType() == targetType)
    return value;
  if ((isa<IndexType>(value.getType()) && isa<IntegerType>(targetType)) ||
      (isa<IntegerType>(value.getType()) && isa<IndexType>(targetType)))
    return arith::IndexCastOp::create(builder, loc, targetType, value);
  return Value();
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

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert cuda_tile.constant to arith/vector constants.
///   - Scalar tiles (rank 0): Convert to scalar arith ops
///   - Ranked tiles: Convert to vector constants
///     - Splat values -> vector.broadcast
///     - Dense values -> arith.constant with DenseElementsAttr
///   - Scalar integer conversion preserves the integer type; explicit casts to
///     `index` are inserted later at the ops that require them.
///   - Pointer types in scalar tiles are intermediate and will become dead
///   after
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
      // Tile -> vector splat or dense constant
      auto vecTy = cast<VectorType>(resultType);
      if (denseVal.isSplat()) {
        auto elemTy = tileType.getElementType();
        Value splatVal;
        if (isa<IntegerType>(elemTy)) {
          auto splat = denseVal.getSplatValue<APInt>();
          splatVal = arith::ConstantIntOp::create(rewriter, loc, elemTy,
                                                  splat.getSExtValue());
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

/// Convert a cuda_tile op that returns three i32 values (one per grid
/// dimension) into three GPU dimension-query ops (x, y, z) with index_cast.
/// Used for both get_tile_block_id -> gpu.block_id and
/// get_num_tile_blocks -> gpu.grid_dim.
template <typename SrcOp, typename GpuDimOp>
struct ConvertDimQueryOp : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op, typename OpConversionPattern<SrcOp>::OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultTy =
        this->getTypeConverter()->convertType(op.getResult(0).getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "cannot convert result type");
    Value x = castValueToType(
        rewriter, loc, GpuDimOp::create(rewriter, loc, gpu::Dimension::x),
        resultTy);
    Value y = castValueToType(
        rewriter, loc, GpuDimOp::create(rewriter, loc, gpu::Dimension::y),
        resultTy);
    Value z = castValueToType(
        rewriter, loc, GpuDimOp::create(rewriter, loc, gpu::Dimension::z),
        resultTy);
    if (!x || !y || !z)
      return rewriter.notifyMatchFailure(
          op, "cannot cast dim query results to target type");
    rewriter.replaceOp(op, {x, y, z});
    return success();
  }
};

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

template <typename SrcOp, bool IsMax>
struct ConvertMinMaxFOp : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getFlushToZero())
      return rewriter.notifyMatchFailure(
          op, IsMax ? "maxf flush_to_zero is not representable in arith max "
                      "operations"
                    : "minf flush_to_zero is not representable in arith min "
                      "operations");

    if (op.getPropagateNan()) {
      if constexpr (IsMax) {
        rewriter.template replaceOpWithNewOp<arith::MaximumFOp>(
            op, adaptor.getLhs(), adaptor.getRhs());
      } else {
        rewriter.template replaceOpWithNewOp<arith::MinimumFOp>(
            op, adaptor.getLhs(), adaptor.getRhs());
      }
    } else {
      if constexpr (IsMax) {
        rewriter.template replaceOpWithNewOp<arith::MaxNumFOp>(
            op, adaptor.getLhs(), adaptor.getRhs());
      } else {
        rewriter.template replaceOpWithNewOp<arith::MinNumFOp>(
            op, adaptor.getLhs(), adaptor.getRhs());
      }
    }
    return success();
  }
};

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

struct ConvertCmpF : public OpConversionPattern<cuda_tile::CmpFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pred = mapCmpFPredicate(op.getComparisonPredicate(),
                                 op.getComparisonOrdering());
    if (failed(pred))
      return rewriter.notifyMatchFailure(op,
                                         "unsupported cmpf predicate/ordering");
    rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, *pred, adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};

struct ConvertExp2 : public OpConversionPattern<cuda_tile::Exp2Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::Exp2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getFlushToZero())
      return rewriter.notifyMatchFailure(
          op, "exp2 flush_to_zero is not representable in math.exp2");
    rewriter.replaceOpWithNewOp<math::Exp2Op>(op, adaptor.getSource());
    return success();
  }
};

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

struct ConvertRsqrt : public OpConversionPattern<cuda_tile::RsqrtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::RsqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getFlushToZero())
      return rewriter.notifyMatchFailure(
          op, "rsqrt flush_to_zero is not representable in math.rsqrt");
    rewriter.replaceOpWithNewOp<math::RsqrtOp>(op, adaptor.getSource());
    return success();
  }
};

struct ConvertTanH : public OpConversionPattern<cuda_tile::TanHOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::TanHOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getRoundingMode() != cuda_tile::RoundingMode::FULL)
      return rewriter.notifyMatchFailure(
          op, "tanh rounding<approx> is not representable in math.tanh");
    rewriter.replaceOpWithNewOp<math::TanhOp>(op, adaptor.getSource());
    return success();
  }
};

struct ConvertCmpI : public OpConversionPattern<cuda_tile::CmpIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pred =
        mapCmpIPredicate(op.getComparisonPredicate(), op.getSignedness());
    if (failed(pred))
      return rewriter.notifyMatchFailure(
          op, "unsupported cmpi predicate/signedness");
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, *pred, adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};

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

struct ConvertNegI : public OpConversionPattern<cuda_tile::NegIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::NegIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type ty = adaptor.getSource().getType();
    auto zeroAttr = rewriter.getZeroAttr(ty);
    if (!zeroAttr)
      return rewriter.notifyMatchFailure(
          op, "cannot create zero value for negi source type");
    Value zero = arith::ConstantOp::create(rewriter, op.getLoc(), ty, zeroAttr);
    rewriter.replaceOpWithNewOp<arith::SubIOp>(op, zero, adaptor.getSource());
    return success();
  }
};

/// Convert cuda_tile.for to scf.for.
///   - Create scf.ForOp with converted bounds and initial values.
///     Use bounds in index space, not tile space; 
///   - Convert region types in old body using type converter.
///   - Merge old body into new body, replacing block args (induction var +
///     iter args).
///   - Replace old op with new ForOp results.
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

    // Check if the upper bound originates from get_index_space_shape.
    // If so, rescale the loop to iterate in element-index space and introduce
    // a divui at the top of the body to recover the tile-space index for
    // existing consumers.
    int64_t tileSize = 0;
    if (auto issOp = op.getUpperBound()
                         .getDefiningOp<cuda_tile::GetIndexSpaceShapeOp>()) {
      unsigned resultIdx = 0;
      for (auto result : issOp->getResults()) {
        if (result == op.getUpperBound())
          break;
        ++resultIdx;
      }
      auto pvType =
          cast<cuda_tile::PartitionViewType>(issOp.getSrc().getType());
      tileSize = pvType.getTileShape().asArrayRef()[resultIdx];
    }

    if (tileSize > 0) {
      Value tileSizeVal =
          arith::ConstantIndexOp::create(rewriter, loc, tileSize);
      lb = arith::MulIOp::create(rewriter, loc, lb, tileSizeVal);
      ub = arith::MulIOp::create(rewriter, loc, ub, tileSizeVal);
      step = arith::MulIOp::create(rewriter, loc, step, tileSizeVal);
    }

    auto newForOp = scf::ForOp::create(rewriter, loc, lb, ub, step,
                                       adaptor.getInitValues());

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
      // The loop now iterates in element space. Introduce a divui to recover
      // the tile-space index that existing body ops expect.
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
/// Source-op semantics (from cuda_tile.mmai verifier):
///   Unbatched: lhs[M,K] x rhs[K,N] + acc[M,N] -> result[M,N]
///   Batched:   lhs[B,M,K] x rhs[B,K,N] + acc[B,M,N] -> result[B,M,N]
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

    // Explicit combining kind = add (mmai is integer multiply-accumulate).
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        op, adaptor.getLhs(), adaptor.getRhs(), adaptor.getAcc(),
        rewriter.getAffineMapArrayAttr({spec->mapA, spec->mapB, spec->mapC}),
        rewriter.getArrayAttr(spec->iterTypes), vector::CombiningKind::ADD);
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

/// Convert cuda_tile.reshape to vector.shape_cast / vector.broadcast /
/// vector.extract depending on source/result ranks.
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
      rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, dstVecTy, source);
    } else if (srcVecTy && !dstVecTy) {
      SmallVector<int64_t> indices(srcVecTy.getRank(), 0);
      rewriter.replaceOpWithNewOp<vector::ExtractOp>(op, source, indices);
    } else {
      rewriter.replaceOp(op, source);
    }
    return success();
  }
};

/// Convert cuda_tile.get_index_space_shape.
/// For
/// partition_view<tile=(T0xT1x...), tensor_view<?x?x...>, dim_map=[d0,d1,...]>
/// index_space_shape[i] = ceildiv(tensor_shape[dimMap[i]], tileShape[i]).
struct ConvertGetIndexSpaceShape
    : public OpConversionPattern<cuda_tile::GetIndexSpaceShapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::GetIndexSpaceShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pvInfo = getPartitionViewInfo(op.getSrc(), adaptor.getSrc());

    Location loc = op.getLoc();
    unsigned rank = pvInfo.tileShape.size();

    // For each tile dimension i:
    // - The corresponding tensor_view dimension is dimMap[i]
    // - index_space_dim_i = ceildiv(memref.dim(dimMap[i]), tileShape[i])
    // When the memref dimension is statically known, fold to a constant.
    auto memrefTy = cast<MemRefType>(pvInfo.memref.getType());
    auto memrefShape = memrefTy.getShape();

    SmallVector<Value> results;
    for (unsigned i = 0; i < rank; ++i) {
      int64_t tileSize = pvInfo.tileShape[i];
      unsigned tensorDim = pvInfo.dimMap[i];
      Type resultTy =
          getTypeConverter()->convertType(op->getResult(i).getType());
      if (!resultTy)
        return rewriter.notifyMatchFailure(
            op, "cannot convert get_index_space_shape result type");

      int64_t dimSize = memrefShape[tensorDim];
      Value castedResult;
      if (dimSize != ShapedType::kDynamic) {
        // Static dimension: compute ceildiv at compile time.
        int64_t numTiles = (dimSize + tileSize - 1) / tileSize;
        Value cst = arith::ConstantIndexOp::create(rewriter, loc, numTiles);
        castedResult = castValueToType(rewriter, loc, cst, resultTy);
      } else {
        // Dynamic dimension: emit memref.dim + ceildivui.
        Value dimVal = memref::DimOp::create(
            rewriter, loc, pvInfo.memref,
            arith::ConstantIndexOp::create(rewriter, loc, tensorDim));
        Value tileSizeVal =
            arith::ConstantIndexOp::create(rewriter, loc, tileSize);
        Value divResult =
            arith::CeilDivUIOp::create(rewriter, loc, dimVal, tileSizeVal);
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

/// Keeps the information needed by vector.transfer_read / transfer_write to
/// access a memref through a partition_view.
struct TransferViewAccessPlan {
  PartitionViewInfo pvInfo;
  SmallVector<Value> memrefIndices;
  AffineMap permutationMap;
  SmallVector<bool> inBounds;
};

/// Build a TransferViewAccessPlan for a load_view_tko or store_view_tko.
///
/// Translate partition-view tile indices into the concrete memref indices,
/// permutation map, and in-bounds flags required by vector.transfer_read/write.
/// 1. Cast each tile-level index to `index` and scale by the tile extent.
/// 2. Place the scaled index into the memref-dimension slot given by dim_map.
/// 3. Build a permutation_map that maps memref dims back to tile dims.
/// 4. Set inBounds[i] = true only when the tensor extent is static and evenly
///    divisible by the tile extent along that dimension.
static FailureOr<TransferViewAccessPlan>
buildTransferViewAccessPlan(ConversionPatternRewriter &rewriter, Operation *op,
                            Value view, Value convertedView, VectorType vecTy,
                            ValueRange convertedIndices) {
  auto pvInfo = getPartitionViewInfo(view, convertedView);

  unsigned tileRank = pvInfo.tileShape.size();
  unsigned tensorRank = pvInfo.tensorViewRank;

  if ((unsigned)vecTy.getRank() != tileRank)
    return rewriter.notifyMatchFailure(
        op, "converted tile rank does not match partition tile_shape rank");

  Location loc = op->getLoc();
  auto *ctx = rewriter.getContext();

  // Build memref indices in tensor-dimension order.
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> memrefIndices(tensorRank, zero);
  for (unsigned i = 0; i < tileRank; ++i) {
    unsigned tensorDim = pvInfo.dimMap[i];
    int64_t tileSize = pvInfo.tileShape[i];
    Value tileIndex = castValueToType(rewriter, loc, convertedIndices[i],
                                      rewriter.getIndexType());
    if (!tileIndex)
      return rewriter.notifyMatchFailure(
          op, "view index could not be converted to index");
    Value tileSizeVal = arith::ConstantIndexOp::create(rewriter, loc, tileSize);
    Value elemOffset =
        arith::MulIOp::create(rewriter, loc, tileIndex, tileSizeVal);
    memrefIndices[tensorDim] = elemOffset;
  }

  SmallVector<AffineExpr> permExprs;
  permExprs.reserve(tileRank);
  for (int32_t td : pvInfo.dimMap)
    permExprs.push_back(getAffineDimExpr(td, ctx));
  auto permutationMap = AffineMap::get(tensorRank, 0, permExprs, ctx);

  auto memrefTy = cast<MemRefType>(pvInfo.memref.getType());
  auto memrefShape = memrefTy.getShape();
  SmallVector<bool> inBounds(tileRank, false);
  for (unsigned i = 0; i < tileRank; ++i) {
    int64_t ms = memrefShape[pvInfo.dimMap[i]];
    int64_t ts = pvInfo.tileShape[i];
    inBounds[i] = (ms != ShapedType::kDynamic && ts > 0 && (ms % ts) == 0);
  }

  return TransferViewAccessPlan{std::move(pvInfo), std::move(memrefIndices),
                                permutationMap, std::move(inBounds)};
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
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::LoadViewTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (failed(checkCommonTkoGuards(op, rewriter)))
      return failure();

    auto vecTy = cast<VectorType>(
        getTypeConverter()->convertType(op.getTile().getType()));

    auto plan = buildTransferViewAccessPlan(rewriter, op, op.getView(),
                                            adaptor.getView(), vecTy,
                                            adaptor.getIndex());
    if (failed(plan))
      return failure();

    Value padding = makePaddingValue(rewriter, loc, vecTy.getElementType(),
                                     plan->pvInfo.paddingValue);
    auto readOp = vector::TransferReadOp::create(
        rewriter, loc, vecTy, plan->pvInfo.memref, plan->memrefIndices,
        AffineMapAttr::get(plan->permutationMap), padding,
        /*mask=*/Value(), rewriter.getBoolArrayAttr(plan->inBounds));

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
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::StoreViewTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (failed(checkCommonTkoGuards(op, rewriter)))
      return failure();

    auto vecTy = cast<VectorType>(
        getTypeConverter()->convertType(op.getTile().getType()));

    auto plan = buildTransferViewAccessPlan(rewriter, op, op.getView(),
                                            adaptor.getView(), vecTy,
                                            adaptor.getIndex());
    if (failed(plan))
      return failure();

    auto writeOp = vector::TransferWriteOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange{}, adaptor.getTile(),
        plan->pvInfo.memref, plan->memrefIndices,
        AffineMapAttr::get(plan->permutationMap),
        /*mask=*/Value(), rewriter.getBoolArrayAttr(plan->inBounds));
    (void)writeOp;

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// View construction: make_tensor_view / make_partition_view
//===----------------------------------------------------------------------===//

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

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, resultTy, adaptor.getBase(), /*offset=*/rewriter.getIndexAttr(0),
        sizes, strides);
    return success();
  }
};

/// Convert cuda_tile.make_partition_view: the partition_view type maps to the
/// same ranked memref as its underlying tensor_view, so this pattern just
/// forwards the already-converted memref.
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

//===----------------------------------------------------------------------===//
// cuda_tile.module / cuda_tile.entry -> gpu.module / gpu.func
//===----------------------------------------------------------------------===//

/// Convert cuda_tile.entry to gpu.func.
///
/// The gpu.func signature is derived by applying the type converter to each
/// entry argument type (e.g. `tile<ptr<T>>` -> `memref<*xT>`, `tile<i32>` ->
/// `i32`). The entry body is then signature-converted in place and merged into
/// the gpu.func body.
struct ConvertEntry : public OpConversionPattern<cuda_tile::EntryOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::EntryOp entryOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = entryOp.getContext();
    Location loc = entryOp.getLoc();
    Block *entryBlock = &entryOp.getBody().front();
    unsigned numArgs = entryBlock->getNumArguments();

    // Compute gpu.func arg types and prepare a signature conversion for the
    // entry block.
    const TypeConverter *tc = getTypeConverter();
    TypeConverter::SignatureConversion sigConv(numArgs);
    SmallVector<Type> gpuFuncArgTypes;
    gpuFuncArgTypes.reserve(numArgs);
    for (unsigned i = 0; i < numArgs; ++i) {
      Type origTy = entryBlock->getArgument(i).getType();
      Type converted = tc->convertType(origTy);
      if (!converted)
        return rewriter.notifyMatchFailure(entryOp,
                                           "cannot convert entry arg type");
      gpuFuncArgTypes.push_back(converted);
      sigConv.addInputs(i, converted);
    }

    auto gpuFunc =
        gpu::GPUFuncOp::create(rewriter, loc, entryOp.getSymName(),
                               FunctionType::get(ctx, gpuFuncArgTypes, {}));
    gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                     rewriter.getUnitAttr());

    // Convert the entry block's arg types; this replaces the block with a
    // new one having the converted signature and rewires uses via source
    // materializations.
    FailureOr<Block *> convertedBlock =
        rewriter.convertRegionTypes(&entryOp.getBody(), *tc, &sigConv);
    if (failed(convertedBlock))
      return failure();

    // Merge the (now type-converted) entry block into the gpu.func body.
    Block *gpuBlock = &gpuFunc.getBody().front();
    rewriter.mergeBlocks(*convertedBlock, gpuBlock, gpuBlock->getArguments());
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
  //   - ints        -> preserved scalar integer type
  //   - float       -> preserved scalar type
  //   - ptr<T>      -> memref<*xT> (unranked memref backing the pointer)
  converter.addConversion([ctx](cuda_tile::TileType tileTy) -> Type {
    auto shape = tileTy.getShape();
    auto elemTy = tileTy.getElementType();

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

static void populateTileIRToGPUConversionPatterns(TypeConverter &converter,
                                                  RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns
      .add<ConvertModule, ConvertEntry, ConvertMakeTensorView,
           ConvertMakePartitionView, ConvertConstant,
           ConvertDimQueryOp<cuda_tile::GetTileBlockIdOp, gpu::BlockIdOp>,
           ConvertDimQueryOp<cuda_tile::GetNumTileBlocksOp, gpu::GridDimOp>,
           ConvertBinaryLhsRhsOp<cuda_tile::MulIOp, arith::MulIOp>,
           ConvertAtan2, ConvertUnarySourceOp<cuda_tile::CeilOp, math::CeilOp>,
           ConvertCmpF, ConvertUnarySourceOp<cuda_tile::CosOp, math::CosOp>,
           ConvertExp2, ConvertUnarySourceOp<cuda_tile::ExpOp, math::ExpOp>,
           ConvertUnarySourceOp<cuda_tile::FloorOp, math::FloorOp>,
           ConvertUnarySourceOp<cuda_tile::Log2Op, math::Log2Op>,
           ConvertMinMaxFOp<cuda_tile::MaxFOp, /*IsMax=*/true>,
           ConvertMinMaxFOp<cuda_tile::MinFOp, /*IsMax=*/false>,
           ConvertUnarySourceOp<cuda_tile::NegFOp, arith::NegFOp>, ConvertPow,
           ConvertRsqrt, ConvertUnarySourceOp<cuda_tile::SinOp, math::SinOp>,
           ConvertTanH, ConvertCmpI,
           ConvertMinMaxIOp<cuda_tile::MaxIOp, /*IsMax=*/true>,
           ConvertMinMaxIOp<cuda_tile::MinIOp, /*IsMax=*/false>, ConvertMmaI,
           ConvertMulhiI, ConvertNegI,
           ConvertBinaryLhsRhsOp<cuda_tile::XOrIOp, arith::XOrIOp>, ConvertFor,
           ConvertIf, ConvertToScfYield<cuda_tile::ContinueOp>,
           ConvertToScfYield<cuda_tile::YieldOp>, ConvertReturn, ConvertMmaF,
           ConvertAssume, ConvertReshape, ConvertGetIndexSpaceShape,
           ConvertLoadViewTko, ConvertStoreViewTko>(converter, ctx);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct ConvertTileIRToGPUPass
    : public PassWrapper<ConvertTileIRToGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTileIRToGPUPass)

  StringRef getArgument() const override { return "convert-cuda-tile-to-gpu"; }

  StringRef getDescription() const override {
    return "Convert CudaTile IR to GPU/vector/scf/arith ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cuda_tile::CudaTileDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<gpu::GPUDialect>();
    registry.insert<ub::UBDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    TypeConverter typeConverter;
    populateTileIRToGPUTypeConverter(typeConverter, ctx);

    RewritePatternSet patterns(ctx);
    populateTileIRToGPUConversionPatterns(typeConverter, patterns);

    ConversionTarget target(*ctx);

    // GPU/vector/arith/scf/memref/ub ops are legal.
    target.addLegalDialect<arith::ArithDialect, gpu::GPUDialect,
                           math::MathDialect, memref::MemRefDialect,
                           scf::SCFDialect, ub::UBDialect,
                           vector::VectorDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    // CudaTile ops are illegal (target of conversion).
    target.addIllegalDialect<cuda_tile::CudaTileDialect>();

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return signalPassFailure();

    // Post-conversion cleanup: fold muli(divui(x, c), c) → x.
    // The divui result may pass through index_cast ops before reaching the
    // muli, so we look through index_casts to find the underlying divui.
    // The divisor and multiplier may be separate constant ops with the same
    // value, so we compare constant attribute values.
    module.walk([](arith::MulIOp op) {
      for (auto [mulOperand, otherOperand] :
           {std::pair(op.getLhs(), op.getRhs()),
            std::pair(op.getRhs(), op.getLhs())}) {
        // Look through index_casts.
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
        // Check if they are distinct constants with the same value.
        auto divCst = divOp.getRhs().getDefiningOp<arith::ConstantIndexOp>();
        auto mulCst = otherOperand.getDefiningOp<arith::ConstantIndexOp>();
        if (divCst && mulCst && divCst.value() == mulCst.value()) {
          op.replaceAllUsesWith(divOp.getLhs());
          op->erase();
          return;
        }
      }
    });

    // Clean up any leftover unrealized_conversion_casts that became dead.
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
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertTileIRToGPUPass() {
  return std::make_unique<ConvertTileIRToGPUPass>();
}
