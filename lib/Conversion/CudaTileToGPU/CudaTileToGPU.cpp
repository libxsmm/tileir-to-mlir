//===- TileIRToGPU.cpp - CudaTile IR to GPU conversion ----------*- C++ -*-===//
//
// Conversion pass from CudaTile IR to GPU/vector/scf/arith/memref ops.
//
//===----------------------------------------------------------------------===//

// cuda_tile ops with registered conversion patterns in this pass:
// cuda_tile::AbsFOp
// cuda_tile::AbsIOp
// cuda_tile::AddFOp
// cuda_tile::AddIOp
// cuda_tile::AndIOp
// cuda_tile::AssumeOp
// cuda_tile::Atan2Op
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
// cuda_tile::LoadViewTkoOp
// cuda_tile::LogOp
// cuda_tile::Log2Op
// cuda_tile::MakePartitionViewOp
// cuda_tile::MakeTensorViewOp
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
// cuda_tile::StoreViewTkoOp
// cuda_tile::SubFOp
// cuda_tile::SubIOp
// cuda_tile::TanOp
// cuda_tile::TanHOp
// cuda_tile::TruncIOp
// cuda_tile::XOrIOp
// cuda_tile::YieldOp
//
// cuda_tile ops without a registered conversion pattern in this pass:
// cuda_tile::AssertOp
// cuda_tile::AtomicCASTkoOp
// cuda_tile::AtomicRMWTkoOp
// cuda_tile::BreakOp
// cuda_tile::IntToPtrOp
// cuda_tile::JoinTokensOp
// cuda_tile::LoadPtrTkoOp
// cuda_tile::LoopOp
// cuda_tile::MakeTokenOp
// cuda_tile::OffsetOp
// cuda_tile::PrintTkoOp
// cuda_tile::PtrToIntOp
// cuda_tile::StorePtrTkoOp

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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/TypeSwitch.h"

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

/// Validate the semantic invariants that transfer and index-space queries rely
/// on for a partition_view.
static LogicalResult
validatePartitionViewInfo(Operation *op, const PartitionViewInfo &info,
                          ConversionPatternRewriter &rewriter) {
  auto memrefTy = dyn_cast<MemRefType>(info.memref.getType());
  if (!memrefTy)
    return rewriter.notifyMatchFailure(
        op, "partition_view source did not convert to a ranked memref");

  if (memrefTy.getRank() != static_cast<int64_t>(info.tensorViewRank))
    return rewriter.notifyMatchFailure(
        op,
        "converted partition_view memref rank does not match tensor_view rank");

  if (info.dimMap.size() != info.tileShape.size())
    return rewriter.notifyMatchFailure(
        op, "partition_view dim_map rank does not match tile_shape rank");

  llvm::SmallBitVector seenDims(info.tensorViewRank);
  for (auto [tileDim, tensorDim] : llvm::enumerate(info.dimMap)) {
    if (tensorDim < 0 ||
        static_cast<unsigned>(tensorDim) >= info.tensorViewRank)
      return rewriter.notifyMatchFailure(
          op,
          "partition_view dim_map references an out-of-range tensor dimension");
    if (seenDims.test(tensorDim))
      return rewriter.notifyMatchFailure(
          op,
          "partition_view dim_map must be a permutation without duplicates");
    seenDims.set(tensorDim);
    if (info.tileShape[tileDim] <= 0)
      return rewriter.notifyMatchFailure(
          op, "partition_view tile dimensions must be strictly positive");
  }

  return success();
}

/// Map cuda_tile cmpf predicate+ordering to the corresponding arith predicate.
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

/// Map cuda_tile cmpi predicate+signedness to the corresponding arith
/// predicate.
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

/// Build a tensor-typed ElementsAttr for memref.global from a cuda_tile.global
/// initializer.
///
/// memref.global requires an ElementsAttr whose type is a ranked tensor.
/// cuda_tile.global may carry an elements attribute with a non-tensor shaped
/// type (for example, a vector type). Normalize to the equivalent ranked
/// tensor payload while preserving values.
static FailureOr<ElementsAttr>
getGlobalInitializerAttrOrFail(cuda_tile::GlobalOp globalOp,
                               ConversionPatternRewriter &rewriter,
                               Operation *diagnosticOp) {
  auto initAttr = dyn_cast<ElementsAttr>(globalOp.getValue());
  if (!initAttr)
    return rewriter.notifyMatchFailure(
        diagnosticOp, "global initializer must be an elements attribute");

  auto initTy = dyn_cast<ShapedType>(initAttr.getType());
  if (!initTy || !initTy.hasStaticShape())
    return rewriter.notifyMatchFailure(
        diagnosticOp,
        "global initializer must be a statically shaped elements attribute");

  auto tensorTy =
      RankedTensorType::get(initTy.getShape(), initTy.getElementType());
  if (initAttr.getType() == tensorTy)
    return initAttr;

  auto denseAttr = dyn_cast<DenseElementsAttr>(initAttr);
  if (!denseAttr)
    return rewriter.notifyMatchFailure(
        diagnosticOp, "global initializer must be a dense elements attribute "
                      "when retyping is required");

  if (denseAttr.isSplat())
    return ElementsAttr(
        DenseElementsAttr::get(tensorTy, denseAttr.getSplatValue<Attribute>()));

  SmallVector<Attribute> values(denseAttr.getValues<Attribute>().begin(),
                                denseAttr.getValues<Attribute>().end());
  return ElementsAttr(DenseElementsAttr::get(tensorTy, values));
}

/// Resolve a get_global symbol to the ranked memref type of the referenced
/// allocation.
///
/// During dialect conversion, the referenced symbol may still be
/// `cuda_tile.global` or may already have been rewritten to `memref.global`
/// depending on pattern application order. Accept both cases to keep
/// get_global lowering order-independent.
static FailureOr<MemRefType>
getReferencedGlobalMemRefTypeOrFail(cuda_tile::GetGlobalOp getGlobalOp,
                                    ConversionPatternRewriter &rewriter) {
  Operation *symbolOp = SymbolTable::lookupNearestSymbolFrom(
      getGlobalOp, getGlobalOp.getNameAttr());
  if (!symbolOp)
    return rewriter.notifyMatchFailure(getGlobalOp,
                                       "referenced global symbol not found");

  if (auto cudaGlobal = dyn_cast<cuda_tile::GlobalOp>(symbolOp))
    return getGlobalMemRefTypeOrFail(cudaGlobal, rewriter, getGlobalOp);

  if (auto memrefGlobal = dyn_cast<memref::GlobalOp>(symbolOp)) {
    auto memrefTy = dyn_cast<MemRefType>(memrefGlobal.getType());
    if (!memrefTy)
      return rewriter.notifyMatchFailure(
          getGlobalOp,
          "referenced memref.global does not have a ranked memref type");
    return memrefTy;
  }

  return rewriter.notifyMatchFailure(
      getGlobalOp,
      "referenced symbol is neither cuda_tile.global nor memref.global");
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

/// Convert float binary ops that reject flush_to_zero (addf, subf, mulf, divf).
template <typename SrcOp, typename DstOp>
struct ConvertBinaryFloatOp : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getFlushToZero())
      return rewriter.notifyMatchFailure(
          op, "flush_to_zero is not representable in arith float ops");
    rewriter.template replaceOpWithNewOp<DstOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs());
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

/// Convert signedness-directed casts where the destination arith op depends
/// only on the source op's signedness attribute.
///
/// Used by:
///   - cuda_tile.exti  -> arith.extsi / arith.extui
///
///   1. Convert the destination tile type (`to`) via the type converter.
///   2. Dispatch to `UnsignedDstOp` for `signedness = unsigned`, otherwise to
///      `SignedDstOp`.
///   3. Replace the original op with the selected arith op.
template <typename SrcOp, typename SignedDstOp, typename UnsignedDstOp>
struct ConvertFromToSignednessCastOp : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy =
        getConvertedResultTypeOrFail(op, this->getTypeConverter(), rewriter,
                                     "cannot convert cast result type");
    if (failed(resultTy))
      return failure();

    if (op.getSignedness() == cuda_tile::Signedness::Unsigned)
      rewriter.template replaceOpWithNewOp<UnsignedDstOp>(op, resultTy.value(),
                                                          adaptor.getFrom());
    else
      rewriter.template replaceOpWithNewOp<SignedDstOp>(op, resultTy.value(),
                                                        adaptor.getFrom());
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
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename OpConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getRoundingMode() != ExpectedRounding)
      return rewriter.notifyMatchFailure(op,
                                         "unsupported rounding mode for cast");

    auto resultTy =
        getConvertedResultTypeOrFail(op, this->getTypeConverter(), rewriter,
                                     "cannot convert cast result type");
    if (failed(resultTy))
      return failure();

    if (op.getSignedness() == cuda_tile::Signedness::Unsigned)
      rewriter.template replaceOpWithNewOp<UnsignedDstOp>(op, resultTy.value(),
                                                          adaptor.getFrom());
    else
      rewriter.template replaceOpWithNewOp<SignedDstOp>(op, resultTy.value(),
                                                        adaptor.getFrom());
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

/// Convert cuda_tile.maxf/minf based on propagate_nan and flush_to_zero flags.
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

/// Match the body of a cuda_tile.reduce to determine the CombiningKind.
/// The body is expected to have exactly one combining op (ignoring the yield).
static FailureOr<vector::CombiningKind> matchReduceBody(Region &body) {
  Block &block = body.front();
  if (block.getNumArguments() != 2)
    return failure();

  auto yieldOp = dyn_cast<cuda_tile::YieldOp>(block.getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return failure();

  // The body should contain exactly one op besides the terminator (yield).
  Operation *combiningOp = nullptr;
  for (Operation &op : block.without_terminator()) {
    if (combiningOp)
      return failure(); // more than one op
    combiningOp = &op;
  }
  if (!combiningOp)
    return failure();

  if (combiningOp->getNumOperands() != 2 || combiningOp->getNumResults() != 1)
    return failure();
  if (yieldOp.getOperand(0) != combiningOp->getResult(0))
    return failure();

  auto lhsArg = dyn_cast<BlockArgument>(combiningOp->getOperand(0));
  auto rhsArg = dyn_cast<BlockArgument>(combiningOp->getOperand(1));
  if (!lhsArg || !rhsArg || lhsArg.getOwner() != &block ||
      rhsArg.getOwner() != &block)
    return failure();
  if (lhsArg.getArgNumber() == rhsArg.getArgNumber())
    return failure();

  return llvm::TypeSwitch<Operation *, FailureOr<vector::CombiningKind>>(
             combiningOp)
      .Case<cuda_tile::AddFOp, cuda_tile::AddIOp>(
          [](auto) { return vector::CombiningKind::ADD; })
      .Case<cuda_tile::MulFOp, cuda_tile::MulIOp>(
          [](auto) { return vector::CombiningKind::MUL; })
      .Case<cuda_tile::MaxFOp>([](cuda_tile::MaxFOp op) {
        return op.getPropagateNan() ? vector::CombiningKind::MAXIMUMF
                                    : vector::CombiningKind::MAXNUMF;
      })
      .Case<cuda_tile::MinFOp>([](cuda_tile::MinFOp op) {
        return op.getPropagateNan() ? vector::CombiningKind::MINIMUMF
                                    : vector::CombiningKind::MINNUMF;
      })
      .Case<cuda_tile::MaxIOp>([](cuda_tile::MaxIOp op) {
        return op.getSignedness() == cuda_tile::Signedness::Unsigned
                   ? vector::CombiningKind::MAXUI
                   : vector::CombiningKind::MAXSI;
      })
      .Case<cuda_tile::MinIOp>([](cuda_tile::MinIOp op) {
        return op.getSignedness() == cuda_tile::Signedness::Unsigned
                   ? vector::CombiningKind::MINUI
                   : vector::CombiningKind::MINSI;
      })
      .Case<cuda_tile::AndIOp>([](auto) { return vector::CombiningKind::AND; })
      .Case<cuda_tile::OrIOp>([](auto) { return vector::CombiningKind::OR; })
      .Case<cuda_tile::XOrIOp>([](auto) { return vector::CombiningKind::XOR; })
      .Default([](Operation *) { return failure(); });
}

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
                              ConversionPatternRewriter &rewriter,
                              StringRef opName) {
  if (op.getOperands().size() != 1)
    return rewriter.notifyMatchFailure(op, Twine("multi-operand ") + opName +
                                               " not supported");

  auto kind = matchReduceBody(op.getBody());
  if (failed(kind))
    return rewriter.notifyMatchFailure(
        op, Twine("cannot determine combining kind from ") + opName + " body");

  Value source = convertedOperands.front();
  auto srcVecTy = dyn_cast<VectorType>(source.getType());
  if (!srcVecTy)
    return rewriter.notifyMatchFailure(op, "source is not a vector");

  if (srcVecTy.getRank() == 0)
    return rewriter.notifyMatchFailure(op,
                                       "source vector must have positive rank");

  if (op.getDim() >= static_cast<uint32_t>(srcVecTy.getRank()))
    return rewriter.notifyMatchFailure(
        op, Twine(opName) + " reduction dimension is out of bounds");

  if (op.getIdentities().size() != 1)
    return rewriter.notifyMatchFailure(
        op, Twine(opName) + " requires exactly one identity value");

  auto identityAttr = dyn_cast<TypedAttr>(op.getIdentities()[0]);
  if (!identityAttr)
    return rewriter.notifyMatchFailure(op, "identity is not a typed attribute");

  if (identityAttr.getType() != srcVecTy.getElementType())
    return rewriter.notifyMatchFailure(
        op, Twine(opName) +
                " identity type does not match the source element type");

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
  if (failed(validatePartitionViewInfo(op, pvInfo, rewriter)))
    return failure();

  unsigned tileRank = pvInfo.tileShape.size();
  unsigned tensorRank = pvInfo.tensorViewRank;

  if ((unsigned)vecTy.getRank() != tileRank)
    return rewriter.notifyMatchFailure(
        op, "converted tile rank does not match partition tile_shape rank");
  if (convertedIndices.size() != tileRank)
    return rewriter.notifyMatchFailure(
        op, "partition_view index rank does not match tile_shape rank");

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
    auto nswFlag = arith::IntegerOverflowFlagsAttr::get(
        rewriter.getContext(), arith::IntegerOverflowFlags::nsw);
    Value elemOffset =
        arith::MulIOp::create(rewriter, loc, tileIndex, tileSizeVal, nswFlag);
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

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//
using ConvertAbsF = ConvertUnarySourceOp<cuda_tile::AbsFOp, math::AbsFOp>;

using ConvertAbsI = ConvertUnarySourceOp<cuda_tile::AbsIOp, math::AbsIOp>;

using ConvertAddF = ConvertBinaryFloatOp<cuda_tile::AddFOp, arith::AddFOp>;

using ConvertAddI =
    ConvertBinaryLhsRhsWithOverflowOp<cuda_tile::AddIOp, arith::AddIOp>;

using ConvertAndI = ConvertBinaryLhsRhsOp<cuda_tile::AndIOp, arith::AndIOp>;

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
    auto dstVecTy = cast<VectorType>(resultTy);
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, dstVecTy,
                                                     adaptor.getSource());
    return success();
  }
};

/// Convert cuda_tile.cat to vector.insert_strided_slice.
///
/// Source semantics (from Ops.td):
///   `cat %lhs, %rhs dim = d : tile<...>, tile<...> -> tile<...>`
///   Concatenates lhs and rhs along dimension `d`.  All non-concat dimensions
///   must match.  result.shape[d] = lhs.shape[d] + rhs.shape[d].
///
///   cat(x, y, d)[..., i_d, ...] =
///     x[..., i_d, ...]              if i_d < lhs.shape[d]
///     y[..., i_d - lhs.shape[d], ...] otherwise
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

/// Convert cuda_tile.cmpi to arith.cmpi.
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

/// Convert cuda_tile.constant to arith/vector constants.
///   - Scalar tiles (rank 0): Convert to scalar arith ops
///   - Ranked tiles: Convert to an arith.constant with a DenseElementsAttr of
///     the target vector type (splat values use the splat form, e.g.
///     `arith.constant dense<7> : vector<1x1xi32>`).
///   - Scalar integer conversion preserves the integer type; explicit casts to
///     `index` are inserted later at the ops that require them.
///   - Pointer types in scalar tiles are intermediate and will become dead
///   after
///     make_tensor_view ops are erased; conversion preserves them as-is.
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

using ConvertDivF = ConvertBinaryFloatOp<cuda_tile::DivFOp, arith::DivFOp>;

/// Convert cuda_tile.divi to arith.divsi/divui; rejects non-ZERO rounding.
struct ConvertDivI : public OpConversionPattern<cuda_tile::DivIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::DivIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getRounding() != cuda_tile::RoundingMode::ZERO)
      return rewriter.notifyMatchFailure(
          op, "only rounding<zero> (truncating) division is supported");
    if (op.getSignedness() == cuda_tile::Signedness::Unsigned)
      rewriter.replaceOpWithNewOp<arith::DivUIOp>(op, adaptor.getLhs(),
                                                  adaptor.getRhs());
    else
      rewriter.replaceOpWithNewOp<arith::DivSIOp>(op, adaptor.getLhs(),
                                                  adaptor.getRhs());
    return success();
  }
};

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

using ConvertExp = ConvertUnarySourceOp<cuda_tile::ExpOp, math::ExpOp>;

/// Convert cuda_tile.exp2 to math.exp2 when flush_to_zero is not requested.
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

using ConvertExtI =
    ConvertFromToSignednessCastOp<cuda_tile::ExtIOp, arith::ExtSIOp,
                                  arith::ExtUIOp>;

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

/// Convert cuda_tile.fma to math.fma; rejects flush_to_zero.
struct ConvertFma : public OpConversionPattern<cuda_tile::FmaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::FmaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getFlushToZero())
      return rewriter.notifyMatchFailure(
          op, "fma flush_to_zero is not representable in math.fma");
    rewriter.replaceOpWithNewOp<math::FmaOp>(
        op, adaptor.getLhs(), adaptor.getRhs(), adaptor.getAcc());
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

/// Convert cuda_tile.ftof to arith.extf / arith.truncf.
///
///   - Only rounding<nearest_even> is representable.
///   - Source and destination element widths must differ.
///   - Narrowing uses arith.truncf; widening uses arith.extf.
///
/// Works for both scalar float and vector<float> types.
struct ConvertFToF : public OpConversionPattern<cuda_tile::FToFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::FToFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getRoundingMode() != cuda_tile::RoundingMode::NEAREST_EVEN)
      return rewriter.notifyMatchFailure(
          op, "ftof only supports rounding<nearest_even>");

    auto resultTy = getConvertedResultTypeOrFail(
        op, getTypeConverter(), rewriter, "cannot convert ftof result type");
    if (failed(resultTy))
      return failure();

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
      rewriter.replaceOpWithNewOp<arith::ExtFOp>(op, resultTy.value(),
                                                 adaptor.getFrom());
      return success();
    }
    if (srcWidth > dstWidth) {
      rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, resultTy.value(),
                                                   adaptor.getFrom());
      return success();
    }

    return rewriter.notifyMatchFailure(op,
                                       "ftof source/result widths must differ");
  }
};

using ConvertFToI = ConvertFromToSignednessCastWithRoundingOp<
    cuda_tile::FToIOp, arith::FPToSIOp, arith::FPToUIOp,
    cuda_tile::RoundingMode::NEAREST_INT_TO_ZERO>;

/// Convert cuda_tile.get_global to memref.get_global (+ memref.cast).
///   1. Resolve the referenced cuda_tile.global symbol.
///   2. Emit memref.get_global with the ranked memref type derived from that
///      global initializer.
///   3. Cast to the converted result type (typically memref<*xT>) so this
///      pass's pointer model remains uniform (tile<ptr<T>> -> memref<*xT>).
struct ConvertGetGlobal : public OpConversionPattern<cuda_tile::GetGlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::GetGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rankedMemRefTy = getReferencedGlobalMemRefTypeOrFail(op, rewriter);
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
    if (failed(validatePartitionViewInfo(op, pvInfo, rewriter)))
      return failure();

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

using ConvertGetNumTileBlocks =
    ConvertDimQueryOp<cuda_tile::GetNumTileBlocksOp, gpu::GridDimOp>;

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
    ConvertDimQueryOp<cuda_tile::GetTileBlockIdOp, gpu::BlockIdOp>;

/// Convert cuda_tile.global to memref.global.
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

    auto initAttr = getGlobalInitializerAttrOrFail(op, rewriter, op);
    if (failed(initAttr))
      return failure();

    IntegerAttr alignmentAttr;
    if (op.getAlignment() != 0)
      alignmentAttr = rewriter.getI64IntegerAttr(op.getAlignment());

    rewriter.replaceOpWithNewOp<memref::GlobalOp>(
        op, op.getSymName(), /*sym_visibility=*/StringAttr(), *memrefTy,
        *initAttr, /*constant=*/false, alignmentAttr);
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

    Value padding = makePaddingValue(rewriter, loc, vecTy->getElementType(),
                                     plan->pvInfo.paddingValue);
    auto readOp = vector::TransferReadOp::create(
        rewriter, loc, *vecTy, plan->pvInfo.memref, plan->memrefIndices,
        AffineMapAttr::get(plan->permutationMap), padding,
        /*mask=*/Value(), rewriter.getBoolArrayAttr(plan->inBounds));

    // result_token has no live uses (guarded above); drop it.
    rewriter.replaceOp(op, {readOp.getResult(), Value()});
    return success();
  }
};

using ConvertLog = ConvertUnarySourceOp<cuda_tile::LogOp, math::LogOp>;

using ConvertLog2 = ConvertUnarySourceOp<cuda_tile::Log2Op, math::Log2Op>;

//===----------------------------------------------------------------------===//
// View construction: make_tensor_view / make_partition_view
//===----------------------------------------------------------------------===//

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

using ConvertMaxF = ConvertMinMaxFOp<cuda_tile::MaxFOp, /*IsMax=*/true>;

using ConvertMaxI = ConvertMinMaxIOp<cuda_tile::MaxIOp, /*IsMax=*/true>;

using ConvertMinF = ConvertMinMaxFOp<cuda_tile::MinFOp, /*IsMax=*/false>;

using ConvertMinI = ConvertMinMaxIOp<cuda_tile::MinIOp, /*IsMax=*/false>;

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

using ConvertNegF = ConvertUnarySourceOp<cuda_tile::NegFOp, arith::NegFOp>;

/// Convert cuda_tile.negi to arith.subi(0, source).
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

using ConvertOrI = ConvertBinaryLhsRhsOp<cuda_tile::OrIOp, arith::OrIOp>;

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
/// Pointer model in this pass:
///   tile<ptr<T>> -> memref<*xT>
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
    auto pre = matchSingleOperandCombiningOp(op, adaptor.getOperands(),
                                             rewriter, "reduce");
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

/// Convert cuda_tile.rsqrt to math.rsqrt when flush_to_zero is not requested.
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

    auto pre = matchSingleOperandCombiningOp(op, adaptor.getOperands(),
                                             rewriter, "scan");
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

/// Convert cuda_tile.sqrt to math.sqrt; rejects flush_to_zero.
struct ConvertSqrt : public OpConversionPattern<cuda_tile::SqrtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::SqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getFlushToZero())
      return rewriter.notifyMatchFailure(
          op, "sqrt flush_to_zero is not representable in math.sqrt");
    rewriter.replaceOpWithNewOp<math::SqrtOp>(op, adaptor.getSource());
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
        plan->pvInfo.memref, plan->memrefIndices,
        AffineMapAttr::get(plan->permutationMap),
        /*mask=*/Value(), rewriter.getBoolArrayAttr(plan->inBounds));
    (void)writeOp;

    rewriter.eraseOp(op);
    return success();
  }
};

using ConvertSubF = ConvertBinaryFloatOp<cuda_tile::SubFOp, arith::SubFOp>;

using ConvertSubI =
    ConvertBinaryLhsRhsWithOverflowOp<cuda_tile::SubIOp, arith::SubIOp>;

using ConvertTan = ConvertUnarySourceOp<cuda_tile::TanOp, math::TanOp>;

/// Convert cuda_tile.tanh to math.tanh when rounding mode is FULL.
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

using ConvertXOrI = ConvertBinaryLhsRhsOp<cuda_tile::XOrIOp, arith::XOrIOp>;

using ConvertYield = ConvertToScfYield<cuda_tile::YieldOp>;

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Type converter and conversion pattern population
//===----------------------------------------------------------------------===//

/// Populate type-conversion rules for cuda_tile -> gpu/vector lowering.
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

/// Register all cuda_tile -> gpu/vector conversion patterns.
static void populateTileIRToGPUConversionPatterns(TypeConverter &converter,
                                                  RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<
      ConvertAbsF, ConvertAbsI, ConvertAddF, ConvertAddI, ConvertAndI,
      ConvertAssume, ConvertAtan2, ConvertBitcast, ConvertBroadcast, ConvertCat,
      ConvertCeil, ConvertCmpF, ConvertCmpI, ConvertConstant, ConvertContinue,
      ConvertCos, ConvertCosH, ConvertDivF, ConvertDivI, ConvertEntry,
      ConvertExp, ConvertExp2, ConvertExtI, ConvertExtract, ConvertFloor,
      ConvertFma, ConvertFor, ConvertFToF, ConvertFToI, ConvertGetGlobal,
      ConvertGetIndexSpaceShape, ConvertGetNumTileBlocks, ConvertGetTensorShape,
      ConvertGetTileBlockId, ConvertGlobal, ConvertIf, ConvertIota, ConvertIToF,
      ConvertLoadViewTko, ConvertLog, ConvertLog2, ConvertMakePartitionView,
      ConvertMakeTensorView, ConvertMaxF, ConvertMaxI, ConvertMinF, ConvertMinI,
      ConvertMmaF, ConvertMmaI, ConvertModule, ConvertMulF, ConvertMulhiI,
      ConvertMulI, ConvertNegF, ConvertNegI, ConvertOrI, ConvertPermute,
      ConvertPow, ConvertPtrToPtrCastOrFail, ConvertReduce, ConvertRemF,
      ConvertRemI, ConvertReshape, ConvertReturn, ConvertRsqrt, ConvertScan,
      ConvertSelect, ConvertShLI, ConvertShRI, ConvertSin, ConvertSinH,
      ConvertSqrt, ConvertStoreViewTko, ConvertSubF, ConvertSubI, ConvertTan,
      ConvertTanH, ConvertTruncI, ConvertXOrI, ConvertYield>(converter, ctx);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

/// Pass driver for lowering CudaTile IR to GPU/vector/scf/arith/memref.
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

    // Post-conversion cleanup: fold muli(divui(x, c), c) -> x.
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

    // Mark the module as a GPU container module.
    module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                    UnitAttr::get(ctx));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertTileIRToGPUPass() {
  return std::make_unique<ConvertTileIRToGPUPass>();
}
