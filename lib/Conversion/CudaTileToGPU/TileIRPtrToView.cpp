//===- TileIRPtrToView.cpp - Triton ptr-arith -> CudaTile view ops --------===//
//
// Pre-conversion pass that recognises the canonical Triton-emitted
// iota+reshape+broadcast+offset pointer-arithmetic feeding a
// cuda_tile.load_ptr_tko/store_ptr_tko and rewrites it into the higher-level
// make_tensor_view + make_partition_view + load_view_tko/store_view_tko form.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/CudaTileToGPU/TileIRPtrToView.h"

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::cuda_tile;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Walk through chains of `cuda_tile.assume` ops, returning the underlying
/// value.  Assume operations are pure metadata and have no semantic effect.
static Value lookThroughAssume(Value v) {
  while (v) {
    auto a = v.getDefiningOp<AssumeOp>();
    if (!a)
      break;
    v = a.getValue();
  }
  return v;
}

/// Returns `true` iff `tt` is a TileType with empty shape (scalar tile).
static bool isScalarTile(Type t) {
  auto tt = dyn_cast<TileType>(t);
  return tt && tt.getShape().empty();
}

/// Strip a chain of `broadcast` and `reshape` ops applied to a scalar tile,
/// returning the scalar tile value.  Assume ops are also transparent.
/// Returns null if `v` does not bottom out in a scalar tile through those ops.
static Value matchScalarBroadcastReshape(Value v) {
  while (v) {
    v = lookThroughAssume(v);
    if (!v)
      return nullptr;
    if (isScalarTile(v.getType()))
      return v;
    if (auto b = v.getDefiningOp<BroadcastOp>()) {
      v = b.getSource();
      continue;
    }
    if (auto r = v.getDefiningOp<ReshapeOp>()) {
      v = r.getSource();
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

/// Per-dimension information recovered from a Triton ptr-arithmetic chain.
struct DimInfo {
  /// Tile-side size (= shape of the tile produced by the load/store).
  int64_t tileSize = 0;
  /// Scalar `start` value (= base index into the global tensor along this
  /// dimension).  May be null, in which case start == 0.
  Value start;
  /// Scalar `stride` value (= number of elements between consecutive entries
  /// along this dimension in the global buffer).  May be null, in which case
  /// stride == 1.
  Value stride;
  /// Scalar `size` value extracted from the corresponding mask (the global
  /// tensor's size along this dimension).  May be null if no comparison was
  /// found for this dimension (we then fall back to the tile size).
  Value size;
};

/// Information recovered for a complete pointer-arithmetic chain.
struct PtrAccess {
  /// The scalar base pointer (tile<ptr<T>>).
  Value base;
  /// Per-tile-dimension info, ordered by dimension.
  SmallVector<DimInfo> dims;
};

/// Look at `addend` (a tile expression of shape `tileShape`) and try to figure
/// out which dimension it indexes (writing into `dim`), and recover its
/// `start` and `stride` scalar values.  Returns success on a recognised
/// pattern.
///
/// We accept the following shapes for `addend` (after stripping outer
/// broadcasts that simply replicate across orthogonal dimensions):
///
///   1. `reshape(addi(broadcast(reshape(start)), iota))` or
///      `reshape(iota)` — stride defaults to 1.
///   2. `muli(reshape(...above...), broadcast(reshape(stride)))` (operands may
///      be swapped) — stride is the broadcast scalar.
///   3. The pure 1-D forms (no outer reshape) — same content as above
///      collapsed.
static LogicalResult decomposeAddend(Value addend, ArrayRef<int64_t> tileShape,
                                     int &dim, DimInfo &info) {
  // Strip outer broadcasts that just replicate this 1-D pattern across
  // orthogonal dimensions of the tile.
  Value cur = lookThroughAssume(addend);
  while (cur) {
    if (auto bcast = cur.getDefiningOp<BroadcastOp>()) {
      cur = bcast.getSource();
      cur = lookThroughAssume(cur);
      continue;
    }
    break;
  }
  if (!cur)
    return failure();

  // Optional `muli` with one side being a broadcast-of-reshape-of-scalar (the
  // stride).
  if (auto mul = cur.getDefiningOp<MulIOp>()) {
    Value lhs = mul.getLhs();
    Value rhs = mul.getRhs();
    for (auto [a, b] : {std::pair<Value, Value>(lhs, rhs),
                        std::pair<Value, Value>(rhs, lhs)}) {
      if (Value scalar = matchScalarBroadcastReshape(a)) {
        info.stride = scalar;
        cur = b;
        break;
      }
    }
    if (!info.stride)
      return failure();
    cur = lookThroughAssume(cur);
    // Strip broadcasts again (the index side of the mul may itself be a
    // broadcast).
    while (cur) {
      if (auto bcast = cur.getDefiningOp<BroadcastOp>()) {
        cur = bcast.getSource();
        cur = lookThroughAssume(cur);
        continue;
      }
      break;
    }
  }
  if (!cur)
    return failure();

  // After stripping, we should have either a reshape of a 1-D tile, or a 1-D
  // tile directly.  In the reshape case, find the single non-1 dim of the
  // reshape result to determine which dimension of the tile this addend
  // populates.
  Value oneD = cur;
  if (auto rs = cur.getDefiningOp<ReshapeOp>()) {
    auto resShape = cast<TileType>(rs.getResult().getType()).getShape();
    int found = -1;
    for (int i = 0, e = resShape.size(); i < e; ++i) {
      if (resShape[i] == 1)
        continue;
      if (found != -1)
        return failure();
      found = i;
    }
    if (found < 0)
      return failure();
    dim = found;
    oneD = rs.getSource();
  } else {
    // 1-D tile: only valid when the tile itself is 1-D.
    if (tileShape.size() != 1)
      return failure();
    dim = 0;
  }
  oneD = lookThroughAssume(oneD);
  auto oneDTy = dyn_cast<TileType>(oneD.getType());
  if (!oneDTy || oneDTy.getShape().size() != 1)
    return failure();
  info.tileSize = oneDTy.getShape()[0];
  if (info.tileSize != tileShape[dim])
    return failure();

  // The 1-D content is either pure `iota` (start = 0) or
  // `addi(broadcast(reshape(start)), iota)`.
  if (oneD.getDefiningOp<IotaOp>())
    return success();
  if (auto add = oneD.getDefiningOp<AddIOp>()) {
    Value lhs = add.getLhs();
    Value rhs = add.getRhs();
    for (auto [a, b] : {std::pair<Value, Value>(lhs, rhs),
                        std::pair<Value, Value>(rhs, lhs)}) {
      if (lookThroughAssume(a).getDefiningOp<IotaOp>()) {
        if (Value scalar = matchScalarBroadcastReshape(b)) {
          info.start = scalar;
          return success();
        }
      }
    }
  }
  return failure();
}

/// Recognise the per-dim sizes encoded in `mask`.  We walk a tree of
/// `cmpi less_than` / `andi` / `exti` / `trunci` / `broadcast` / `reshape`
/// ops.  Each `cmpi less_than` compares a `reshape(iota...)` against a
/// broadcast-of-reshape-of-scalar — that scalar is the per-dim size.
static void analyzeMask(Value mask, unsigned rank,
                        SmallVectorImpl<Value> &dimSize) {
  dimSize.assign(rank, Value());
  SmallVector<Value> work;
  work.push_back(mask);
  while (!work.empty()) {
    Value v = lookThroughAssume(work.pop_back_val());
    if (!v)
      continue;
    if (auto a = v.getDefiningOp<AndIOp>()) {
      work.push_back(a.getLhs());
      work.push_back(a.getRhs());
      continue;
    }
    if (auto e = v.getDefiningOp<ExtIOp>()) {
      work.push_back(e.getFrom());
      continue;
    }
    if (auto t = v.getDefiningOp<TruncIOp>()) {
      work.push_back(t.getFrom());
      continue;
    }
    if (auto b = v.getDefiningOp<BroadcastOp>()) {
      work.push_back(b.getSource());
      continue;
    }
    if (auto cmp = v.getDefiningOp<CmpIOp>()) {
      // The index side may itself be reshape(iota+start) or
      // reshape(reshape...). Locate dim from the reshape that produces the
      // cmp's lhs.
      Value lhs = lookThroughAssume(cmp.getLhs());
      int dim = -1;
      if (auto rs = lhs.getDefiningOp<ReshapeOp>()) {
        auto shape = cast<TileType>(rs.getResult().getType()).getShape();
        if (shape.size() != rank)
          continue;
        for (int i = 0, e = shape.size(); i < e; ++i) {
          if (shape[i] == 1)
            continue;
          if (dim != -1) {
            dim = -1;
            break;
          }
          dim = i;
        }
      } else if (auto tt = dyn_cast<TileType>(lhs.getType())) {
        if (tt.getShape().size() == 1 && rank == 1)
          dim = 0;
      }
      if (dim < 0)
        continue;
      Value size = matchScalarBroadcastReshape(cmp.getRhs());
      if (!size)
        continue;
      if (!dimSize[dim])
        dimSize[dim] = size;
    }
  }
}

/// Walk `ptr` (a tile<ptr<T>>) backwards through `offset` ops and through
/// transparent reshape/broadcast ops on the ptr side, collecting one DimInfo
/// per encountered offset addend.  On success `out.base` is the scalar base
/// pointer and `out.dims` has been populated for every covered dimension.
static LogicalResult analyzePtr(Value ptr, ArrayRef<int64_t> tileShape,
                                PtrAccess &out) {
  out.dims.assign(tileShape.size(), DimInfo{});
  for (unsigned i = 0; i < tileShape.size(); ++i)
    out.dims[i].tileSize = tileShape[i];
  SmallVector<bool> covered(tileShape.size(), false);

  Value cur = ptr;
  while (cur) {
    cur = lookThroughAssume(cur);
    if (!cur)
      return failure();
    if (auto bcast = cur.getDefiningOp<BroadcastOp>()) {
      cur = bcast.getSource();
      continue;
    }
    if (auto rs = cur.getDefiningOp<ReshapeOp>()) {
      // Reshape of the base scalar ptr down/up.  Just strip.
      cur = rs.getSource();
      continue;
    }
    if (auto off = cur.getDefiningOp<OffsetOp>()) {
      DimInfo info;
      int dim = -1;
      if (failed(decomposeAddend(off.getOffset(), tileShape, dim, info)))
        return failure();
      if (dim < 0 || dim >= (int)tileShape.size())
        return failure();
      if (covered[dim])
        return failure();
      // Preserve the recovered tile size from analysis.
      out.dims[dim].start = info.start;
      out.dims[dim].stride = info.stride;
      covered[dim] = true;
      cur = off.getPtr();
      continue;
    }
    break;
  }
  // Base must be a scalar ptr tile.
  if (!cur || !isScalarTile(cur.getType()))
    return failure();
  out.base = cur;
  return success();
}

/// Match a `cuda_tile.constant` splat that encodes one of the supported
/// padding values.  Returns null when no match.
static PaddingValueAttr matchPadding(MLIRContext *ctx, Value v) {
  v = lookThroughAssume(v);
  auto cst = v.getDefiningOp<ConstantOp>();
  if (!cst)
    return nullptr;
  auto attr = cst.getValueAttr();
  auto fp = dyn_cast<DenseFPElementsAttr>(attr);
  if (!fp || !fp.isSplat())
    return nullptr;
  APFloat val = fp.getSplatValue<APFloat>();
  PaddingValue pv;
  if (val.isNaN())
    pv = PaddingValue::nan;
  else if (val.isInfinity() && val.isNegative())
    pv = PaddingValue::neg_inf;
  else if (val.isInfinity())
    pv = PaddingValue::pos_inf;
  else if (val.isZero() && val.isNegative())
    pv = PaddingValue::neg_zero;
  else if (val.isZero())
    pv = PaddingValue::zero;
  else
    return nullptr;
  return PaddingValueAttr::get(ctx, pv);
}

/// Build the (TensorViewType, PartitionViewType, dynamic-shape, dynamic-stride,
/// per-dim partition index) tuple for a recovered PtrAccess.
struct BuiltViews {
  TensorViewType tvTy;
  PartitionViewType pvTy;
  SmallVector<Value> dynamicShape;
  SmallVector<Value> dynamicStride;
  SmallVector<Value> indices;
};

/// Given a per-dim `start` scalar that is expected to be `multiplier *
/// tileSize`, extract the multiplier scalar (the per-dim partition index).
/// Returns null when the pattern doesn't match.
static Value extractTileMultiplier(Value start, int64_t tileSize) {
  if (!start)
    return nullptr;
  start = lookThroughAssume(start);
  auto mul = start.getDefiningOp<MulIOp>();
  if (!mul)
    return nullptr;
  for (auto [a, b] : {std::pair<Value, Value>(mul.getLhs(), mul.getRhs()),
                      std::pair<Value, Value>(mul.getRhs(), mul.getLhs())}) {
    Value cstSide = lookThroughAssume(a);
    if (auto cst = cstSide.getDefiningOp<ConstantOp>()) {
      auto attr = cst.getValueAttr();
      if (auto ints = dyn_cast<DenseIntElementsAttr>(attr)) {
        if (ints.isSplat() &&
            ints.getSplatValue<APInt>().getSExtValue() == tileSize)
          return b;
      }
    }
  }
  return nullptr;
}

/// Build a scalar `tile<i32>` constant of value 0 at the current insertion
/// point.
static Value buildZeroI32(OpBuilder &b, Location loc) {
  auto i32 = b.getI32Type();
  auto tileTy = TileType::get(b.getContext(), {}, i32);
  auto attr = DenseElementsAttr::get(tileTy, APInt(32, 0));
  return ConstantOp::create(b, loc, tileTy, cast<DenseTypedElementsAttr>(attr));
}

/// Materialise the views and per-dim indices required to express `access` as
/// a load_view_tko / store_view_tko at the current builder location.  Returns
/// failure when the per-dim partition index cannot be recovered.
static LogicalResult buildViews(OpBuilder &b, Location loc,
                                const PtrAccess &access, Type elementType,
                                PaddingValueAttr padding, BuiltViews &out) {
  MLIRContext *ctx = b.getContext();
  unsigned rank = access.dims.size();

  // 1) Compute per-dim partition indices by stripping the `tileSize * idx`
  //    multiplication out of the `start` scalar.  Bail early when this fails
  //    (the rewrite would otherwise lose information about the alignment of
  //    each tile within the global tensor).
  SmallVector<Value> indices;
  indices.reserve(rank);
  for (unsigned d = 0; d < rank; ++d) {
    const DimInfo &di = access.dims[d];
    if (!di.start) {
      indices.push_back(buildZeroI32(b, loc));
      continue;
    }
    Value idx = extractTileMultiplier(di.start, di.tileSize);
    if (!idx)
      return failure();
    indices.push_back(idx);
  }

  // 2) Build the TensorViewType.  Shape is dynamic per-dim (extracted from
  //    the recovered mask sizes, or falls back to the tile size).  The
  //    innermost dim is contiguous (stride == 1); other dims are dynamic
  //    when their stride was recovered from a `muli`, else also 1.
  SmallVector<int64_t> shape(rank, TensorViewType::kDynamic);
  SmallVector<int64_t> strides(rank, TensorViewType::kDynamic);
  SmallVector<Value> dynShape, dynStride;
  for (unsigned d = 0; d < rank; ++d) {
    const DimInfo &di = access.dims[d];
    if (di.size) {
      dynShape.push_back(di.size);
    } else {
      // No mask info: shape unknown.  Fall back to a static tile-aligned shape.
      shape[d] = di.tileSize;
    }
    if (di.stride) {
      dynStride.push_back(di.stride);
    } else {
      strides[d] = 1;
    }
  }
  auto tvTy = TensorViewType::get(ctx, elementType, shape, strides);

  // 3) Build the PartitionViewType.  tile_shape is the static per-dim tile
  //    size in i32; dim_map is the identity.
  SmallVector<int32_t> tileShape32;
  tileShape32.reserve(rank);
  for (unsigned d = 0; d < rank; ++d)
    tileShape32.push_back(static_cast<int32_t>(access.dims[d].tileSize));
  auto tileShapeAttr = DenseI32ArrayAttr::get(ctx, tileShape32);
  SmallVector<int32_t> dimMap(rank);
  for (unsigned d = 0; d < rank; ++d)
    dimMap[d] = static_cast<int32_t>(d);
  if (!padding)
    padding = PaddingValueAttr::get(ctx, PaddingValue::zero);
  auto pvTy = PartitionViewType::get(ctx, tileShapeAttr, tvTy, dimMap, padding);

  out.tvTy = tvTy;
  out.pvTy = pvTy;
  out.dynamicShape = std::move(dynShape);
  out.dynamicStride = std::move(dynStride);
  out.indices = std::move(indices);
  return success();
}

//===----------------------------------------------------------------------===//
// Rewrite drivers
//===----------------------------------------------------------------------===//

static LogicalResult rewriteLoad(LoadPtrTkoOp op) {
  Location loc = op.getLoc();
  auto resultTy = cast<TileType>(op.getResult().getType());
  Type elemTy = resultTy.getElementType();
  ArrayRef<int64_t> tileShape = resultTy.getShape();

  // The pass currently requires a mask so that we can recover the per-dim
  // global sizes.
  if (!op.getMask()) {
    op.emitRemark("tileir-ptr-to-view: load has no mask; skipping");
    return failure();
  }

  PaddingValueAttr padding;
  if (op.getPaddingValue()) {
    padding = matchPadding(op.getContext(), op.getPaddingValue());
    if (!padding) {
      op.emitRemark("tileir-ptr-to-view: padding value not recognised");
      return failure();
    }
  }

  PtrAccess access;
  if (failed(analyzePtr(op.getSource(), tileShape, access))) {
    op.emitRemark("tileir-ptr-to-view: pointer-arithmetic pattern not "
                  "recognised; skipping");
    return failure();
  }

  // Recover per-dim sizes from the mask.
  SmallVector<Value> dimSizes;
  analyzeMask(op.getMask(), tileShape.size(), dimSizes);
  for (unsigned d = 0; d < tileShape.size(); ++d)
    access.dims[d].size = dimSizes[d];

  OpBuilder b(op);
  BuiltViews bv;
  if (failed(buildViews(b, loc, access, elemTy, padding, bv))) {
    op.emitRemark("tileir-ptr-to-view: could not recover per-dim partition "
                  "index; skipping");
    return failure();
  }
  auto tvOp = MakeTensorViewOp::create(b, loc, bv.tvTy, access.base,
                                       bv.dynamicShape, bv.dynamicStride);
  auto pvOp = MakePartitionViewOp::create(b, loc, bv.pvTy, tvOp.getResult());

  auto moAttr = MemoryOrderingSemanticsAttr::get(op.getContext(),
                                                 MemoryOrderingSemantics::WEAK);
  auto newOp = LoadViewTkoOp::create(
      b, loc, resultTy, op.getResultToken().getType(), moAttr,
      /*memory_scope=*/nullptr, pvOp.getResult(), bv.indices,
      /*token=*/op.getToken(), op.getOptimizationHintsAttr());
  op.getResult().replaceAllUsesWith(newOp.getTile());
  op.getResultToken().replaceAllUsesWith(newOp.getResultToken());
  op.erase();
  return success();
}

static LogicalResult rewriteStore(StorePtrTkoOp op) {
  Location loc = op.getLoc();
  auto valueTy = cast<TileType>(op.getValue().getType());
  Type elemTy = valueTy.getElementType();
  ArrayRef<int64_t> tileShape = valueTy.getShape();

  if (!op.getMask()) {
    op.emitRemark("tileir-ptr-to-view: store has no mask; skipping");
    return failure();
  }

  PtrAccess access;
  if (failed(analyzePtr(op.getDestination(), tileShape, access))) {
    op.emitRemark("tileir-ptr-to-view: pointer-arithmetic pattern not "
                  "recognised; skipping");
    return failure();
  }
  SmallVector<Value> dimSizes;
  analyzeMask(op.getMask(), tileShape.size(), dimSizes);
  for (unsigned d = 0; d < tileShape.size(); ++d)
    access.dims[d].size = dimSizes[d];

  OpBuilder b(op);
  BuiltViews bv;
  PaddingValueAttr padding =
      PaddingValueAttr::get(op.getContext(), PaddingValue::zero);
  if (failed(buildViews(b, loc, access, elemTy, padding, bv))) {
    op.emitRemark("tileir-ptr-to-view: could not recover per-dim partition "
                  "index; skipping");
    return failure();
  }
  auto tvOp = MakeTensorViewOp::create(b, loc, bv.tvTy, access.base,
                                       bv.dynamicShape, bv.dynamicStride);
  auto pvOp = MakePartitionViewOp::create(b, loc, bv.pvTy, tvOp.getResult());

  auto moAttr = MemoryOrderingSemanticsAttr::get(op.getContext(),
                                                 MemoryOrderingSemantics::WEAK);
  auto newOp = StoreViewTkoOp::create(
      b, loc, op.getResultToken().getType(), moAttr, /*memory_scope=*/nullptr,
      op.getValue(), pvOp.getResult(), bv.indices, /*token=*/op.getToken(),
      op.getOptimizationHintsAttr());
  op.getResultToken().replaceAllUsesWith(newOp.getResultToken());
  op.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct TileIRPtrToViewPass
    : public PassWrapper<TileIRPtrToViewPass, OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileIRPtrToViewPass)

  StringRef getArgument() const final { return "tileir-ptr-to-view"; }
  StringRef getDescription() const final {
    return "Rewrite Triton-style cuda_tile ptr-arithmetic into "
           "make_tensor_view + make_partition_view + load/store_view_tko";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cuda_tile::CudaTileDialect>();
  }

  void runOnOperation() override {
    ::mlir::ModuleOp mod = getOperation();

    // Collect load/store ops first to avoid mutating the IR while walking.
    SmallVector<LoadPtrTkoOp> loads;
    SmallVector<StorePtrTkoOp> stores;
    mod.walk([&](Operation *op) {
      if (auto l = dyn_cast<LoadPtrTkoOp>(op))
        loads.push_back(l);
      else if (auto s = dyn_cast<StorePtrTkoOp>(op))
        stores.push_back(s);
    });

    for (auto l : loads)
      (void)rewriteLoad(l);
    for (auto s : stores)
      (void)rewriteStore(s);

    // Clean up now-dead pure cuda_tile ops left over from the ptr-arithmetic
    // chains.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> toErase;
      mod.walk([&](Operation *op) {
        if (op->getNumResults() == 0)
          return;
        if (!op->use_empty())
          return;
        if (!isa<OffsetOp, BroadcastOp, ReshapeOp, IotaOp, MulIOp, AddIOp,
                 CmpIOp, AndIOp, ExtIOp, TruncIOp>(op))
          return;
        if (!isMemoryEffectFree(op))
          return;
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

std::unique_ptr<OperationPass<::mlir::ModuleOp>>
mlir::createTileIRPtrToViewPass() {
  return std::make_unique<TileIRPtrToViewPass>();
}
