//===- TileIRPtrToView.cpp - ptr-arith -> CudaTile view ops ---------------===//
//
// Pre-conversion pass that recognises the canonical
// iota+reshape+broadcast+offset pointer-arithmetic feeding a
// cuda_tile.load_ptr_tko/store_ptr_tko and rewrites it into the higher-level
// make_tensor_view + make_partition_view + load_view_tko/store_view_tko form.
//
// ============================================================================
// Detected pattern (per load/store, shown for a 2-D tile<NxM x T>)
// ============================================================================
//
//   // base pointer (scalar)
//   %base : tile<ptr<T>>
//
//   // per-dim index construction (one offset op per dimension)
//   %iota0  = iota                           : tile<N x i32>
//   %start0 = ...                            : tile<ptr<T>>  // scalar
//   %off0   = addi broadcast(reshape(%start0)), reshape(%iota0)
//                                            : tile<N x i32>
//   // optional: %off0 = muli %off0, broadcast(reshape(%stride0))
//   %ptr1   = offset %base,   reshape(%off0) : tile<N x 1 x ptr<T>>
//
//   %iota1  = iota                           : tile<M x i32>
//   %start1 = ...                            : tile<ptr<T>>  // scalar
//   %off1   = addi broadcast(reshape(%start1)), reshape(%iota1)
//                                            : tile<M x i32>
//   // optional: %off1 = muli %off1, broadcast(reshape(%stride1))
//   %ptr    = offset %ptr1,   reshape(%off1) : tile<N x M x ptr<T>>
//
//   // mask encodes per-dim bounds
//   %mask   = andi cmpi(lt, reshape(%iota0), broadcast(reshape(%size0))),
//                  cmpi(lt, reshape(%iota1), broadcast(reshape(%size1)))
//                                            : tile<N x M x i1>
//
//   %result = load_ptr_tko %ptr, %mask [, %pad] : tile<N x M x T>
//
// The start values are expected to be of the form `muli(%idx_d, tile_size_d)`,
// allowing the per-dimension partition index %idx_d to be recovered.
//
// The rewrite is conservative: it leaves the original
// load_ptr_tko/store_ptr_tko untouched whenever it cannot fully recover the
// access, rather than fabricating a shape/stride. In particular it requires:
//   * a recovered global size (from the mask) for every dimension;
//   * the innermost dimension to be contiguous (unit stride) and every other
//     dimension to carry an explicitly recovered stride (row-major layout);
//   * exactly one loop-advancing (start-less) dimension when the access is
//     carried by a `for` iter_arg.
//
// ============================================================================
// Produced pattern
// ============================================================================
//
//   // recovered scalars: %idx0 = start0 / N, %idx1 = start1 / M
//   %tv  = make_tensor_view %base [%size0, %size1] [%stride0, 1]
//              : tensor_view<T, [?, ?], [?, 1]>
//   %pv  = make_partition_view %tv
//              : partition_view<tile_shape=[N, M], dim_map=[0, 1], ...>
//   %result = load_view_tko %pv [%idx0, %idx1] : tile<N x M x T>
//
// The store pattern is identical except that load_ptr_tko/load_view_tko are
// replaced by store_ptr_tko/store_view_tko (no padding argument).
//
// After the rewrites, dead pure ops left over from the pointer-arithmetic
// chains (offset, broadcast, reshape, iota, muli, addi, cmpi, andi, exti,
// trunci) are removed iteratively.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/CudaTileToGPU/TileIRPtrToView.h"

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
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

/// The "definition point" operation for `v`: its defining op, or the first op
/// of its owning block when `v` is a block argument (block args are available
/// at block entry, i.e. before every op in the block).  Returns null only for a
/// block argument of an empty block.
static Operation *defPointOp(Value v) {
  if (Operation *d = v.getDefiningOp())
    return d;
  Block *b = cast<BlockArgument>(v).getOwner();
  return b->empty() ? nullptr : &b->front();
}

/// Set `b`'s insertion point immediately after the latest-defined value among
/// `values` (all of which dominate `anchor`, hence are totally ordered by
/// dominance), so the ops built there appear as early as legally possible.
/// Falls back to `anchor` when `values` is empty.  `dom` must be valid for the
/// region being edited (only operations are added to existing blocks here, so a
/// single instance stays valid across the rewrite).
static void setInsertionPointAfterLatestDef(OpBuilder &b, DominanceInfo &dom,
                                            ArrayRef<Value> values,
                                            Operation *anchor) {
  if (values.empty()) {
    b.setInsertionPoint(anchor);
    return;
  }
  Value latest = values.front();
  for (Value v : values.drop_front()) {
    Operation *lp = defPointOp(latest);
    Operation *vp = defPointOp(v);
    // `v` is strictly later than `latest` iff `latest`'s point dominates `v`'s.
    if (lp && vp && dom.properlyDominates(lp, vp))
      latest = v;
  }
  if (Operation *d = latest.getDefiningOp())
    b.setInsertionPointAfter(d);
  else
    b.setInsertionPointToStart(cast<BlockArgument>(latest).getOwner());
}

/// Forwards `assume` metadata that the source IR attached to the values the
/// rewrite reuses (base pointers and shape/stride scalars) onto the new view
/// operands, preserving alignment/bounds metadata (e.g. `div_by`) on
/// `make_tensor_view`.
///
/// Forwarding is deduplicated and shared across all accesses:
///   * an existing source `assume` op carrying the wanted predicate is reused
///     directly when it dominates the use (so e.g. a `div_by<16>` on `%argN`
///     feeds every view that references `%argN` instead of being re-emitted);
///   * otherwise a single fresh `assume` is created immediately after the
///     definition of the value it annotates, and cached for reuse.
/// Chains of `assume` ops (`assume` of an `assume`) are followed transitively,
/// since each link is a value-preserving passthrough decorating the same
/// scalar. Only predicates the source already attached are forwarded.
struct AssumeForwarder {
  DominanceInfo &dom;
  DenseMap<std::pair<Value, Attribute>, Value> cache;

  explicit AssumeForwarder(DominanceInfo &dom) : dom(dom) {}

  /// Returns `scalar` decorated with all source assume predicates.  `anchor` is
  /// the load/store op the rewritten access is being built for.
  Value forward(Value scalar, Operation *anchor) {
    SmallVector<AssumePredicateAttrInterface> preds;
    SmallVector<Value> work{scalar};
    SmallPtrSet<Value, 4> seen;
    while (!work.empty()) {
      Value v = work.pop_back_val();
      for (Operation *user : v.getUsers()) {
        auto a = dyn_cast<AssumeOp>(user);
        if (!a || a.getValue() != v)
          continue;
        AssumePredicateAttrInterface p = a.getPredicateAttr();
        if (!llvm::is_contained(preds, p))
          preds.push_back(p);
        if (seen.insert(a.getResult()).second)
          work.push_back(a.getResult());
      }
    }
    Value result = scalar;
    for (AssumePredicateAttrInterface p : preds)
      result = getOrCreate(result, p, anchor);
    return result;
  }

private:
  Value getOrCreate(Value base, AssumePredicateAttrInterface pred,
                    Operation *anchor) {
    std::pair<Value, Attribute> key{base, pred};
    if (Value cached = cache.lookup(key))
      return cached;

    // Prefer reusing an existing source `assume` op carrying this predicate.
    for (Operation *user : base.getUsers()) {
      auto a = dyn_cast<AssumeOp>(user);
      if (!a || a.getValue() != base || a.getPredicateAttr() != pred)
        continue;
      if (!dom.dominates(a.getResult(), anchor))
        continue;
      cache[key] = a.getResult();
      return a.getResult();
    }

    // Otherwise create one fresh assume right after the def of `base`.
    OpBuilder b(base.getContext());
    if (auto barg = dyn_cast<BlockArgument>(base))
      b.setInsertionPointToStart(barg.getOwner());
    else
      b.setInsertionPointAfter(base.getDefiningOp());
    Value res = AssumeOp::create(b, base.getLoc(), base, pred).getResult();
    cache[key] = res;
    return res;
  }
};

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

/// Per-dimension information recovered from a ptr-arithmetic chain.
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
  /// If the pointer source was a for-loop iter_arg, this captures the context.
  struct LoopInfo {
    ForOp forOp;
    /// Index of this iter_arg within the ForOp's region iter values.
    unsigned iterArgIndex = 0;
    /// The loop induction variable (tile<i32>).
    Value inductionVar;
  };
  std::optional<LoopInfo> loop;
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
///
/// If the pointer source is a for-loop iter_arg, the function traces through
/// to the initial value and records loop advancement info in `out.loop`.
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
      if (succeeded(decomposeAddend(off.getOffset(), tileShape, dim, info))) {
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

      // Unrecognised offset addend. If the offset op itself produces a scalar
      // pointer tile (e.g. `offset(base, row_start)` that pre-shifts the base
      // to the beginning of a row), stop tracing here and use `cur` as the
      // base.  The pre-computed start is already baked into the pointer;
      // partition index 0 will land on the correct element without needing to
      // recover the multiplier.
      if (isScalarTile(cur.getType()))
        break;

      return failure();
    }
    // Handle for-loop iter_arg: trace through to initial value and record
    // per-iteration advancement.
    if (auto blockArg = dyn_cast<BlockArgument>(cur)) {
      auto *parentOp = blockArg.getOwner()->getParentOp();
      auto forOp = dyn_cast_or_null<ForOp>(parentOp);
      if (!forOp)
        break; // Not a for-loop iter_arg — treat as base pointer.
      // Must be an iter_arg (not the induction variable).
      unsigned argNum = blockArg.getArgNumber();
      if (argNum < forOp.getNumInductionVars())
        return failure();
      unsigned iterIdx = argNum - forOp.getNumInductionVars();

      // Analyze per-iteration step from the ContinueOp (loop terminator).
      // The yielded value should be `offset(same_iter_arg, step_addend)`.
      auto contOp = cast<ContinueOp>(forOp.getBody()->getTerminator());
      Value yieldedPtr = contOp.getOperand(iterIdx);
      Value yieldCur = lookThroughAssume(yieldedPtr);
      if (auto yOff = yieldCur.getDefiningOp<OffsetOp>()) {
        Value yPtr = lookThroughAssume(yOff.getPtr());
        if (yPtr != blockArg)
          return failure();
      } else {
        return failure();
      }

      // Record loop info.  The exact advancing dimension will be determined
      // later based on which dim has no start in the initial value analysis.
      PtrAccess::LoopInfo li;
      li.forOp = forOp;
      li.iterArgIndex = iterIdx;
      li.inductionVar = forOp.getInductionVar();
      out.loop = std::move(li);

      // Follow to the initial value of this iter_arg.
      cur = forOp.getInitValues()[iterIdx];
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
  DenseFPElementsAttr fp;
  if (!matchPattern(v, m_Constant(&fp)) || !fp.isSplat())
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
    DenseIntElementsAttr ints;
    if (matchPattern(cstSide, m_Constant(&ints)) && ints.isSplat() &&
        ints.getSplatValue<APInt>().getSExtValue() == tileSize)
      return b;
  }
  return nullptr;
}

/// Build a scalar `tile<i32>` constant of value 0 at the current insertion
/// point.
static Value buildZeroI32(OpBuilder &b, Location loc) {
  auto i32 = b.getI32Type();
  auto tileTy = TileType::get(b.getContext(), {}, i32);
  auto attr = DenseElementsAttr::get(tileTy, APInt(32, 0));
  return ConstantOp::create(b, loc, tileTy,
                            cast<DenseIntOrFPElementsAttr>(attr));
}

/// Materialise the views and per-dim indices required to express `access` as
/// a load_view_tko / store_view_tko at the current builder location.  Returns
/// failure when the per-dim partition index cannot be recovered.
///
/// When `access.loop` is set, the index for the advancing dimension becomes
/// `initial_idx + loopIdx` (the loop induction variable).
static LogicalResult buildViews(OpBuilder &b, Location loc,
                                const PtrAccess &access, Type elementType,
                                PaddingValueAttr padding, BuiltViews &out) {
  MLIRContext *ctx = b.getContext();
  unsigned rank = access.dims.size();

  // When the access advances through a loop iter_arg, exactly one tile
  // dimension must lack a `start` (that is the dimension the loop advances).
  // More than one start-less dimension makes the advancing dimension
  // ambiguous, so we bail rather than guess.
  if (access.loop) {
    unsigned numStartless = 0;
    for (const DimInfo &di : access.dims)
      if (!di.start)
        ++numStartless;
    if (numStartless != 1)
      return failure();
  }

  // 1) Compute per-dim partition indices by stripping the `tileSize * idx`
  //    multiplication out of the `start` scalar.  Bail early when this fails
  //    (the rewrite would otherwise lose information about the alignment of
  //    each tile within the global tensor).
  //
  //    When the access comes from a for-loop iter_arg, the advancing dimension
  //    (the one with no start in the initial pattern) uses the loop induction
  //    variable as its partition index.
  SmallVector<Value> indices;
  indices.reserve(rank);
  for (unsigned d = 0; d < rank; ++d) {
    const DimInfo &di = access.dims[d];

    // When a loop is present, the advancing dimension is identified as
    // the one without a start value.
    bool isLoopAdvancingDim = access.loop && !di.start;

    if (!di.start && !isLoopAdvancingDim) {
      indices.push_back(buildZeroI32(b, loc));
      continue;
    }

    // Compute the base index (from initial ptr pattern).
    Value baseIdx;
    if (di.start) {
      baseIdx = extractTileMultiplier(di.start, di.tileSize);
      if (!baseIdx)
        return failure();
    }

    if (isLoopAdvancingDim) {
      // The partition index for this dim is just loopIdx (induction var).
      Value loopIdx = access.loop->inductionVar;
      if (baseIdx) {
        baseIdx = AddIOp::create(b, loc, baseIdx, loopIdx);
      } else {
        baseIdx = loopIdx;
      }
    }

    indices.push_back(baseIdx ? baseIdx : buildZeroI32(b, loc));
  }

  // 2) Validate the recovered layout and build the TensorViewType.  We refuse
  //    to fabricate missing information:
  //      * every dimension must have a recovered global size (from the mask),
  //        otherwise the tensor extent -- and thus the masking behaviour --
  //        would be unknown;
  //      * only the innermost dimension may be contiguous (unit stride); every
  //        other dimension must carry an explicitly recovered stride.  This
  //        matches the canonical layout produced by the TileIR frontend.
  for (unsigned d = 0; d < rank; ++d) {
    const DimInfo &di = access.dims[d];
    if (!di.size)
      return failure();
    bool isMinor = (d == rank - 1);
    if (isMinor && di.stride)
      return failure(); // innermost dim must be contiguous
    if (!isMinor && !di.stride)
      return failure(); // outer dims must have an explicit stride
  }

  // Shape is fully dynamic (taken from the recovered mask sizes).  Strides are
  // dynamic for the outer dims and a static 1 for the contiguous innermost dim.
  SmallVector<int64_t> shape(rank, TensorViewType::kDynamic);
  SmallVector<int64_t> strides(rank, TensorViewType::kDynamic);
  SmallVector<Value> dynShape, dynStride;
  for (unsigned d = 0; d < rank; ++d) {
    const DimInfo &di = access.dims[d];
    dynShape.push_back(di.size);
    if (di.stride)
      dynStride.push_back(di.stride);
    else
      strides[d] = 1;
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

/// Analyze the pointer-arithmetic chain feeding a load/store (`ptr` + `mask`)
/// and materialise the corresponding `make_tensor_view` + `make_partition_view`
/// at the builder's current insertion point.  On success `outView` is the
/// partition-view value and `outIndices` are the per-dim partition indices to
/// pass to the load_view_tko/store_view_tko.  `mask` must be non-null.
static LogicalResult lowerAccess(OpBuilder &b, Location loc, Value ptr,
                                 Value mask, ArrayRef<int64_t> tileShape,
                                 Type elemTy, PaddingValueAttr padding,
                                 AssumeForwarder &fwd, Operation *anchor,
                                 Value &outView,
                                 SmallVectorImpl<Value> &outIndices) {
  PtrAccess access;
  if (failed(analyzePtr(ptr, tileShape, access)))
    return failure();

  // Recover per-dim global sizes from the mask.
  SmallVector<Value> dimSizes;
  analyzeMask(mask, tileShape.size(), dimSizes);
  for (unsigned d = 0; d < tileShape.size(); ++d)
    access.dims[d].size = dimSizes[d];

  BuiltViews bv;
  if (failed(buildViews(b, loc, access, elemTy, padding, bv)))
    return failure();

  // Forward any `assume` metadata the source attached to the operands we reuse
  // (base pointer, dynamic shape/stride scalars) onto the rewritten view.
  Value base = fwd.forward(access.base, anchor);
  SmallVector<Value> dynShape, dynStride;
  dynShape.reserve(bv.dynamicShape.size());
  dynStride.reserve(bv.dynamicStride.size());
  for (Value v : bv.dynamicShape)
    dynShape.push_back(fwd.forward(v, anchor));
  for (Value v : bv.dynamicStride)
    dynStride.push_back(fwd.forward(v, anchor));

  // Build the views right after the last definition among their operands so
  // they appear as early as legally possible (e.g. hoisted out of a loop whose
  // induction variable they do not depend on), rather than at the load/store.
  SmallVector<Value> viewOperands;
  viewOperands.push_back(base);
  viewOperands.append(dynShape.begin(), dynShape.end());
  viewOperands.append(dynStride.begin(), dynStride.end());
  OpBuilder vb(b.getContext());
  setInsertionPointAfterLatestDef(vb, fwd.dom, viewOperands, anchor);

  auto tvOp =
      MakeTensorViewOp::create(vb, loc, bv.tvTy, base, dynShape, dynStride);
  auto pvOp = MakePartitionViewOp::create(vb, loc, bv.pvTy, tvOp.getResult());
  outView = pvOp.getResult();
  outIndices.assign(bv.indices.begin(), bv.indices.end());
  return success();
}

static LogicalResult rewriteLoad(LoadPtrTkoOp op, AssumeForwarder &fwd) {
  Location loc = op.getLoc();
  auto resultTy = cast<TileType>(op.getResult().getType());
  Type elemTy = resultTy.getElementType();
  ArrayRef<int64_t> tileShape = resultTy.getShape();

  // The pass requires a mask so that we can recover the per-dim global sizes.
  // Rank-0 (scalar) loads carry no per-dim information and are lowered
  // directly by --convert-cuda-tile-to-gpu, so we silently skip them here
  // rather than emitting a misleading remark.
  if (!op.getMask()) {
    if (!tileShape.empty())
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

  OpBuilder b(op);
  Value view;
  SmallVector<Value> indices;
  if (failed(lowerAccess(b, loc, op.getSource(), op.getMask(), tileShape,
                         elemTy, padding, fwd, op, view, indices))) {
    op.emitRemark("tileir-ptr-to-view: pointer-arithmetic pattern not "
                  "recognised; skipping");
    return failure();
  }

  auto moAttr = MemoryOrderingSemanticsAttr::get(op.getContext(),
                                                 MemoryOrderingSemantics::WEAK);
  auto newOp = LoadViewTkoOp::create(
      b, loc, resultTy, op.getResultToken().getType(), moAttr,
      /*memory_scope=*/nullptr, view, indices,
      /*token=*/op.getToken(), op.getOptimizationHintsAttr());
  op.getResult().replaceAllUsesWith(newOp.getTile());
  op.getResultToken().replaceAllUsesWith(newOp.getResultToken());
  op.erase();
  return success();
}

static LogicalResult rewriteStore(StorePtrTkoOp op, AssumeForwarder &fwd) {
  Location loc = op.getLoc();
  auto valueTy = cast<TileType>(op.getValue().getType());
  Type elemTy = valueTy.getElementType();
  ArrayRef<int64_t> tileShape = valueTy.getShape();

  // Same rationale as in rewriteLoad: scalar (rank-0) stores are handled by
  // the direct --convert-cuda-tile-to-gpu pattern; only emit the remark for
  // higher-rank stores that genuinely need a mask.
  if (!op.getMask()) {
    if (!tileShape.empty())
      op.emitRemark("tileir-ptr-to-view: store has no mask; skipping");
    return failure();
  }

  OpBuilder b(op);
  // Stores mask out-of-bounds elements, so the padding value is never observed;
  // we use `zero` to match the canonical partition view emitted by the TileIR
  // frontend.
  PaddingValueAttr padding =
      PaddingValueAttr::get(op.getContext(), PaddingValue::zero);
  Value view;
  SmallVector<Value> indices;
  if (failed(lowerAccess(b, loc, op.getDestination(), op.getMask(), tileShape,
                         elemTy, padding, fwd, op, view, indices))) {
    op.emitRemark("tileir-ptr-to-view: pointer-arithmetic pattern not "
                  "recognised; skipping");
    return failure();
  }

  auto moAttr = MemoryOrderingSemanticsAttr::get(op.getContext(),
                                                 MemoryOrderingSemantics::WEAK);
  auto newOp = StoreViewTkoOp::create(
      b, loc, op.getResultToken().getType(), moAttr, /*memory_scope=*/nullptr,
      op.getValue(), view, indices, /*token=*/op.getToken(),
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

    // Dominance is queried to decide whether an existing source `assume` may be
    // reused and to place the new view ops after their latest-defined operand.
    // The load/store rewrites only add ops to existing blocks (never add, move,
    // or erase blocks), so a single instance stays valid across the phase.
    DominanceInfo domInfo(mod);
    AssumeForwarder fwd(domInfo);
    for (auto l : loads)
      (void)rewriteLoad(l, fwd);
    for (auto s : stores)
      (void)rewriteStore(s, fwd);

    // Remove dead iter_args from for loops. After loads/stores are rewritten
    // the ptr-typed results become unused; rebuild the loop without them.
    //
    // This is done by hand because cuda_tile's ForOp does not implement
    // RegionBranchOpInterface/LoopLikeOpInterface and ships no canonicalizer,
    // so the upstream dead-iter-arg elimination cannot be reused here.
    SmallVector<ForOp> forOps;
    mod.walk([&](ForOp op) { forOps.push_back(op); });
    for (ForOp forOp : forOps) {
      unsigned numIter = forOp.getNumResults();
      if (numIter == 0)
        continue;
      SmallVector<unsigned> keepIter;
      for (unsigned i = 0; i < numIter; ++i) {
        if (!forOp.getResult(i).use_empty())
          keepIter.push_back(i);
      }
      if (keepIter.size() == numIter)
        continue;

      unsigned numInduction = forOp.getNumInductionVars();
      OpBuilder builder(forOp);
      SmallVector<Value> newInits;
      for (unsigned i : keepIter)
        newInits.push_back(forOp.getInitValues()[i]);

      auto newFor =
          ForOp::create(builder, forOp.getLoc(), forOp.getLowerBound(),
                        forOp.getUpperBound(), forOp.getStep(), newInits);

      // Map old block args → new block args.
      Block *oldBody = forOp.getBody();
      Block *newBody = newFor.getBody();
      oldBody->getArgument(0).replaceAllUsesWith(newBody->getArgument(0));
      for (unsigned newIdx = 0; newIdx < keepIter.size(); ++newIdx) {
        unsigned oldIdx = keepIter[newIdx];
        oldBody->getArgument(numInduction + oldIdx)
            .replaceAllUsesWith(newBody->getArgument(numInduction + newIdx));
      }
      // Dropped iter_args: erase all their uses (which are dead ops feeding
      // only the continue).
      SmallVector<unsigned> dropIter;
      for (unsigned i = 0; i < numIter; ++i) {
        if (forOp.getResult(i).use_empty())
          dropIter.push_back(i);
      }
      for (unsigned i : dropIter) {
        Value deadArg = oldBody->getArgument(numInduction + i);
        // Collect and erase users of the dead arg (they only feed continue).
        SmallVector<Operation *> toErase;
        for (OpOperand &use : deadArg.getUses()) {
          Operation *user = use.getOwner();
          if (!isa<ContinueOp>(user))
            toErase.push_back(user);
        }
        for (Operation *op : toErase)
          op->dropAllUses();
        for (Operation *op : toErase)
          op->erase();
      }

      // Splice old body ops into new body (replacing its auto-generated
      // terminator if any).
      if (newBody->mightHaveTerminator())
        newBody->getTerminator()->erase();
      newBody->getOperations().splice(newBody->end(), oldBody->getOperations());

      // Fix the continue op to only yield the kept values.
      auto contOp = cast<ContinueOp>(newBody->getTerminator());
      SmallVector<Value> newYields;
      for (unsigned i : keepIter)
        newYields.push_back(contOp.getOperand(i));
      builder.setInsertionPoint(contOp);
      ContinueOp::create(builder, contOp.getLoc(), newYields);
      contOp.erase();

      // Replace kept results and erase old for.
      for (unsigned newIdx = 0; newIdx < keepIter.size(); ++newIdx)
        forOp.getResult(keepIter[newIdx])
            .replaceAllUsesWith(newFor.getResult(newIdx));
      forOp.erase();
    }

    // Erase the now-dead ptr-arithmetic ops to a fixed point.
    // `isOpTriviallyDead` covers the Pure ptr-arith ops; `assume` is handled
    // explicitly because it is not Pure but is safe to drop once unused.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> toErase;
      mod.walk([&](Operation *op) {
        if (!op->use_empty())
          return;
        if (isOpTriviallyDead(op) || isa<AssumeOp>(op))
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
