//===- PostConversion.cpp - Rewrites after the Tile IR conversion ---------===//
//
// Part of the tileir-to-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Rewrites of the IR that --convert-tileir-to-mlir produces. They run after the
// dialect conversion so that they see the final vector, arith and scf ops:
//   - rescaleTileLoops makes loops over tile indices step over element indices,
//     so that transfers are indexed by the induction variable itself;
//   - scopeLoopAllocations bounds stack allocations in loops to one iteration,
//     as in cuda_tile.for;
//   - local patterns fold selects into transfer_read masks, load contiguous
//     gather rows with masked loads, and flatten n-D gathers and scatters for
//     the LLVM lowering.
// Facts beyond the local IR are proven with integer range analysis
// (IntegerFacts).
//
//===----------------------------------------------------------------------===//

#include "PostConversion.h"

#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace {

/// Value identities of integer arithmetic, proven with integer range analysis.
class IntegerFacts {
public:
  explicit IntegerFacts(Operation *root) {
    dataflow::loadBaselineAnalyses(solver);
    solver.load<dataflow::IntegerRangeAnalysis>();
    valid = succeeded(solver.initializeAndRun(root));
  }

  bool isValid() const { return valid; }

  std::optional<ConstantIntRanges> getRange(Value value) const {
    auto *lattice =
        solver.lookupState<dataflow::IntegerValueRangeLattice>(value);
    if (!lattice || lattice->getValue().isUninitialized())
      return std::nullopt;
    return lattice->getValue().getValue();
  }

  bool isNonNegative(Value value) const {
    std::optional<ConstantIntRanges> range = getRange(value);
    return range && range->smin().isNonNegative();
  }

  /// Whether `value * factor` cannot overflow a 63-bit signed integer, which
  /// leaves headroom for a loop step past the upper bound.
  bool fitsScaled(Value value, int64_t factor) const {
    std::optional<ConstantIntRanges> range = getRange(value);
    if (!range)
      return false;
    for (const APInt &bound : {range->smin(), range->smax()}) {
      bool overflow = false;
      APInt scaled = bound.sext(64).smul_ov(
          APInt(64, factor, /*isSigned=*/true), overflow);
      if (overflow || scaled.getSignificantBits() > 63)
        return false;
    }
    return true;
  }

  /// Whether `mul` cannot overflow as a signed multiplication.
  bool isExactMul(arith::MulIOp mul) const {
    if (bitEnumContainsAll(mul.getOverflowFlags(),
                           arith::IntegerOverflowFlags::nsw))
      return true;
    std::optional<ConstantIntRanges> lhs = getRange(mul.getLhs());
    std::optional<ConstantIntRanges> rhs = getRange(mul.getRhs());
    if (!lhs || !rhs)
      return false;
    for (const APInt &a : {lhs->smin(), lhs->smax()}) {
      for (const APInt &b : {rhs->smin(), rhs->smax()}) {
        bool overflow = false;
        (void)a.smul_ov(b, overflow);
        if (overflow)
          return false;
      }
    }
    return true;
  }

  /// Return a value equal to `value`, looking through integer casts that
  /// preserve their (signed) operand and through `(x * c) / c`.
  Value stripExact(Value value) const {
    while (true) {
      if (Value source = getExactCastSource(value)) {
        value = source;
        continue;
      }
      Operation *div = value.getDefiningOp();
      APInt divisor;
      if (!isa_and_nonnull<arith::DivSIOp, arith::DivUIOp>(div) ||
          !matchPattern(div->getOperand(1), m_ConstantInt(&divisor)) ||
          divisor.isZero())
        return value;
      auto mul = div->getOperand(0).getDefiningOp<arith::MulIOp>();
      if (!mul || !isExactMul(mul))
        return value;
      Value factor = getFactorOf(mul, divisor);
      // Unsigned division only undoes the product of non-negative factors.
      if (!factor || (isa<arith::DivUIOp>(div) && !isNonNegative(factor)))
        return value;
      value = factor;
    }
  }

  /// Whether `value` is a multiple of `factor`.
  bool isMultipleOf(Value value, int64_t factor) const {
    value = stripExact(value);
    APInt constant;
    if (matchPattern(value, m_ConstantInt(&constant)))
      return constant.getSExtValue() % factor == 0;
    auto mul = value.getDefiningOp<arith::MulIOp>();
    return mul && isExactMul(mul) &&
           (isMultipleOf(mul.getLhs(), factor) ||
            isMultipleOf(mul.getRhs(), factor));
  }

private:
  /// The operand of the cast defining `value` if the cast preserves it.
  Value getExactCastSource(Value value) const {
    Operation *cast = value.getDefiningOp();
    if (!isa_and_nonnull<arith::IndexCastOp, arith::IndexCastUIOp,
                         arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(cast))
      return {};
    Value source = cast->getOperand(0);
    std::optional<ConstantIntRanges> range = getRange(source);
    if (!range)
      return {};
    unsigned width = ConstantIntRanges::getStorageBitwidth(value.getType());
    if (width > ConstantIntRanges::getStorageBitwidth(source.getType())) {
      // Zero extension changes negative values, sign extension none.
      bool zeroExtends = isa<arith::IndexCastUIOp, arith::ExtUIOp>(cast);
      return zeroExtends && range->smin().isNegative() ? Value() : source;
    }
    // Truncation preserves exactly the values that fit the result type.
    if (range->smin().getSignificantBits() > width ||
        range->smax().getSignificantBits() > width)
      return {};
    return source;
  }

  /// The operand of `mul` that is multiplied by the constant `factor`.
  static Value getFactorOf(arith::MulIOp mul, const APInt &factor) {
    APInt constant;
    if (matchPattern(mul.getRhs(), m_ConstantInt(&constant)) &&
        constant == factor)
      return mul.getLhs();
    if (matchPattern(mul.getLhs(), m_ConstantInt(&constant)) &&
        constant == factor)
      return mul.getRhs();
    return {};
  }

  DataFlowSolver solver;
  bool valid = false;
};

} // namespace

/// The loop whose `index` induction variable `value` is, if any. Unsigned loops
/// are skipped because the range analysis treats loop bounds as signed.
static scf::ForOp getInductionLoop(Value value) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg || !arg.getType().isIndex())
    return {};
  auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
  if (!loop || loop.getInductionVar() != arg || loop.getUnsignedCmp())
    return {};
  return loop;
}

/// Erase `op` and the operations it kept alive, if it is dead.
static void eraseDeadChain(Operation *op) {
  llvm::SetVector<Operation *> worklist;
  worklist.insert(op);
  while (!worklist.empty()) {
    Operation *candidate = worklist.pop_back_val();
    if (!isOpTriviallyDead(candidate))
      continue;
    for (Value operand : candidate->getOperands())
      if (Operation *def = operand.getDefiningOp())
        worklist.insert(def);
    candidate->erase();
  }
}

namespace {

/// A loop whose induction variable indexes transfers as `iv * tileSize`.
struct TileLoop {
  /// Zero once the transfers disagree on the tile size.
  int64_t tileSize = 0;
  /// The `index` multiplications that compute `iv * tileSize`.
  SmallVector<Operation *> elementIndices;
};

} // namespace

/// The scalar constant splatted across `value`, or null. Broadcasts are looked
/// through only from single-element sources, whose lanes hold one value.
static Attribute getSplatConstant(Value value) {
  while (auto broadcast = value.getDefiningOp<vector::BroadcastOp>()) {
    auto sourceTy = dyn_cast<VectorType>(broadcast.getSource().getType());
    if (sourceTy && (sourceTy.isScalable() || sourceTy.getNumElements() != 1))
      break;
    value = broadcast.getSource();
  }
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return {};
  if (auto elements = dyn_cast<SplatElementsAttr>(attr))
    return elements.getSplatValue<Attribute>();
  return isa<VectorType>(value.getType()) ? Attribute() : attr;
}

namespace {

/// Fold `select(mask, transfer_read(.., pad), splat(pad))` into a masked
/// transfer_read. Both yield `pad` on masked-off lanes, but the masked read
/// lets the backend skip their loads. The in_bounds flags are kept, so lanes
/// the mask leaves enabled keep their out-of-bounds protection.
struct FoldSelectIntoTransferReadMask
    : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    auto read = op.getTrueValue().getDefiningOp<vector::TransferReadOp>();
    if (!read || !read->hasOneUse() || read.getMask())
      return rewriter.notifyMatchFailure(op, "no unmasked read to fold into");
    // The mask is indexed in source (pre-permutation) order.
    if (!read.getPermutationMap().isMinorIdentity())
      return rewriter.notifyMatchFailure(op, "read permutes its source");
    // A scalar condition does not select per lane.
    if (!isa<VectorType>(op.getCondition().getType()))
      return rewriter.notifyMatchFailure(op, "condition is not per lane");
    // The mask must be available at the read, which must not move.
    if (!DominanceInfo().properlyDominates(op.getCondition(), read))
      return rewriter.notifyMatchFailure(op, "condition defined after read");
    Attribute padding;
    if (!matchPattern(read.getPadding(), m_Constant(&padding)) ||
        getSplatConstant(op.getFalseValue()) != padding)
      return rewriter.notifyMatchFailure(op, "false value is not the padding");

    rewriter.modifyOpInPlace(
        read, [&] { read.getMaskMutable().assign(op.getCondition()); });
    rewriter.replaceOp(op, read.getResult());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Row-wise loads
//
// A gather is a sequence of contiguous row loads when, within every row, its
// offsets grow by exactly one from column to column. `hasUnitMinorStride`
// proves that structurally: it walks down to the `vector.step` that supplies
// the iota, allowing replication across leading dimensions and minor-invariant
// shifts (row bases, strides, buffer offsets) at any level. That covers the
// canonical `row_base + column_iota` form as well as `broadcast(start + iota)`.
//
// A cast may change the lane-to-lane stride when arithmetic in its narrower
// source type wraps. Casts of an unshifted step whose lanes fit are exact. For
// one widening cast of shifted lanes, the rows are only contiguous if no row
// wraps; that is checked at runtime, with a gather as fallback.
//===----------------------------------------------------------------------===//

/// Whether `ty` spans `minorSize` elements in the minor dimension and one in
/// all others. Comparing element counts alone would also accept e.g.
/// `vector<Nx1>`, whose elements lie along a column.
static bool isMinorOnlyShape(VectorType ty, int64_t minorSize) {
  return ty && ty.getRank() > 0 && ty.getShape().back() == minorSize &&
         ty.getNumElements() == minorSize;
}

/// Match a value that is constant along the minor dimension, so adding it
/// shifts a whole row without disturbing its unit spacing.
static bool isMinorInvariant(Value value) {
  auto valueTy = dyn_cast<VectorType>(value.getType());
  if (!valueTy)
    return true;
  DenseElementsAttr elements;
  if (matchPattern(value, m_Constant(&elements)) && elements.isSplat())
    return true;
  auto broadcast = value.getDefiningOp<vector::BroadcastOp>();
  if (!broadcast)
    return false;
  // `vector.broadcast` aligns trailing dimensions, so a source whose minor
  // extent is 1 (or a scalar source) is stretched uniformly across columns.
  auto sourceTy = dyn_cast<VectorType>(broadcast.getSource().getType());
  return !sourceTy ||
         (sourceTy.getRank() > 0 && sourceTy.getShape().back() == 1);
}

namespace {
/// Where `matchUnitMinorStride` is in the chain of offset computations.
enum class StrideMode {
  /// Above all casts, in the gather's index type.
  Wide,
  /// Below the cast checked at runtime, in its narrower source type.
  Narrow,
  /// Below a cast that must preserve the lanes exactly, so shifts must not wrap.
  Exact,
};
} // namespace

/// Prove that offsets within each row increase by exactly one per column.
/// Returns the cast whose source rows must be checked at runtime not to wrap,
/// or null if no check is needed. In `Exact` mode, shifts carrying `noWrap`,
/// the flags under which the enclosing cast preserves them, are allowed.
static FailureOr<Operation *> matchUnitMinorStride(
    Value value, int64_t minorSize, StrideMode mode = StrideMode::Wide,
    arith::IntegerOverflowFlags noWrap = arith::IntegerOverflowFlags::none) {
  Operation *def = value.getDefiningOp();
  if (isa_and_nonnull<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp,
                     arith::ExtUIOp>(def)) {
    Value source = def->getOperand(0);
    unsigned sourceWidth =
        ConstantIntRanges::getStorageBitwidth(source.getType());
    unsigned resultWidth =
        ConstantIntRanges::getStorageBitwidth(value.getType());
    bool signedCast = isa<arith::IndexCastOp, arith::ExtSIOp>(def);
    unsigned requiredBits =
        APInt(64, minorSize - 1).getActiveBits() + unsigned(signedCast);
    if (sourceWidth < requiredBits || resultWidth < requiredBits)
      return failure();
    auto castNoWrap = signedCast ? arith::IntegerOverflowFlags::nsw
                                 : arith::IntegerOverflowFlags::nuw;
    if (succeeded(matchUnitMinorStride(source, minorSize, StrideMode::Exact,
                                       castNoWrap)))
      return static_cast<Operation *>(nullptr);
    if (mode != StrideMode::Wide || sourceWidth > resultWidth ||
        failed(matchUnitMinorStride(source, minorSize, StrideMode::Narrow)))
      return failure();
    return def;
  }
  // Base case: the iota itself, which must span exactly the minor dimension.
  if (auto step = value.getDefiningOp<vector::StepOp>()) {
    if (!isMinorOnlyShape(step.getType(), minorSize))
      return failure();
    return static_cast<Operation *>(nullptr);
  }
  // Replication preserves the step only when the source already carries it
  // along its own minor dimension; the shape guard rejects transposed sources.
  if (auto shapeCast = value.getDefiningOp<vector::ShapeCastOp>()) {
    if (!isMinorOnlyShape(shapeCast.getSourceVectorType(), minorSize))
      return failure();
    return matchUnitMinorStride(shapeCast.getSource(), minorSize, mode,
                                noWrap);
  }
  if (auto broadcast = value.getDefiningOp<vector::BroadcastOp>()) {
    // A scalar source cannot carry a step, so a vector source is required.
    auto sourceTy = dyn_cast<VectorType>(broadcast.getSource().getType());
    if (!isMinorOnlyShape(sourceTy, minorSize))
      return failure();
    return matchUnitMinorStride(broadcast.getSource(), minorSize, mode,
                                noWrap);
  }
  // Shifting by a minor-invariant term moves a row without restriding it. The
  // shift may sit either side of a replication, e.g. `broadcast(start + iota)`.
  if (auto add = value.getDefiningOp<arith::AddIOp>();
      add && (mode != StrideMode::Exact ||
              (noWrap != arith::IntegerOverflowFlags::none &&
               bitEnumContainsAll(add.getOverflowFlags(), noWrap)))) {
    for (auto [stride, shift] : {std::pair(add.getLhs(), add.getRhs()),
                                 std::pair(add.getRhs(), add.getLhs())}) {
      if (!isMinorInvariant(shift))
        continue;
      FailureOr<Operation *> check =
          matchUnitMinorStride(stride, minorSize, mode, noWrap);
      if (succeeded(check))
        return check;
    }
  }
  return failure();
}

/// Shape cast `value` to `shape`, keeping its element type.
static Value reshapeVector(PatternRewriter &rewriter, Location loc, Value value,
                           ArrayRef<int64_t> shape) {
  auto type = cast<VectorType>(value.getType());
  if (type.getShape() == shape)
    return value;
  return vector::ShapeCastOp::create(
      rewriter, loc, VectorType::get(shape, type.getElementType()), value);
}

/// Whether no row of the source of `wrapCast`, which a matched stride makes
/// `first + j` modulo its width, wraps across the `columns` lanes it spans.
static Value createNoRowWrapCheck(PatternRewriter &rewriter, Location loc,
                                  Operation *wrapCast, int64_t columns) {
  Value source = wrapCast->getOperand(0);
  auto sourceTy = cast<VectorType>(source.getType());
  int64_t rows = sourceTy.getNumElements() / columns;
  Value firsts = vector::ExtractStridedSliceOp::create(
      rewriter, loc, reshapeVector(rewriter, loc, source, {rows, columns}),
      /*offsets=*/ArrayRef<int64_t>{0, 0}, /*sizes=*/ArrayRef<int64_t>{rows, 1},
      /*strides=*/ArrayRef<int64_t>{1, 1});
  firsts = reshapeVector(rewriter, loc, firsts, rows);

  bool signedCast = isa<arith::IndexCastOp, arith::ExtSIOp>(wrapCast);
  unsigned width = sourceTy.getElementTypeBitWidth();
  APInt max = signedCast ? APInt::getSignedMaxValue(width)
                         : APInt::getMaxValue(width);
  APInt limit = max - (columns - 1);
  Value bound = arith::ConstantOp::create(
      rewriter, loc,
      DenseElementsAttr::get(cast<VectorType>(firsts.getType()), limit));
  Value fits = arith::CmpIOp::create(
      rewriter, loc,
      signedCast ? arith::CmpIPredicate::sle : arith::CmpIPredicate::ule,
      firsts, bound);
  return vector::ReductionOp::create(rewriter, loc, vector::CombiningKind::AND,
                                     fits);
}

/// The result of `gather`, loaded by one masked load per row.
static Value createRowLoads(PatternRewriter &rewriter,
                            vector::GatherOp gather) {
  VectorType resultTy = gather.getVectorType();
  Location loc = gather.getLoc();
  int64_t columns = resultTy.getShape().back();
  int64_t rowsAndColumns[] = {resultTy.getNumElements() / columns, columns};
  Value indices =
      reshapeVector(rewriter, loc, gather.getIndices(), rowsAndColumns);
  Value mask = reshapeVector(rewriter, loc, gather.getMask(), rowsAndColumns);
  Value passThru =
      reshapeVector(rewriter, loc, gather.getPassThru(), rowsAndColumns);

  auto rowTy = VectorType::get({columns}, resultTy.getElementType());
  SmallVector<Value> offsets(gather.getOffsets());
  Value minorOffset = offsets.back();
  Value result = passThru;
  for (int64_t row = 0; row < rowsAndColumns[0]; ++row) {
    Value first = vector::ExtractOp::create(rewriter, loc, indices,
                                            ArrayRef<int64_t>{row, 0});
    offsets.back() =
        rewriter.createOrFold<arith::AddIOp>(loc, first, minorOffset);
    Value rowMask = vector::ExtractOp::create(rewriter, loc, mask, row);
    Value rowPassThru = vector::ExtractOp::create(rewriter, loc, passThru, row);
    // The gather's alignment holds only for the lanes it accesses, which
    // need not include the first lane of the row.
    auto load = vector::MaskedLoadOp::create(
        rewriter, loc, rowTy, gather.getBase(), offsets, rowMask, rowPassThru,
        llvm::MaybeAlign());
    load->setDiscardableAttrs(gather->getDiscardableAttrDictionary());
    result =
        vector::InsertOp::create(rewriter, loc, load.getResult(), result, row);
  }
  return reshapeVector(rewriter, loc, result, resultTy.getShape());
}

/// The result of `gather`, loaded by an equivalent 1-D gather.
static Value createFlatGather(PatternRewriter &rewriter,
                              vector::GatherOp gather) {
  VectorType resultTy = gather.getVectorType();
  Location loc = gather.getLoc();
  int64_t numElements = resultTy.getNumElements();
  Value indices =
      reshapeVector(rewriter, loc, gather.getIndices(), numElements);
  Value mask = reshapeVector(rewriter, loc, gather.getMask(), numElements);
  Value passThru =
      reshapeVector(rewriter, loc, gather.getPassThru(), numElements);
  auto flat = vector::GatherOp::create(
      rewriter, loc, passThru.getType(), gather.getBase(), gather.getOffsets(),
      indices, mask, passThru, gather.getAlignmentAttr());
  flat->setDiscardableAttrs(gather->getDiscardableAttrDictionary());
  return reshapeVector(rewriter, loc, flat.getResult(), resultTy.getShape());
}

namespace {

/// Replace a gather whose offsets are contiguous within each row by one masked
/// load per row. Leading dimensions are collapsed, so a rank-N gather becomes
/// `rows x columns`. Each row loads from its first offset: lane `j` then reads
/// the element the gather would, and masked-off lanes are never accessed, so a
/// row whose first offset lies outside the buffer stays safe. Rows that are
/// contiguous only without wraparound are guarded by a runtime check.
struct LowerGatherToRowLoads : public OpRewritePattern<vector::GatherOp> {
  LowerGatherToRowLoads(MLIRContext *context)
      : OpRewritePattern(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(vector::GatherOp gather,
                                PatternRewriter &rewriter) const override {
    VectorType resultTy = gather.getVectorType();
    auto baseTy = dyn_cast<MemRefType>(gather.getBaseType());
    if (resultTy.getRank() < 2 || resultTy.isScalable() || !baseTy ||
        !baseTy.isLastDimUnitStride() ||
        !gather.getIndexVectorType().getElementType().isIndex())
      return rewriter.notifyMatchFailure(gather, "unsupported gather");
    int64_t columns = resultTy.getShape().back();
    FailureOr<Operation *> wrapCast =
        matchUnitMinorStride(gather.getIndices(), columns);
    if (failed(wrapCast))
      return rewriter.notifyMatchFailure(gather, "rows are not contiguous");

    if (!*wrapCast) {
      rewriter.replaceOp(gather, createRowLoads(rewriter, gather));
      return success();
    }

    Location loc = gather.getLoc();
    Value noWrap = createNoRowWrapCheck(rewriter, loc, *wrapCast, columns);
    auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{resultTy}, noWrap,
                                  /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      scf::YieldOp::create(rewriter, loc, createRowLoads(rewriter, gather));
      // The 1-D fallback is not matched again by this pattern.
      rewriter.setInsertionPointToStart(ifOp.elseBlock());
      scf::YieldOp::create(rewriter, loc, createFlatGather(rewriter, gather));
    }
    rewriter.replaceOp(gather, ifOp.getResults());
    return success();
  }
};

/// Flatten an n-D gather to 1-D, the only form the LLVM lowering supports.
struct FlattenGather : public OpRewritePattern<vector::GatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::GatherOp gather,
                                PatternRewriter &rewriter) const override {
    VectorType resultTy = gather.getVectorType();
    if (resultTy.getRank() < 2 || resultTy.isScalable())
      return rewriter.notifyMatchFailure(gather, "not an n-D gather");
    rewriter.replaceOp(gather, createFlatGather(rewriter, gather));
    return success();
  }
};

/// Flatten an n-D scatter to 1-D, the only form the LLVM lowering supports.
struct FlattenScatter : public OpRewritePattern<vector::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ScatterOp scatter,
                                PatternRewriter &rewriter) const override {
    VectorType valueTy = scatter.getVectorType();
    if (valueTy.getRank() < 2 || valueTy.isScalable())
      return rewriter.notifyMatchFailure(scatter, "not an n-D scatter");

    Location loc = scatter.getLoc();
    int64_t numElements = valueTy.getNumElements();
    Value indices =
        reshapeVector(rewriter, loc, scatter.getIndices(), numElements);
    Value mask = reshapeVector(rewriter, loc, scatter.getMask(), numElements);
    Value value =
        reshapeVector(rewriter, loc, scatter.getValueToStore(), numElements);
    Type resultTy =
        scatter.getResult() ? scatter.getResult().getType() : Type();
    auto flat = vector::ScatterOp::create(
        rewriter, loc, resultTy, scatter.getBase(), scatter.getOffsets(),
        indices, mask, value, scatter.getAlignmentAttr());
    flat->setDiscardableAttrs(scatter->getDiscardableAttrDictionary());
    rewriter.replaceOp(scatter, flat->getResults());
    return success();
  }
};

} // namespace

void mlir::tileir::populatePostConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<FoldSelectIntoTransferReadMask, LowerGatherToRowLoads,
               FlattenGather, FlattenScatter>(patterns.getContext());
}

void mlir::tileir::rescaleTileLoops(Operation *root) {
  IntegerFacts facts(root);
  if (!facts.isValid())
    return;

  llvm::MapVector<Operation *, TileLoop> tileLoops;
  // Multiplications equal to the induction variable of an element-space loop.
  SmallVector<std::pair<Operation *, Value>> elementIvs;
  llvm::SmallPtrSet<Operation *, 16> visited;
  root->walk([&](VectorTransferOpInterface transfer) {
    for (Value index : transfer.getIndices()) {
      auto mul = index.getDefiningOp<arith::MulIOp>();
      if (!mul || !mul.getType().isIndex() || !visited.insert(mul).second)
        continue;
      APInt scale;
      Value scaled = mul.getLhs();
      if (!matchPattern(mul.getRhs(), m_ConstantInt(&scale))) {
        scaled = mul.getRhs();
        if (!matchPattern(mul.getLhs(), m_ConstantInt(&scale)))
          continue;
      }
      int64_t tileSize = scale.getSExtValue();
      if (tileSize <= 1)
        continue;

      Value base = facts.stripExact(scaled);
      if (scf::ForOp loop = getInductionLoop(base)) {
        TileLoop &tileLoop = tileLoops[loop];
        if (tileLoop.elementIndices.empty())
          tileLoop.tileSize = tileSize;
        else if (tileLoop.tileSize != tileSize)
          tileLoop.tileSize = 0;
        tileLoop.elementIndices.push_back(mul);
        continue;
      }

      // `(x / tileSize) * tileSize` with `x` an induction variable that is a
      // multiple of `tileSize` is `x`.
      Operation *div = base.getDefiningOp();
      APInt divisor;
      if (!isa_and_nonnull<arith::DivSIOp, arith::DivUIOp>(div) ||
          !matchPattern(div->getOperand(1), m_ConstantInt(&divisor)) ||
          divisor.getSExtValue() != tileSize)
        continue;
      Value dividend = div->getOperand(0);
      scf::ForOp loop = getInductionLoop(facts.stripExact(dividend));
      if (!loop || !facts.isMultipleOf(loop.getLowerBound(), tileSize) ||
          !facts.isMultipleOf(loop.getStep(), tileSize) ||
          (isa<arith::DivUIOp>(div) && !facts.isNonNegative(dividend)))
        continue;
      elementIvs.emplace_back(mul, loop.getInductionVar());
    }
  });

  for (auto [mul, iv] : elementIvs) {
    mul->getResult(0).replaceAllUsesWith(iv);
    eraseDeadChain(mul);
  }

  for (auto &[loopOp, tileLoop] : tileLoops) {
    auto loop = cast<scf::ForOp>(loopOp);
    int64_t tileSize = tileLoop.tileSize;
    if (!tileSize || !facts.fitsScaled(loop.getLowerBound(), tileSize) ||
        !facts.fitsScaled(loop.getUpperBound(), tileSize) ||
        !facts.fitsScaled(loop.getStep(), tileSize))
      continue;

    OpBuilder builder(loop);
    Location loc = loop.getLoc();
    Value scale = arith::ConstantIndexOp::create(builder, loc, tileSize);
    auto scaled = [&](Value value) -> Value {
      return arith::MulIOp::create(builder, loc, value, scale);
    };
    loop.getLowerBoundMutable().assign(scaled(loop.getLowerBound()));
    loop.getUpperBoundMutable().assign(scaled(loop.getUpperBound()));
    loop.getStepMutable().assign(scaled(loop.getStep()));

    // After scaling, the induction variable is an element index; its other
    // uses divide it by the tile size to recover the tile index.
    Value iv = loop.getInductionVar();
    builder.setInsertionPointToStart(loop.getBody());
    Value tileIv = arith::DivSIOp::create(builder, loc, iv, scale);
    iv.replaceAllUsesExcept(tileIv, tileIv.getDefiningOp());
    for (Operation *mul : tileLoop.elementIndices) {
      mul->getResult(0).replaceAllUsesWith(iv);
      eraseDeadChain(mul);
    }
  }
}

void mlir::tileir::scopeLoopAllocations(Operation *root) {
  llvm::SetVector<scf::ForOp> loops;
  root->walk([&](memref::AllocaOp alloca) {
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
