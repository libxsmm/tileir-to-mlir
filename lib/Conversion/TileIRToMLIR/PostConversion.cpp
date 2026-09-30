//===- PostConversion.cpp - Rewrites after the Tile IR conversion ---------===//
//
// Part of the tileir-to-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PostConversion.h"

#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

/// A loop whose induction variable indexes transfers as `iv * tileSize`.
struct TileLoop {
  /// Zero once the transfers disagree on the tile size.
  int64_t tileSize = 0;
  /// The `index` multiplications that compute `iv * tileSize`.
  SmallVector<Operation *> elementIndices;
};

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

void mlir::tileir::populatePostConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<FoldSelectIntoTransferReadMask>(patterns.getContext());
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

    // The induction variable now iterates over element indices; the body
    // divides it by the tile size to recover the tile index.
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
