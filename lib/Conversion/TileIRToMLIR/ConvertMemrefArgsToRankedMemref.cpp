//===- ConvertMemrefArgsToRankedMemref.cpp ------------------------------===//
//
// Part of the tileir-to-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Promotes unranked memref function arguments to ranked memrefs when the
// function body immediately reinterprets those arguments with a fixed ranked
// layout and forwards scalar shape/stride arguments into the cast.
//
// This pass is intended to run after --convert-tileir-to-mlir.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TileIRToMLIR/ConvertMemrefArgsToRankedMemref.h"

#include "ArgPromotionUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMEMREFARGSTORANKEDMEMREFPASS
#include "mlir/Conversion/TileIRToMLIR/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

using tileir::isStaticZero;
using tileir::signatureChangeIsSafe;

/// Returns true when `lhs` and `rhs` are identical reinterpret_casts for this
/// transform: same result type and identical mixed offset/size/stride
/// operands. Callers guarantee both share the same source argument.
static bool sameReinterpretCast(memref::ReinterpretCastOp lhs,
                                memref::ReinterpretCastOp rhs) {
  if (lhs.getType() != rhs.getType())
    return false;
  return llvm::equal(lhs.getMixedOffsets(), rhs.getMixedOffsets()) &&
         llvm::equal(lhs.getMixedSizes(), rhs.getMixedSizes()) &&
         llvm::equal(lhs.getMixedStrides(), rhs.getMixedStrides());
}

struct PtrPromotionPlan {
  unsigned argIndex = 0;
  MemRefType rankedType;
  SmallVector<memref::ReinterpretCastOp> casts;
  memref::ReinterpretCastOp canonicalCast;
};

/// If `arg` is unranked and all of its uses are identical reinterpret_casts to
/// a ranked memref with a static-zero offset, fills `plan` and returns true.
/// A non-zero offset cannot be represented by handing back the bare argument,
/// so such casts are left untouched.
static bool collectPtrPromotionPlan(BlockArgument arg, PtrPromotionPlan &plan) {
  auto unranked = dyn_cast<UnrankedMemRefType>(arg.getType());
  if (!unranked || arg.use_empty())
    return false;

  memref::ReinterpretCastOp canonical;
  MemRefType ranked;
  SmallVector<memref::ReinterpretCastOp> casts;

  for (Operation *user : arg.getUsers()) {
    auto rc = dyn_cast<memref::ReinterpretCastOp>(user);
    if (!rc || rc.getSource() != arg)
      return false;
    auto castTy = dyn_cast<MemRefType>(rc.getType());
    if (!castTy || castTy.getElementType() != unranked.getElementType())
      return false;
    SmallVector<OpFoldResult> offsets = rc.getMixedOffsets();
    if (offsets.size() != 1 || !isStaticZero(offsets[0]))
      return false;

    if (!canonical) {
      canonical = rc;
      ranked = castTy;
    } else if (!sameReinterpretCast(canonical, rc)) {
      return false;
    }
    casts.push_back(rc);
  }

  if (casts.empty())
    return false;

  plan.argIndex = arg.getArgNumber();
  plan.rankedType = ranked;
  plan.casts = std::move(casts);
  plan.canonicalCast = canonical;
  return true;
}

enum class ScalarKind {
  Dim,
  Stride,
};

struct ScalarRecipe {
  unsigned memrefArgIndex = 0;
  ScalarKind kind = ScalarKind::Dim;
  unsigned dim = 0;

  bool operator==(const ScalarRecipe &rhs) const {
    return memrefArgIndex == rhs.memrefArgIndex && kind == rhs.kind &&
           dim == rhs.dim;
  }
};

/// Returns the corresponding scalar argument index when `v` is directly a block
/// argument, or an arith.index_cast of one.
static std::optional<unsigned> getScalarArgIndex(Value v) {
  if (auto barg = dyn_cast<BlockArgument>(v))
    return barg.getArgNumber();
  if (auto cast = v.getDefiningOp<arith::IndexCastOp>()) {
    if (auto barg = dyn_cast<BlockArgument>(cast.getIn()))
      return barg.getArgNumber();
  }
  return std::nullopt;
}

/// Build a replacement index value from `recipe` using the promoted ranked
/// memref argument.
static Value buildIndexFromRecipe(OpBuilder &builder, Location loc,
                                  BlockArgument rankedArg,
                                  const ScalarRecipe &recipe) {
  if (recipe.kind == ScalarKind::Dim) {
    Value cstDim = arith::ConstantIndexOp::create(builder, loc, recipe.dim);
    return memref::DimOp::create(builder, loc, rankedArg, cstDim);
  }

  auto meta = memref::ExtractStridedMetadataOp::create(builder, loc, rankedArg);
  return meta.getStrides()[recipe.dim];
}

/// Cast `indexVal` to `dstType` if needed. Returns null when unsupported.
static Value castIndexToType(OpBuilder &builder, Location loc, Value indexVal,
                             Type dstType) {
  if (indexVal.getType() == dstType)
    return indexVal;
  if (auto intTy = dyn_cast<IntegerType>(dstType))
    return arith::IndexCastOp::create(builder, loc, intTy, indexVal);
  return {};
}

/// Erase operations that became trivially dead after cast rewrites.
static void eraseTriviallyDead(SmallVectorImpl<Operation *> &worklist) {
  llvm::SmallPtrSet<Operation *, 16> seen;
  SmallVector<Operation *> deduplicatedWorklist;
  for (Operation *op : worklist)
    if (seen.insert(op).second)
      deduplicatedWorklist.push_back(op);

  while (!deduplicatedWorklist.empty()) {
    Operation *op = deduplicatedWorklist.pop_back_val();
    if (!op || !isOpTriviallyDead(op))
      continue;
    for (Value operand : op->getOperands())
      if (Operation *def = operand.getDefiningOp())
        if (seen.insert(def).second)
          deduplicatedWorklist.push_back(def);
    op->erase();
  }
}

static bool promoteOneFunction(FunctionOpInterface func,
                               MemrefArgRemovalMode removeUnused) {
  if (func.getFunctionBody().empty() || !signatureChangeIsSafe(func))
    return false;

  SmallVector<PtrPromotionPlan> ptrPlans;
  ptrPlans.reserve(func.getNumArguments());
  for (unsigned i = 0, e = func.getNumArguments(); i < e; ++i) {
    auto arg = func.getArgument(i);
    PtrPromotionPlan plan;
    if (collectPtrPromotionPlan(arg, plan))
      ptrPlans.push_back(std::move(plan));
  }
  if (ptrPlans.empty())
    return false;

  llvm::MapVector<unsigned, ScalarRecipe> scalarRecipes;
  llvm::SmallSet<unsigned, 8> conflictingScalarArgs;

  for (PtrPromotionPlan &plan : ptrPlans) {
    memref::ReinterpretCastOp canonical = plan.canonicalCast;
    SmallVector<OpFoldResult> mixedSizes = canonical.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = canonical.getMixedStrides();

    auto registerRecipe = [&](OpFoldResult ofr, ScalarKind kind, unsigned dim) {
      auto val = dyn_cast<Value>(ofr);
      if (!val)
        return;
      std::optional<unsigned> argIdx = getScalarArgIndex(val);
      if (!argIdx)
        return;

      ScalarRecipe recipe{plan.argIndex, kind, dim};
      auto [it, inserted] = scalarRecipes.try_emplace(*argIdx, recipe);
      if (!inserted && !(it->second == recipe))
        conflictingScalarArgs.insert(*argIdx);
    };

    for (unsigned d = 0, rank = plan.rankedType.getRank(); d < rank; ++d) {
      registerRecipe(mixedSizes[d], ScalarKind::Dim, d);
      registerRecipe(mixedStrides[d], ScalarKind::Stride, d);
    }
  }

  SmallVector<Operation *> maybeDead;
  bool changed = false;

  for (const PtrPromotionPlan &plan : ptrPlans) {
    BlockArgument arg = func.getArgument(plan.argIndex);
    arg.setType(plan.rankedType);
    changed = true;

    for (memref::ReinterpretCastOp rc : plan.casts) {
      for (Value operand : rc->getOperands())
        if (Operation *def = operand.getDefiningOp())
          maybeDead.push_back(def);
      rc.getResult().replaceAllUsesWith(arg);
      rc.erase();
    }
  }

  if (changed) {
    SmallVector<Type> currentArgTypes;
    currentArgTypes.reserve(func.getNumArguments());
    for (BlockArgument arg : func.getArguments())
      currentArgTypes.push_back(arg.getType());
    func.setFunctionTypeAttr(TypeAttr::get(
        func.cloneTypeWith(currentArgTypes, func.getResultTypes())));
  }

  llvm::BitVector argsToErase(func.getNumArguments());
  llvm::BitVector memrefDependentArgs(func.getNumArguments());
  llvm::BitVector assumedMemrefDependentArgs(func.getNumArguments());

  for (const PtrPromotionPlan &plan : ptrPlans) {
    unsigned firstAssumedArg = plan.argIndex + 1;
    unsigned numAssumedArgs = 2 * plan.rankedType.getRank();
    if (firstAssumedArg > func.getNumArguments() ||
        numAssumedArgs > func.getNumArguments() - firstAssumedArg)
      continue;
    for (unsigned argIdx = firstAssumedArg;
         argIdx < firstAssumedArg + numAssumedArgs; ++argIdx)
      assumedMemrefDependentArgs.set(argIdx);
  }

  for (auto [argIdx, recipe] : scalarRecipes) {
    memrefDependentArgs.set(argIdx);
    if (conflictingScalarArgs.contains(argIdx))
      continue;

    BlockArgument scalarArg = func.getArgument(argIdx);
    Type scalarTy = scalarArg.getType();
    if (!isa<IndexType, IntegerType>(scalarTy))
      continue;

    SmallVector<OpOperand *> uses;
    for (OpOperand &use : scalarArg.getUses())
      uses.push_back(&use);

    for (OpOperand *use : uses) {
      Operation *user = use->getOwner();
      OpBuilder builder(user);
      Value replIndex =
          buildIndexFromRecipe(builder, user->getLoc(),
                               func.getArgument(recipe.memrefArgIndex), recipe);
      Value repl = castIndexToType(builder, user->getLoc(), replIndex,
                                   use->get().getType());
      if (!repl)
        break;
      use->set(repl);
    }
  }

  for (BlockArgument arg : func.getArguments()) {
    if (!arg.use_empty())
      continue;

    bool isMemrefDependent = memrefDependentArgs.test(arg.getArgNumber());
  bool isAssumedMemrefDependent =
    assumedMemrefDependentArgs.test(arg.getArgNumber());
    bool shouldErase = false;
    switch (removeUnused) {
    case MemrefArgRemovalMode::All:
      shouldErase = true;
      break;
    case MemrefArgRemovalMode::None:
      break;
    case MemrefArgRemovalMode::MemrefDependent:
      shouldErase = isMemrefDependent;
      break;
    case MemrefArgRemovalMode::AssumedMemrefDependent:
      shouldErase = isMemrefDependent || isAssumedMemrefDependent;
      break;
    case MemrefArgRemovalMode::Other:
      shouldErase = !isMemrefDependent;
      break;
    }
    if (shouldErase)
      argsToErase.set(arg.getArgNumber());
  }

  if (argsToErase.any()) {
    if (failed(func.eraseArguments(argsToErase)))
      return false;
    changed = true;
  }

  eraseTriviallyDead(maybeDead);
  return changed;
}

struct ConvertMemrefArgsToRankedMemrefPass
    : public ::mlir::impl::ConvertMemrefArgsToRankedMemrefPassBase<
          ConvertMemrefArgsToRankedMemrefPass> {
  using Base::Base;

  void runOnOperation() override {
    getOperation().walk([&](FunctionOpInterface func) {
      (void)promoteOneFunction(func, removeUnused);
    });
  }
};

} // namespace
