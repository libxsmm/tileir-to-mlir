//===- ConvertMemrefArgsToRankedMemref.cpp ------------------------------===//
//
// Part of the tileir-to-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Promotes unranked memref arguments of functions to ranked memrefs.
//
// --convert-tileir-to-mlir accesses pointer arguments through
// reinterpret_casts, whose dynamic sizes and strides are often further scalar
// arguments. If all uses of a pointer argument are the same reinterpret_cast at
// offset 0, the argument takes the type of the cast and replaces it. The scalar
// arguments that the cast uses as sizes or strides are recomputed from the
// memref with memref.dim and memref.extract_strided_metadata, and unused
// arguments are removed as the `remove-unused` option selects. Callers must
// pass the promoted argument with the sizes and strides the cast computed.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TileIRToMLIR/Passes.h"

#include "ArgPromotionUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

using tileir::signatureChangeIsSafe;

/// Whether the reinterpret_casts `lhs` and `rhs` of the same source yield the
/// same memref.
static bool sameReinterpretCast(memref::ReinterpretCastOp lhs,
                                memref::ReinterpretCastOp rhs) {
  if (lhs.getType() != rhs.getType())
    return false;
  return llvm::equal(lhs.getMixedOffsets(), rhs.getMixedOffsets()) &&
         llvm::equal(lhs.getMixedSizes(), rhs.getMixedSizes()) &&
         llvm::equal(lhs.getMixedStrides(), rhs.getMixedStrides());
}

/// The promotion of argument `argIndex` to `rankedType`, the type of its uses
/// `casts`, which all equal `canonicalCast`.
struct PtrPromotionPlan {
  unsigned argIndex = 0;
  MemRefType rankedType;
  SmallVector<memref::ReinterpretCastOp> casts;
  memref::ReinterpretCastOp canonicalCast;
};

/// Plan the promotion of `arg` if it is an unranked memref whose uses are all
/// the same reinterpret_cast at a static zero offset. The argument replaces the
/// casts, so they must not offset it.
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
    if (offsets.size() != 1 || !isZeroInteger(offsets[0]))
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

/// How to recompute a scalar argument: as the size (Dim) or stride of
/// dimension `dim` of the promoted argument `memrefArgIndex`.
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

/// Build the value of `recipe` from the promoted argument `rankedArg`.
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

/// Erase the operations of `worklist` and their operands that are dead.
static void eraseTriviallyDead(SmallVectorImpl<Operation *> &worklist) {
  llvm::SmallPtrSet<Operation *, 16> seen;
  SmallVector<Operation *> deduplicatedWorklist;
  for (Operation *op : worklist)
    if (seen.insert(op).second)
      deduplicatedWorklist.push_back(op);

  while (!deduplicatedWorklist.empty()) {
    Operation *op = deduplicatedWorklist.pop_back_val();
    if (!isOpTriviallyDead(op))
      continue;
    for (Value operand : op->getOperands())
      if (Operation *def = operand.getDefiningOp())
        if (seen.insert(def).second)
          deduplicatedWorklist.push_back(def);
    op->erase();
  }
}

/// Promote the eligible pointer arguments of `func`, recompute the scalar
/// arguments that their casts use, and remove unused arguments as
/// `removeUnused` selects. Returns whether `func` changed.
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

  // The scalar arguments that the casts use as sizes or strides. One that is
  // used as different sizes or strides is not recomputed.
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

  // The promoted arguments replace their casts.
  for (const PtrPromotionPlan &plan : ptrPlans) {
    BlockArgument arg = func.getArgument(plan.argIndex);
    arg.setType(plan.rankedType);

    for (memref::ReinterpretCastOp rc : plan.casts) {
      for (Value operand : rc->getOperands())
        if (Operation *def = operand.getDefiningOp())
          maybeDead.push_back(def);
      rc.getResult().replaceAllUsesWith(arg);
      rc.erase();
    }
  }

  // `ptrPlans` is non-empty, so at least one argument type just changed.
  SmallVector<Type> currentArgTypes;
  currentArgTypes.reserve(func.getNumArguments());
  for (BlockArgument arg : func.getArguments())
    currentArgTypes.push_back(arg.getType());
  func.setFunctionTypeAttr(TypeAttr::get(
      func.cloneTypeWith(currentArgTypes, func.getResultTypes())));

  llvm::BitVector argsToErase(func.getNumArguments());
  llvm::BitVector memrefDependentArgs(func.getNumArguments());
  // The arguments that follow a promoted rank-N memref in the calling
  // convention `(memref, size_0, ..., size_N-1, stride_0, ..., stride_N-1)`.
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

  // Recompute the scalar arguments from the promoted memrefs.
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

  if (argsToErase.any() && failed(func.eraseArguments(argsToErase)))
    return false;

  eraseTriviallyDead(maybeDead);
  return true;
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
