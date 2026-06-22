//===- ConvertMemrefArgsToPtrArgs.cpp - memref->ptr args -----------------===//
//
// Promotes unranked-memref function arguments to opaque `!llvm.ptr` inputs.
//
// The CudaTile-to-MLIR lowering models pointer-typed kernel inputs as unranked
// `memref<*xT>` and recovers their rank/layout inside the body with a
// `memref.cast` or `memref.reinterpret_cast`. When every use of such an
// argument is one and the same cast, the ranked result type of that cast is the
// argument's true type. This pass rewrites the signature to take a bare
// `!llvm.ptr` and, in place of each redundant cast, builds a standard LLVM
// memref descriptor struct from that pointer -- folding the cast's offset into
// the pointer (a `getelementptr`) and reusing its sizes / strides -- then casts
// the descriptor back to the ranked memref. Baking the offset into the pointer
// (rather than into the descriptor's offset field) is address-equivalent but
// survives a downstream `reinterpret_cast` that resets the offset field to 0.
// The descriptor is laid out exactly as the memref-to-LLVM lowering expects:
//
//   !llvm.struct<(ptr, ptr, i64, array<R x i64>, array<R x i64>)>
//            allocated^  ^aligned  ^offset  ^sizes        ^strides
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/CudaTileToMLIR/ConvertMemrefArgsToPtrArgs.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMEMREFARGSTOPTRARGSPASS
#include "mlir/Conversion/CudaTileToMLIR/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Returns `true` iff changing `func`'s signature is safe, i.e. the function is
/// not referenced (called / launched) from within its nearest symbol table.
/// In-module references would be invalidated by a signature change, so such
/// functions are left untouched. A non-`SymbolOpInterface` or an unresolved
/// use set is treated conservatively as "unsafe".
static bool signatureChangeIsSafe(FunctionOpInterface func) {
  auto symbol = dyn_cast<SymbolOpInterface>(func.getOperation());
  if (!symbol)
    return false;
  Operation *symbolTableOp = func->getParentWithTrait<OpTrait::SymbolTable>();
  if (!symbolTableOp)
    return false;
  std::optional<SymbolTable::UseRange> uses =
      SymbolTable::getSymbolUses(func.getOperation(), symbolTableOp);
  return uses && uses->empty();
}

/// If `arg` is an unranked-memref argument whose every use is the same
/// `memref.cast` or `memref.reinterpret_cast`, return that common ranked result
/// type and collect the casts into `casts`. Otherwise return null.
static MemRefType detectRankedType(BlockArgument arg,
                                   SmallVectorImpl<Operation *> &casts) {
  auto unranked = dyn_cast<UnrankedMemRefType>(arg.getType());
  if (!unranked || arg.use_empty())
    return nullptr;

  MemRefType ranked;
  Operation *first = nullptr;
  for (Operation *user : arg.getUsers()) {
    // Only identical `memref.cast` / `memref.reinterpret_cast` of this exact
    // argument are collapsible; any other user means the unranked type is
    // observed elsewhere.
    Value source;
    MemRefType resTy;
    if (auto c = dyn_cast<memref::CastOp>(user)) {
      source = c.getSource();
      resTy = dyn_cast<MemRefType>(c.getType());
    } else if (auto rc = dyn_cast<memref::ReinterpretCastOp>(user)) {
      source = rc.getSource();
      resTy = dyn_cast<MemRefType>(rc.getType());
    } else {
      return nullptr;
    }
    if (source != arg || !resTy)
      return nullptr;
    if (!first) {
      first = user;
      ranked = resTy;
    } else if (user->getName() != first->getName() || resTy != ranked ||
               !llvm::equal(user->getOperands(), first->getOperands())) {
      // Divergent reinterpretations of the same argument are ambiguous.
      return nullptr;
    }
    casts.push_back(user);
  }

  // The cast preserves the element type; guard defensively anyway.
  if (!ranked || ranked.getElementType() != unranked.getElementType())
    return nullptr;
  return ranked;
}

/// Materializes `ofr` as a descriptor index-typed (`i64`) value. Static folds
/// become `llvm.mlir.constant`; dynamic `index` operands are converted with
/// `arith.index_cast`.
static Value materializeIndex(OpBuilder &builder, Location loc,
                              OpFoldResult ofr, Type indexTy) {
  if (auto attr = dyn_cast<Attribute>(ofr)) {
    int64_t v = cast<IntegerAttr>(attr).getInt();
    return LLVM::ConstantOp::create(builder, loc, indexTy,
                                    builder.getIntegerAttr(indexTy, v));
  }
  Value val = cast<Value>(ofr);
  if (val.getType() == indexTy)
    return val;
  return arith::IndexCastOp::create(builder, loc, indexTy, val);
}

/// Returns `true` if `ofr` is a statically-known zero.
static bool isStaticZero(OpFoldResult ofr) {
  auto attr = dyn_cast<Attribute>(ofr);
  return attr && cast<IntegerAttr>(attr).getInt() == 0;
}

/// Computes the descriptor layout for `cast` -- an unranked->ranked
/// `memref.cast` or `memref.reinterpret_cast`. `offset` receives the element
/// offset as an index-typed value, or null when the offset is statically zero
/// (or, for a plain cast, not recoverable). `sizes` and `strides` receive the
/// per-dimension extents.
static void getLayout(OpBuilder &builder, Location loc, Operation *cast,
                      MemRefType ranked, Type indexTy, Value &offset,
                      SmallVectorImpl<Value> &sizes,
                      SmallVectorImpl<Value> &strides) {
  if (auto rc = dyn_cast<memref::ReinterpretCastOp>(cast)) {
    OpFoldResult off = rc.getMixedOffsets()[0];
    if (!isStaticZero(off))
      offset = materializeIndex(builder, loc, off, indexTy);
    for (OpFoldResult size : rc.getMixedSizes())
      sizes.push_back(materializeIndex(builder, loc, size, indexTy));
    for (OpFoldResult stride : rc.getMixedStrides())
      strides.push_back(materializeIndex(builder, loc, stride, indexTy));
    return;
  }

  // Plain `memref.cast`: the layout is taken from the ranked result type.
  // Dynamic sizes are not recoverable once the unranked descriptor is dropped,
  // but the consumers only use the base pointer, offset and strides, so a zero
  // placeholder size is sufficient and never observed.
  auto constIndex = [&](int64_t v) -> Value {
    return LLVM::ConstantOp::create(
        builder, loc, indexTy,
        builder.getIntegerAttr(indexTy, ShapedType::isDynamic(v) ? 0 : v));
  };
  SmallVector<int64_t> strideVals;
  int64_t offsetVal;
  if (failed(ranked.getStridesAndOffset(strideVals, offsetVal))) {
    // Non-strided layout: fall back to an identity row-major interpretation.
    offsetVal = 0;
    strideVals.assign(ranked.getRank(), 1);
  }
  if (!ShapedType::isDynamic(offsetVal) && offsetVal != 0)
    offset = constIndex(offsetVal);
  for (int64_t size : ranked.getShape())
    sizes.push_back(constIndex(size));
  for (int64_t stride : strideVals)
    strides.push_back(constIndex(stride));
}

/// Promote eligible unranked-memref arguments of `func` to ranked memrefs.
/// Returns `true` if the signature changed.
static bool promoteFunctionArgs(FunctionOpInterface func) {
  // Declarations have no body to inspect.
  if (func.getFunctionBody().empty())
    return false;
  if (!signatureChangeIsSafe(func))
    return false;

  SmallVector<Type> argTypes(func.getArgumentTypes().begin(),
                             func.getArgumentTypes().end());
  bool changed = false;

  auto ptrTy = LLVM::LLVMPointerType::get(func.getContext());
  // The descriptor layout (and its index type) follows the standard
  // memref-to-LLVM lowering, so consumers can reconcile the cast later.
  LLVMTypeConverter typeConverter(func.getContext());
  Type indexTy = typeConverter.getIndexType();

  for (unsigned i = 0, e = func.getNumArguments(); i < e; ++i) {
    BlockArgument arg = func.getArgument(i);
    SmallVector<Operation *> casts;
    MemRefType ranked = detectRankedType(arg, casts);
    if (!ranked)
      continue;

    // Promote: the argument becomes an opaque `!llvm.ptr`. Each redundant cast
    // is replaced by a freshly built memref descriptor that wraps the pointer
    // with the cast's own offset / sizes / strides, then cast back to the
    // ranked memref. The descriptor is built right before the cast so the
    // (dynamic) shape operands are guaranteed to dominate it.
    arg.setType(ptrTy);
    argTypes[i] = ptrTy;

    for (Operation *cast : casts) {
      OpBuilder builder(cast);
      Location loc = cast->getLoc();

      Value offset;
      SmallVector<Value> sizes, strides;
      getLayout(builder, loc, cast, ranked, indexTy, offset, sizes, strides);

      // Fold the element offset into the base pointer (a `getelementptr`) and
      // leave the descriptor's offset field at zero. Keeping the offset in the
      // pointer rather than in the struct field is address-equivalent, but it
      // survives a downstream `reinterpret_cast` that resets the offset field
      // to 0 (which would otherwise silently discard the offset).
      Value basePtr = arg;
      if (offset)
        basePtr =
            LLVM::GEPOp::create(builder, loc, ptrTy, ranked.getElementType(),
                                arg, ArrayRef<LLVM::GEPArg>{offset});
      Value zeroOffset = LLVM::ConstantOp::create(
          builder, loc, indexTy, builder.getIntegerAttr(indexTy, 0));

      SmallVector<Value> values;
      values.reserve(3 + 2 * ranked.getRank());
      values.push_back(basePtr);    // allocated pointer
      values.push_back(basePtr);    // aligned pointer
      values.push_back(zeroOffset); // offset (folded into the pointer above)
      llvm::append_range(values, sizes);
      llvm::append_range(values, strides);

      Value descriptor =
          MemRefDescriptor::pack(builder, loc, typeConverter, ranked, values);
      Value materialized =
          UnrealizedConversionCastOp::create(builder, loc, ranked, descriptor)
              .getResult(0);
      cast->getResult(0).replaceAllUsesWith(materialized);
      cast->erase();
    }
    changed = true;
  }

  if (changed)
    func.setFunctionTypeAttr(
        TypeAttr::get(func.cloneTypeWith(argTypes, func.getResultTypes())));
  return changed;
}

struct ConvertMemrefArgsToPtrArgsPass
    : public ::mlir::impl::ConvertMemrefArgsToPtrArgsPassBase<
          ConvertMemrefArgsToPtrArgsPass> {
  using Base::Base;

  void runOnOperation() override {
    getOperation().walk(
        [](FunctionOpInterface func) { (void)promoteFunctionArgs(func); });
  }
};

} // namespace
