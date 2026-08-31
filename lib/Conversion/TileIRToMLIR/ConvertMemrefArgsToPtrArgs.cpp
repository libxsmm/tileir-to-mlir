//===- ConvertMemrefArgsToPtrArgs.cpp - memref->ptr args -----------------===//
//
// Part of the tileir-to-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Promotes unranked-memref function arguments to opaque `!llvm.ptr` inputs.
//
// The tileir-to-mlir lowering models pointer-typed kernel inputs as unranked
// `memref<*xT>` and recovers their rank/layout inside the body with a
// `memref.cast` or `memref.reinterpret_cast`. When every use of such an
// argument is one of these casts -- or when the argument has no uses -- the
// argument is really just an opaque pointer. This pass rewrites the signature
// to take a bare
// `!llvm.ptr` and, in place of each redundant cast, builds a standard LLVM
// memref descriptor struct from that pointer -- using the argument pointer as
// the descriptor's base buffer and storing the cast's offset / sizes / strides
// directly in the descriptor -- then casts the descriptor back to the ranked
// memref. Keeping the offset in the descriptor's offset field (rather than
// folding it into the base pointer) preserves the argument pointer as the base
// buffer, so a chained `memref.reinterpret_cast` -- whose offset is absolute to
// that buffer -- or a `memref.extract_strided_metadata` observes the same base
// pointer and offset as the original source. An argument may be reinterpreted
// several different ways; each cast is rebuilt independently.
// The descriptor is laid out exactly as the memref-to-LLVM lowering expects:
//
//   !llvm.struct<(ptr, ptr, i64, array<R x i64>, array<R x i64>)>
//            allocated^  ^aligned  ^offset  ^sizes        ^strides
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TileIRToMLIR/ConvertMemrefArgsToPtrArgs.h"

#include "ArgPromotionUtils.h"
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
#include "mlir/Conversion/TileIRToMLIR/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

using tileir::signatureChangeIsSafe;

/// If `arg` is an unranked-memref argument with no uses, or whose every use is
/// a `memref.cast` or `memref.reinterpret_cast` of that exact argument, collect
/// those casts together with their ranked result types into `casts` and return
/// `true`. Different casts may reinterpret the argument in incompatible ways
/// (distinct ranks, offsets or layouts); that is fine, since each cast is
/// rebuilt independently from the recovered pointer. Returns `false` (leaving
/// `casts` in an unspecified state) if any use is something other than such a
/// cast, so that the unranked descriptor is never otherwise observed.
static bool collectPromotableCasts(
    BlockArgument arg,
    SmallVectorImpl<std::pair<Operation *, MemRefType>> &casts) {
  auto unranked = dyn_cast<UnrankedMemRefType>(arg.getType());
  if (!unranked)
    return false;

  for (Operation *user : arg.getUsers()) {
    // Only a `memref.cast` / `memref.reinterpret_cast` of this exact argument
    // is collapsible; any other user means the unranked type is observed
    // elsewhere and the argument must stay as-is.
    Value source;
    MemRefType resTy;
    if (auto c = dyn_cast<memref::CastOp>(user)) {
      source = c.getSource();
      resTy = dyn_cast<MemRefType>(c.getType());
    } else if (auto rc = dyn_cast<memref::ReinterpretCastOp>(user)) {
      source = rc.getSource();
      resTy = dyn_cast<MemRefType>(rc.getType());
    } else {
      return false;
    }
    // The cast must apply to this argument and yield a ranked memref; a cast
    // never changes the element type, so guard that defensively too.
    if (source != arg || !resTy ||
        resTy.getElementType() != unranked.getElementType())
      return false;
    casts.emplace_back(user, resTy);
  }
  return true;
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

/// Computes the descriptor layout for `cast` -- an unranked->ranked
/// `memref.cast` or `memref.reinterpret_cast`. `offset` receives the element
/// offset as an index-typed value (zero when statically zero or, for a plain
/// cast, not recoverable). `sizes` and `strides` receive the per-dimension
/// extents.
static void getLayout(OpBuilder &builder, Location loc, Operation *cast,
                      MemRefType ranked, Type indexTy, Value &offset,
                      SmallVectorImpl<Value> &sizes,
                      SmallVectorImpl<Value> &strides) {
  if (auto rc = dyn_cast<memref::ReinterpretCastOp>(cast)) {
    offset = materializeIndex(builder, loc, rc.getMixedOffsets()[0], indexTy);
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
    SmallVector<std::pair<Operation *, MemRefType>> casts;
    if (!collectPromotableCasts(arg, casts))
      continue;

    // Promote: the argument becomes an opaque `!llvm.ptr`. Each redundant cast
    // is replaced by a freshly built memref descriptor that wraps the pointer
    // with the cast's own offset / sizes / strides, then cast back to the
    // cast's ranked result type. The descriptor is built right before the cast
    // so the (dynamic) shape operands are guaranteed to dominate it.
    arg.setType(ptrTy);
    argTypes[i] = ptrTy;

    for (auto [cast, ranked] : casts) {
      OpBuilder builder(cast);
      Location loc = cast->getLoc();

      Value offset;
      SmallVector<Value> sizes, strides;
      getLayout(builder, loc, cast, ranked, indexTy, offset, sizes, strides);

      // Store the element offset in the descriptor's offset field and keep the
      // argument pointer as the (allocated / aligned) base buffer. This mirrors
      // the source `memref.reinterpret_cast`, whose offset is absolute to the
      // underlying buffer: a chained reinterpret_cast or
      // extract_strided_metadata then recovers the same base pointer and
      // offset. Folding the offset into the pointer instead would move the base
      // buffer and silently change those observations.
      SmallVector<Value> values;
      values.reserve(3 + 2 * ranked.getRank());
      values.push_back(arg);    // allocated pointer
      values.push_back(arg);    // aligned pointer
      values.push_back(offset); // offset (absolute to the base buffer)
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
