//===- ConvertMemrefArgsToPtrArgs.cpp - memref->ptr args -----------------===//
//
// Part of the tileir-to-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Promotes unranked memref arguments of functions to `!llvm.ptr`.
//
// --convert-tileir-to-mlir passes pointers as `memref<*xT>` and casts them to
// ranked memrefs where it accesses them. An argument that only such casts use
// is just a pointer: the pass makes it an `!llvm.ptr` and replaces each cast by
// a memref descriptor built from the pointer and the offset, sizes and strides
// of the cast, in the layout of the memref-to-LLVM lowering:
//
//   !llvm.struct<(ptr, ptr, i64, array<R x i64>, array<R x i64>)>
//            allocated^  ^aligned  ^offset  ^sizes        ^strides
//
// The pointer stays the base buffer and the offset stays in the descriptor, so
// chained reinterpret_casts, whose offsets are absolute to the buffer, and
// extract_strided_metadata observe the same buffer and offset as before.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TileIRToMLIR/Passes.h"

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

/// Collect the casts of `arg` to ranked memrefs into `casts`, if `arg` is an
/// unranked memref that nothing else uses and the casts can be rebuilt from a
/// pointer. The casts may differ in rank and layout, as each one is replaced
/// separately.
static bool collectPromotableCasts(
    BlockArgument arg, const LLVMTypeConverter &typeConverter,
    SmallVectorImpl<std::pair<Operation *, MemRefType>> &casts) {
  auto unranked = dyn_cast<UnrankedMemRefType>(arg.getType());
  if (!unranked)
    return false;

  for (Operation *user : arg.getUsers()) {
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
    // Guaranteed by the op verifiers; checked defensively.
    if (source != arg || !resTy ||
        resTy.getElementType() != unranked.getElementType())
      return false;
    // The descriptor needs strides and a type of the memref-to-LLVM lowering. A
    // plain cast would read sizes and strides from the unranked descriptor,
    // which a pointer does not have, so they must be static.
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(resTy.getStridesAndOffset(strides, offset)) ||
        (isa<memref::CastOp>(user) &&
         (llvm::any_of(resTy.getShape(), ShapedType::isDynamic) ||
          llvm::any_of(strides, ShapedType::isDynamic))) ||
        !typeConverter.convertType(resTy))
      return false;
    casts.emplace_back(user, resTy);
  }
  return true;
}

/// `ofr` as a value of the descriptor index type `indexTy`.
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

/// The offset, sizes and strides, as `indexTy` values, of the descriptor that
/// replaces `cast`, whose result type is `ranked`.
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

  // A plain cast takes the static sizes and strides of its type. A dynamic
  // offset is 0 by the calling convention of pointer arguments.
  auto constIndex = [&](int64_t v) -> Value {
    return LLVM::ConstantOp::create(
        builder, loc, indexTy,
        builder.getIntegerAttr(indexTy, ShapedType::isDynamic(v) ? 0 : v));
  };
  auto [strideVals, offsetVal] = ranked.getStridesAndOffset();
  offset = constIndex(offsetVal);
  for (int64_t size : ranked.getShape())
    sizes.push_back(constIndex(size));
  for (int64_t stride : strideVals)
    strides.push_back(constIndex(stride));
}

/// Promote the eligible arguments of `func` to `!llvm.ptr`. Returns whether the
/// signature changed.
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
  // Build the descriptors of the memref-to-LLVM lowering, so that the casts
  // back to memrefs cancel out when it runs.
  LLVMTypeConverter typeConverter(func.getContext());
  Type indexTy = typeConverter.getIndexType();

  for (unsigned i = 0, e = func.getNumArguments(); i < e; ++i) {
    BlockArgument arg = func.getArgument(i);
    SmallVector<std::pair<Operation *, MemRefType>> casts;
    if (!collectPromotableCasts(arg, typeConverter, casts))
      continue;

    arg.setType(ptrTy);
    argTypes[i] = ptrTy;

    for (auto [cast, ranked] : casts) {
      // Build the descriptor at the cast, where its dynamic operands exist.
      OpBuilder builder(cast);
      Location loc = cast->getLoc();

      Value offset;
      SmallVector<Value> sizes, strides;
      getLayout(builder, loc, cast, ranked, indexTy, offset, sizes, strides);

      SmallVector<Value> values;
      values.reserve(3 + 2 * ranked.getRank());
      values.push_back(arg);    // allocated pointer
      values.push_back(arg);    // aligned pointer
      values.push_back(offset); // offset into the buffer
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
