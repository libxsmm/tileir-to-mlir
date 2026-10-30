// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Conversion/TileIRToMLIR/PackCPUKernelArguments.h"

#include "ArgPromotionUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/BitVector.h"

namespace mlir {
#define GEN_PASS_DEF_PACKCPUKERNELARGUMENTSPASS
#include "mlir/Conversion/TileIRToMLIR/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

bool isSlotType(Type type) {
  if (auto pointer = dyn_cast<LLVM::LLVMPointerType>(type))
    return pointer.getAddressSpace() == 0;
  if (auto integer = dyn_cast<IntegerType>(type)) {
    unsigned width = integer.getWidth();
    return width == 1 || width == 8 || width == 16 || width == 32 ||
           width == 64;
  }
  return isa<Float32Type, Float64Type>(type);
}

struct PackCPUKernelArgumentsPass
    : impl::PackCPUKernelArgumentsPassBase<PackCPUKernelArgumentsPass> {
  void runOnOperation() override {
    for (auto function : getOperation().getOps<func::FuncOp>()) {
      if (function.isDeclaration() || function.isPrivate() ||
          function->hasAttr("tileir.cpu_arg_count"))
        continue;
      unsigned count = function.getNumArguments();
      if (function.getNumResults() != 0 || count < 6 ||
          !llvm::all_of(
              function.getArgumentTypes().take_back(6),
              [](Type type) { return type.isInteger(32); })) {
        function.emitError(
            "expected a void CPU kernel with six trailing i32 grid arguments");
        return signalPassFailure();
      }
      if (!tileir::signatureChangeIsSafe(function)) {
        function.emitError(
            "cannot pack a CPU kernel with in-module symbol uses");
        return signalPassFailure();
      }
      unsigned packedCount = count - 6;
      if (!llvm::all_of(function.getArgumentTypes().take_front(packedCount),
                        isSlotType)) {
        function.emitError("CPU packed arguments must be host pointers, "
                           "i1/i8/i16/i32/i64, f32, or f64");
        return signalPassFailure();
      }

      Location loc = function.getLoc();
      auto pointer = LLVM::LLVMPointerType::get(&getContext());
      if (failed(function.insertArgument(0, pointer, DictionaryAttr{}, loc))) {
        function.emitError("failed to insert CPU argument buffer");
        return signalPassFailure();
      }
      OpBuilder builder = OpBuilder::atBlockBegin(&function.front());
      Value buffer = function.getArgument(0);
      for (unsigned index = 0; index < packedCount; ++index) {
        BlockArgument argument = function.getArgument(index + 1);
        Value address = LLVM::GEPOp::create(
            builder, loc, pointer, builder.getI64Type(), buffer,
            ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(index)});
        Value value = LLVM::LoadOp::create(builder, loc, argument.getType(),
                                           address, /*alignment=*/1);
        argument.replaceAllUsesWith(value);
      }
      llvm::BitVector erased(count + 1);
      erased.set(1, packedCount + 1);
      if (failed(function.eraseArguments(erased))) {
        function.emitError("failed to erase packed CPU arguments");
        return signalPassFailure();
      }
      function->setAttr("tileir.cpu_arg_count",
                        builder.getI64IntegerAttr(packedCount));
    }
  }
};

} // namespace
