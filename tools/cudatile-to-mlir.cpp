//===- cudatile-to-mlir.cpp - Driver for CudaTileToMLIR pass -------------===//
//
// Simple mlir-opt-style driver that registers the CudaTileToMLIR conversion
// pass together with the dialects it depends on.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/CudaTileToMLIR/CudaTileToMLIR.h"
#include "mlir/Conversion/CudaTileToMLIR/TileIRPtrToView.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<
      mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::gpu::GPUDialect,
      mlir::memref::MemRefDialect, mlir::scf::SCFDialect, mlir::ub::UBDialect,
      mlir::vector::VectorDialect, mlir::cuda_tile::CudaTileDialect>();

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createConvertTileIRToMLIRPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createTileIRPtrToViewPass();
  });

  mlir::registerTransformsPasses();

  // If no pass/pipeline flags are given, default to
  // --convert-cuda-tile-to-mlir.
  bool hasPassFlag = false;
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg.starts_with("--convert-") || arg.starts_with("--pass-pipeline") ||
        arg.starts_with("-convert-") || arg.starts_with("-pass-pipeline"))
      hasPassFlag = true;
  }

  std::vector<const char *> newArgv(argv, argv + argc);
  // if (!hasPassFlag) {
  //   newArgv.push_back("--pass-pipeline=builtin.module("
  //                     "convert-cuda-tile-to-mlir,"
  //                     "loop-invariant-code-motion,"
  //                     "canonicalize,"
  //                     "cse)");
  // }
  int newArgc = static_cast<int>(newArgv.size());

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(newArgc, const_cast<char **>(newArgv.data()),
                        "CudaTileToMLIR optimizer driver\n", registry));
}
