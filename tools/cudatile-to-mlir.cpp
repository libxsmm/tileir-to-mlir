//===- cudatile-to-mlir.cpp - Driver for CudaTileToMLIR pass -------------===//
//
// Simple mlir-opt-style driver that registers the CudaTileToMLIR conversion
// pass together with the dialects it depends on.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/CudaTileToMLIR/ConvertMemrefArgsToPtrArgs.h"
#include "mlir/Conversion/CudaTileToMLIR/CudaTileToMLIR.h"
#include "mlir/Conversion/CudaTileToMLIR/TileIRPtrToView.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "cuda_tile/Bytecode/Reader/BytecodeReader.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"

#include <iostream>

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<
      mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::gpu::GPUDialect,
      mlir::memref::MemRefDialect, mlir::scf::SCFDialect, mlir::ub::UBDialect,
      mlir::vector::VectorDialect, mlir::cuda_tile::CudaTileDialect>();

  mlir::func::registerInlinerExtension(registry);

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createConvertTileIRToMLIRPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createTileIRPtrToViewPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createConvertMemrefArgsToPtrArgsPass();
  });

  mlir::registerTransformsPasses();

  llvm::InitLLVM y(argc, argv);

  // Register and parse the command line options up front so we can decide how
  // to load the input (textual MLIR vs. TileIR bytecode) before handing the
  // already-parsed IR to MlirOptMain.
  std::string inputFilename, outputFilename;
  std::tie(inputFilename, outputFilename) = mlir::registerAndParseCLIOptions(
      argc, argv, "CudaTileToMLIR optimizer driver\n", registry);

  mlir::MlirOptMainConfig config =
      mlir::MlirOptMainConfig::createFromCLOptions();

  // When reading from stdin and the input is a tty, warn the user (mirrors the
  // behavior of the default MlirOptMain driver).
  if (inputFilename == "-" &&
      llvm::sys::Process::FileDescriptorIsDisplayed(fileno(stdin)))
    llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                    "interrupt)\n";

  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> input =
      mlir::openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return EXIT_FAILURE;
  }

  std::unique_ptr<llvm::ToolOutputFile> output =
      mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return EXIT_FAILURE;
  }

  // Treat the input as TileIR bytecode when the file has the ".tileirbc"
  // extension, or (e.g. when coming from stdin) when the buffer carries the
  // TileIR bytecode magic.
  bool isBytecode = llvm::StringRef(inputFilename).ends_with(".tileirbc") ||
                    mlir::cuda_tile::isTileIRBytecode(input->getMemBufferRef());

  std::unique_ptr<llvm::MemoryBuffer> buffer;
  if (isBytecode) {
    // Decode the bytecode into a TileIR module and re-serialize it as textual
    // MLIR so it can flow through the regular MlirOptMain processing pipeline.
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    mlir::OwningOpRef<mlir::cuda_tile::ModuleOp> module =
        mlir::cuda_tile::readBytecode(input->getMemBufferRef(), context);
    if (!module) {
      llvm::errs() << "failed to read TileIR bytecode from '" << inputFilename
                   << "'\n";
      return EXIT_FAILURE;
    }
    std::cout << "Successfully read TileIR bytecode from '" << inputFilename
              << std::flush;

    std::string text;
    llvm::raw_string_ostream os(text);
    module.get().getOperation()->print(os);
    os.flush();
    buffer = llvm::MemoryBuffer::getMemBufferCopy(text, inputFilename);
  } else {
    buffer = std::move(input);
  }

  if (failed(
          mlir::MlirOptMain(output->os(), std::move(buffer), registry, config)))
    return EXIT_FAILURE;

  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return EXIT_SUCCESS;
}
