// SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Conversion/TileIRToMLIR/Passes.h"
#include "mlir/Pass/PassManager.h"

mlir::LogicalResult mlir::buildTileIRToMLIRPipeline(
    OpPassManager &pm, llvm::StringRef options,
    llvm::function_ref<LogicalResult(const llvm::Twine &)> errorHandler) {
  auto convert = createConvertTileIRToMLIRPass();
  if (failed(convert->initializeOptions(options, errorHandler)))
    return failure();
  pm.addPass(createTileIRPtrToViewPass());
  pm.addPass(std::move(convert));
  pm.addPass(createCanonicalizeFromTileIRPass());
  return success();
}
