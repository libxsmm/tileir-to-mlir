//===- TileIRToGPU.h - CudaTile IR to GPU/vector conversion -----*- C++ -*-===//
//
// Conversion pass from CudaTile IR to GPU/vector/scf/arith/memref ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CUDATILETOGPU_CUDATILETOGPU_H
#define MLIR_CONVERSION_CUDATILETOGPU_CUDATILETOGPU_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

/// Creates the pass that converts CudaTile IR to GPU/vector IR.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTileIRToGPUPass();

} // namespace mlir

#endif // MLIR_CONVERSION_CUDATILETOGPU_CUDATILETOGPU_H
