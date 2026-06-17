//===- TileIRToMLIR.h - CudaTile IR to MLIR conversion ----------*- C++ -*-===//
//
// Conversion pass from CudaTile IR to GPU/vector/scf/arith/memref ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CUDATILETOMLIR_CUDATILETOMLIR_H
#define MLIR_CONVERSION_CUDATILETOMLIR_CUDATILETOMLIR_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

/// Creates the pass that converts CudaTile IR to GPU/vector IR.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTileIRToMLIRPass();

} // namespace mlir

#endif // MLIR_CONVERSION_CUDATILETOMLIR_CUDATILETOMLIR_H
