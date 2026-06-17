//===- CudaTileToMLIR.h - CudaTile IR to MLIR conversion --------*- C++ -*-===//
//
// Conversion pass from CudaTile IR to GPU/vector/scf/arith/memref ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CUDATILETOMLIR_CUDATILETOMLIR_H
#define MLIR_CONVERSION_CUDATILETOMLIR_CUDATILETOMLIR_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

/// Selects the lowering target for the CudaTile-to-MLIR conversion.
enum class CudaTileTarget { GPU, CPU };

/// Generated pass declarations (ConvertTileIRToMLIRPassOptions and the
/// createConvertTileIRToMLIRPass factories). The CudaTileTarget enum above must
/// be defined before this include, as it is referenced by the generated code.
#define GEN_PASS_DECL_CONVERTTILEIRTOMLIRPASS
#include "mlir/Conversion/CudaTileToMLIR/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_CUDATILETOMLIR_CUDATILETOMLIR_H
