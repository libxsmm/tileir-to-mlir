//===- ConvertMemrefArgsToPtrArgs.h - memref->ptr args ---------*- C++ -*-===//
//
// Pass that promotes unranked-memref function arguments to `!llvm.ptr` when
// every use of the argument is an identical `memref.reinterpret_cast` that pins
// down a concrete ranked layout, dropping the otherwise-redundant casts.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CUDATILETOMLIR_CONVERTMEMREFARGSTOPTRARGS_H
#define MLIR_CONVERSION_CUDATILETOMLIR_CONVERTMEMREFARGSTOPTRARGS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

/// Generated pass declarations (createConvertMemrefArgsToPtrArgsPass factory).
#define GEN_PASS_DECL_CONVERTMEMREFARGSTOPTRARGSPASS
#include "mlir/Conversion/CudaTileToMLIR/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_CUDATILETOMLIR_CONVERTMEMREFARGSTOPTRARGS_H
