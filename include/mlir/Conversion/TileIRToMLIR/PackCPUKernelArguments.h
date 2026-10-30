// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MLIR_CONVERSION_TILEIRTOMLIR_PACKCPUKERNELARGUMENTS_H
#define MLIR_CONVERSION_TILEIRTOMLIR_PACKCPUKERNELARGUMENTS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

#define GEN_PASS_DECL_PACKCPUKERNELARGUMENTSPASS
#include "mlir/Conversion/TileIRToMLIR/Passes.h.inc"
} // namespace mlir

#endif
