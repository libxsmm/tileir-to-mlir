//===- TileIRToMLIR.h - Tile IR to MLIR conversion --------------*- C++ -*-===//
//
// Part of the tileir-to-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Conversion pass from Tile IR to GPU/vector/scf/arith/memref ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_TILEIRTOMLIR_TILEIRTOMLIR_H
#define MLIR_CONVERSION_TILEIRTOMLIR_TILEIRTOMLIR_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

/// Selects the lowering target for the tileir-to-mlir conversion.
enum class TileIRTarget { GPU, CPU };

/// Generated pass declarations (ConvertTileIRToMLIRPassOptions and the
/// createConvertTileIRToMLIRPass factories). The TileIRTarget enum above must
/// be defined before this include, as it is referenced by the generated code.
#define GEN_PASS_DECL_CONVERTTILEIRTOMLIRPASS
#include "mlir/Conversion/TileIRToMLIR/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_TILEIRTOMLIR_TILEIRTOMLIR_H
