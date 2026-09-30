//===- Passes.h - TileIRToMLIR passes ---------------------------*- C++ -*-===//
//
// Part of the tileir-to-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_TILEIRTOMLIR_PASSES_H
#define MLIR_CONVERSION_TILEIRTOMLIR_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

/// Selects the lowering target of --convert-tileir-to-mlir.
enum class TileIRTarget { GPU, CPU };

/// Selects which unused arguments --convert-memref-args-to-ranked-memref
/// removes.
enum class MemrefArgRemovalMode {
  All,
  None,
  MemrefDependent,
  AssumedMemrefDependent,
  Other,
};

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "mlir/Conversion/TileIRToMLIR/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_TILEIRTOMLIR_PASSES_H
