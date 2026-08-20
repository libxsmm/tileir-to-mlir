//===- ConvertMemrefArgsToRankedMemref.h -----------------------*- C++ -*-===//
//
// Part of the tileir-to-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Pass that rewrites unranked memref kernel arguments plus scalar
// shape/stride arguments into ranked memref arguments.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_TILEIRTOMLIR_CONVERTMEMREFARGSTORANKEDMEMREF_H
#define MLIR_CONVERSION_TILEIRTOMLIR_CONVERTMEMREFARGSTORANKEDMEMREF_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

/// Selects which unused arguments are removed after memref promotion.
enum class MemrefArgRemovalMode {
	All,
	None,
	MemrefDependent,
	AssumedMemrefDependent,
	Other,
};

/// Generated pass declarations (createConvertMemrefArgsToRankedMemrefPass
/// factory). The MemrefArgRemovalMode enum above must be defined before this
/// include, as it is referenced by the generated code.
#define GEN_PASS_DECL_CONVERTMEMREFARGSTORANKEDMEMREFPASS
#include "mlir/Conversion/TileIRToMLIR/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_TILEIRTOMLIR_CONVERTMEMREFARGSTORANKEDMEMREF_H
