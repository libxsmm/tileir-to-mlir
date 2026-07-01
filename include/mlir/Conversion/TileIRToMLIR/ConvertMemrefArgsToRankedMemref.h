//===- ConvertMemrefArgsToRankedMemref.h -----------------------*- C++ -*-===//
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

/// Generated pass declarations (createConvertMemrefArgsToRankedMemrefPass
/// factory).
#define GEN_PASS_DECL_CONVERTMEMREFARGSTORANKEDMEMREFPASS
#include "mlir/Conversion/TileIRToMLIR/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_TILEIRTOMLIR_CONVERTMEMREFARGSTORANKEDMEMREF_H
