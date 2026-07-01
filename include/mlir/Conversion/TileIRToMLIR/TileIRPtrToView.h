//===- TileIRPtrToView.h - Triton ptr-arith -> TileIR view ops *- C++ -*-===//
//
// Preprocessing pass that rewrites the canonical Triton-emitted
// iota+reshape+broadcast+offset pointer-arithmetic chain feeding a
// `cuda_tile.load_ptr_tko` / `cuda_tile.store_ptr_tko` into the higher-level
// `cuda_tile.make_tensor_view` + `cuda_tile.make_partition_view` +
// `cuda_tile.load_view_tko` / `cuda_tile.store_view_tko` form, which the
// `--convert-tileir-to-mlir` pass can lower.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_TILEIRTOMLIR_TILEIRPTRTOVIEW_H
#define MLIR_CONVERSION_TILEIRTOMLIR_TILEIRPTRTOVIEW_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

/// Generated pass declarations (createTileIRPtrToViewPass factory).
#define GEN_PASS_DECL_TILEIRPTRTOVIEWPASS
#include "mlir/Conversion/TileIRToMLIR/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_TILEIRTOMLIR_TILEIRPTRTOVIEW_H
