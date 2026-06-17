//===- TileIRPtrToView.h - Triton ptr-arith -> CudaTile view ops *- C++ -*-===//
//
// Preprocessing pass that rewrites the canonical Triton-emitted
// iota+reshape+broadcast+offset pointer-arithmetic chain feeding a
// `cuda_tile.load_ptr_tko` / `cuda_tile.store_ptr_tko` into the higher-level
// `cuda_tile.make_tensor_view` + `cuda_tile.make_partition_view` +
// `cuda_tile.load_view_tko` / `cuda_tile.store_view_tko` form, which the
// `--convert-cuda-tile-to-mlir` pass can lower.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CUDATILETOMLIR_TILEIRPTRTOVIEW_H
#define MLIR_CONVERSION_CUDATILETOMLIR_TILEIRPTRTOVIEW_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

/// Creates the pass that recognizes Triton-style ptr-arithmetic feeding
/// `load_ptr_tko`/`store_ptr_tko` and rewrites it into view-based loads/stores.
std::unique_ptr<OperationPass<ModuleOp>> createTileIRPtrToViewPass();

} // namespace mlir

#endif // MLIR_CONVERSION_CUDATILETOMLIR_TILEIRPTRTOVIEW_H
