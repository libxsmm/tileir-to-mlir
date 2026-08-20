//===- TileIRPtrToView.h - Triton ptr-arith -> TileIR view ops *- C++ -*-===//
//
// Part of the tileir-to-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
