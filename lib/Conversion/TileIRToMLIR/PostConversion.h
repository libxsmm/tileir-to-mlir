//===- PostConversion.h - Rewrites after the Tile IR conversion -*- C++ -*-===//
//
// Part of the tileir-to-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Rewrites of the IR produced by --convert-tileir-to-mlir.
//
//===----------------------------------------------------------------------===//

#ifndef TILEIRTOMLIR_POSTCONVERSION_H
#define TILEIRTOMLIR_POSTCONVERSION_H

namespace mlir {
class Operation;
class RewritePatternSet;
namespace tileir {

/// Index vector transfers directly by loop induction variables:
///   - a loop whose induction variable `iv` indexes transfers as `iv * T` is
///     rescaled to step over element indices, which makes those indices `iv`;
///     other uses get `iv / T`;
///   - a transfer index `(iv / T) * T`, where `iv` is always a multiple of
///     `T`, becomes `iv`.
/// Integer range analysis proves every rewrite exact.
void rescaleTileLoops(Operation *root);

/// Wrap the body of every scf.for that allocates on the stack in a
/// memref.alloca_scope, which cuda_tile.for implies but scf.for does not.
void scopeLoopAllocations(Operation *root);

/// Collect the local rewrite patterns for the converted IR.
void populatePostConversionPatterns(RewritePatternSet &patterns);

} // namespace tileir
} // namespace mlir

#endif // TILEIRTOMLIR_POSTCONVERSION_H
