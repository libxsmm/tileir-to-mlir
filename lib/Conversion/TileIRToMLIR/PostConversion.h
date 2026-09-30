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

/// Make vector transfers index loops over tiles by the loop induction variable:
///   - a loop whose induction variable `iv` indexes transfers as `iv * T` is
///     rescaled to iterate over `iv * T`, and
///   - a transfer index `(iv / T) * T` of a loop whose induction variable is a
///     multiple of `T` becomes `iv`.
/// Every rewrite is proven exact with integer range analysis.
void rescaleTileLoops(Operation *root);

/// Collect the patterns that optimize the converted IR locally.
void populatePostConversionPatterns(RewritePatternSet &patterns);

} // namespace tileir
} // namespace mlir

#endif // TILEIRTOMLIR_POSTCONVERSION_H
