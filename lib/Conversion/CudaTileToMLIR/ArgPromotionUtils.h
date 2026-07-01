//===- ArgPromotionUtils.h - shared entry-arg promotion helpers ---------===//
//
// Small utilities shared by the entry-argument promotion passes
// (ConvertMemrefArgsToPtrArgs, ConvertMemrefArgsToRankedMemref).
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILETOMLIR_ARGPROMOTIONUTILS_H
#define CUDATILETOMLIR_ARGPROMOTIONUTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include <optional>

namespace mlir {
namespace cudatile {

/// Returns `true` when `ofr` is a statically-known zero offset.
inline bool isStaticZero(OpFoldResult ofr) {
  auto attr = dyn_cast<Attribute>(ofr);
  auto intAttr = attr ? dyn_cast<IntegerAttr>(attr) : nullptr;
  return intAttr && intAttr.getValue().isZero();
}

/// Returns `true` iff changing `func`'s signature is safe, i.e. the function is
/// not referenced (called / launched) from within its nearest symbol table.
/// In-module references would be invalidated by a signature change, so such
/// functions are left untouched. A non-`SymbolOpInterface` or an unresolved
/// use set is treated conservatively as "unsafe".
inline bool signatureChangeIsSafe(FunctionOpInterface func) {
  auto symbol = dyn_cast<SymbolOpInterface>(func.getOperation());
  if (!symbol)
    return false;
  Operation *symbolTableOp = func->getParentWithTrait<OpTrait::SymbolTable>();
  if (!symbolTableOp)
    return false;
  std::optional<SymbolTable::UseRange> uses =
      SymbolTable::getSymbolUses(func.getOperation(), symbolTableOp);
  return uses && uses->empty();
}

} // namespace cudatile
} // namespace mlir

#endif // CUDATILETOMLIR_ARGPROMOTIONUTILS_H
