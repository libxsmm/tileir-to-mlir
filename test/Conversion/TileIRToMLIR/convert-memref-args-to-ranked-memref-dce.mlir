// RUN: tileir-to-mlir --convert-memref-args-to-ranked-memref %s | FileCheck %s

// A definition used in multiple operand positions of one reinterpret_cast must
// only enter the dead-operation worklist once.
// CHECK-LABEL: func.func private @duplicate_dead_def(
// CHECK-SAME: %{{[^:]+}}: memref<?x?xf32, strided<[?, ?], offset: ?>>)
// CHECK-NOT: arith.constant
// CHECK-NOT: memref.reinterpret_cast
func.func private @duplicate_dead_def(%arg0: memref<*xf32>) {
  %c8 = arith.constant 8 : index
  %view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%c8, %c8], strides: [%c8, 1] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return
}