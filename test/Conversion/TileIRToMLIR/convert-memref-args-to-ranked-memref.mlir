// RUN: tileir-to-mlir --convert-memref-args-to-ranked-memref %s | FileCheck %s
// RUN: tileir-to-mlir --convert-memref-args-to-ranked-memref=remove-unused=memref-dependent %s | FileCheck %s --check-prefix=MEMREF
// RUN: tileir-to-mlir --convert-memref-args-to-ranked-memref=remove-unused=assumed-memref-dependent %s | FileCheck %s --check-prefix=ASSUMED
// RUN: tileir-to-mlir --convert-memref-args-to-ranked-memref=remove-unused=other %s | FileCheck %s --check-prefix=OTHER
// RUN: tileir-to-mlir --convert-memref-args-to-ranked-memref=remove-unused=all %s | FileCheck %s --check-prefix=ALL
// RUN: tileir-to-mlir --convert-memref-args-to-ranked-memref=remove-unused=none %s | FileCheck %s --check-prefix=NONE

// Verifies that unranked memref kernel arguments that are always reinterpreted
// the same way are promoted to ranked memref arguments, and scalar shape/stride
// arguments are replaced with memref.dim / extract_strided_metadata-derived
// values. The default memref-dependent mode removes only those arguments, not
// unrelated unused arguments.
//
// The assumed-memref-dependent mode additionally removes the unused final
// stride slot of the rank-2 calling convention, even though the cast hardcodes
// its stride to 1. Explicit mode checks cover all five removal policies.

module attributes {gpu.container_module} {
  gpu.module @m {
    // CHECK-LABEL: gpu.func @kernel(
    // CHECK-SAME: %[[A:[^:]+]]: memref<?x64xf16, strided<[?, 1], offset: ?>>,
    // CHECK-SAME: %[[N:[^:]+]]: i32, %{{[^:]+}}: i32)
    // CHECK-NOT: memref.reinterpret_cast
    // MEMREF-LABEL: gpu.func @kernel(
    // MEMREF-SAME: %{{[^:]+}}: memref<?x64xf16, strided<[?, 1], offset: ?>>, %{{[^:]+}}: i32, %{{[^:]+}}: i32)
    // MEMREF-NOT: memref.reinterpret_cast
    // ASSUMED-LABEL: gpu.func @kernel(
    // ASSUMED-SAME: %{{[^:]+}}: memref<?x64xf16, strided<[?, 1], offset: ?>>, %{{[^:]+}}: i32)
    // ASSUMED-NOT: memref.reinterpret_cast
    // OTHER-LABEL: gpu.func @kernel(
    // OTHER-SAME: %{{[^:]+}}: memref<?x64xf16, strided<[?, 1], offset: ?>>, %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32)
    // OTHER-NOT: memref.reinterpret_cast
    // ALL-LABEL: gpu.func @kernel(
    // ALL-SAME: %{{[^:]+}}: memref<?x64xf16, strided<[?, 1], offset: ?>>, %{{[^:]+}}: i32)
    // ALL-NOT: memref.reinterpret_cast
    // NONE-LABEL: gpu.func @kernel(
    // NONE-SAME: %{{[^:]+}}: memref<?x64xf16, strided<[?, 1], offset: ?>>, %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32)
    // NONE-NOT: memref.reinterpret_cast
    gpu.func @kernel(%arg0: memref<*xf16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) attributes {sym_visibility = "private"} {
      %c0 = arith.constant 0 : index
      %m = arith.index_cast %arg1 : i32 to index
      %s = arith.index_cast %arg2 : i32 to index
      %view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%m, 64], strides: [%s, 1] : memref<*xf16> to memref<?x64xf16, strided<[?, 1], offset: ?>>

      // CHECK: %[[D0:.*]] = memref.dim %[[A]], %{{.*}} : memref<?x64xf16, strided<[?, 1], offset: ?>>
      // CHECK: %[[D0_I32:.*]] = arith.index_cast %[[D0]] : index to i32
      // CHECK: arith.addi %[[D0_I32]], %[[N]] : i32
      %d0 = memref.dim %view, %c0 : memref<?x64xf16, strided<[?, 1], offset: ?>>
      %d0_i32 = arith.index_cast %d0 : index to i32
      %sum = arith.addi %d0_i32, %arg3 : i32

      // CHECK: %{{.*}}, %{{.*}}, %{{.*}}:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[A]] : memref<?x64xf16, strided<[?, 1], offset: ?>> -> memref<f16>, index, index, index, index, index
      // CHECK: %[[S0_I32:.*]] = arith.index_cast %[[STRIDES]]#0 : index to i32
      // CHECK: arith.muli %[[S0_I32]], %[[N]] : i32
      %base, %off, %sz0, %sz1, %st0, %st1 = memref.extract_strided_metadata %view : memref<?x64xf16, strided<[?, 1], offset: ?>> -> memref<f16>, index, index, index, index, index
      %st0_i32 = arith.index_cast %st0 : index to i32
      %prod = arith.muli %st0_i32, %arg3 : i32

      gpu.return
    }
  }
}
