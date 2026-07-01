// RUN: cudatile-to-mlir --convert-memref-args-to-ranked-memref %s | FileCheck %s
// RUN: cudatile-to-mlir --convert-memref-args-to-ranked-memref=remove-unused=false %s | FileCheck %s --check-prefix=KEEP

// Verifies that unranked memref kernel arguments that are always reinterpreted
// the same way are promoted to ranked memref arguments, and scalar shape/stride
// arguments are replaced with memref.dim / extract_strided_metadata-derived
// values and removed from the function signature.
//
// With remove-unused=false the now-unused shape/stride scalar arguments are
// kept in the signature.

module attributes {gpu.container_module} {
  gpu.module @m {
    // CHECK-LABEL: gpu.func @kernel(
    // CHECK-SAME: %[[A:[^:]+]]: memref<?x64xf16, strided<[?, 1], offset: ?>>,
    // CHECK-SAME: %[[N:[^:]+]]: i32)
    // CHECK-NOT: memref.reinterpret_cast
    // KEEP-LABEL: gpu.func @kernel(
    // KEEP-SAME: %{{[^:]+}}: memref<?x64xf16, strided<[?, 1], offset: ?>>, %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32)
    // KEEP-NOT: memref.reinterpret_cast
    gpu.func @kernel(%arg0: memref<*xf16>, %arg1: i32, %arg2: i32, %arg3: i32) kernel {
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
