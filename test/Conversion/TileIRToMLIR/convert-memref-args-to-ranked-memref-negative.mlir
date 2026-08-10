// RUN: tileir-to-mlir --convert-memref-args-to-ranked-memref=remove-unused=none %s | FileCheck %s

// Verifies conservative behavior:
// 1) referenced functions are not rewritten (signature-change safety),
// 2) arguments with non-identical reinterpret casts are not promoted,
// 3) shared scalar shape args with conflicting memref sources are not removed.
// 4) non-zero reinterpret offsets are not folded into the ranked argument.

// CHECK-LABEL: func.func @callee(
// CHECK-SAME: %[[P:[^:]+]]: memref<*xf32>, %[[N:[^:]+]]: i32)
// CHECK: memref.reinterpret_cast %[[P]]
func.func @callee(%arg0: memref<*xf32>, %arg1: i32) {
  %n = arith.index_cast %arg1 : i32 to index
  %v = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%n], strides: [1] : memref<*xf32> to memref<?xf32, strided<[1], offset: ?>>
  return
}

// CHECK-LABEL: func.func @caller(
// CHECK: call @callee
func.func @caller(%arg0: memref<*xf32>, %arg1: i32) {
  call @callee(%arg0, %arg1) : (memref<*xf32>, i32) -> ()
  return
}

module attributes {gpu.container_module} {
  gpu.module @m {
    // CHECK-LABEL: gpu.func @divergent_cast(
    // CHECK-SAME: memref<*xf32>, %[[N:[^:]+]]: i32)
    // CHECK: memref.reinterpret_cast
    // CHECK: memref.reinterpret_cast
    gpu.func @divergent_cast(%arg0: memref<*xf32>, %arg1: i32) kernel {
      %n = arith.index_cast %arg1 : i32 to index
      %a = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%n], strides: [1] : memref<*xf32> to memref<?xf32, strided<[1], offset: ?>>
      %b = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%n, %n], strides: [%n, 1] : memref<*xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      gpu.return
    }

    // Shared scalar %arg2 feeds two different memrefs' dynamic sizes.
    // Both pointer args can be promoted, but %arg2 must remain because there
    // is no unique memref recipe to replace all of its uses.
    // CHECK-LABEL: gpu.func @conflicting_scalar(
    // CHECK-SAME: %[[A:[^:]+]]: memref<?xf16, strided<[1], offset: ?>>,
    // CHECK-SAME: %[[B:[^:]+]]: memref<?xf16, strided<[1], offset: ?>>,
    // CHECK-SAME: %[[N:[^:]+]]: i32)
    // CHECK-NOT: memref.reinterpret_cast
    gpu.func @conflicting_scalar(%arg0: memref<*xf16>, %arg1: memref<*xf16>, %arg2: i32) kernel {
      %n = arith.index_cast %arg2 : i32 to index
      %a = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%n], strides: [1] : memref<*xf16> to memref<?xf16, strided<[1], offset: ?>>
      %b = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%n], strides: [1] : memref<*xf16> to memref<?xf16, strided<[1], offset: ?>>
      %u = arith.addi %arg2, %arg2 : i32
      gpu.return
    }

    // Non-zero offset cannot be represented by handing back the bare arg,
    // so the unranked argument is left as-is.
    // CHECK-LABEL: gpu.func @nonzero_offset(
    // CHECK-SAME: memref<*xf16>,
    // CHECK: memref.reinterpret_cast
    gpu.func @nonzero_offset(%arg0: memref<*xf16>, %arg1: i32, %arg2: i32) kernel {
      %n = arith.index_cast %arg1 : i32 to index
      %off = arith.index_cast %arg2 : i32 to index
      %a = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [%n], strides: [1] : memref<*xf16> to memref<?xf16, strided<[1], offset: ?>>
      gpu.return
    }
  }
}
