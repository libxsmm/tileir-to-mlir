// RUN: tileir-to-mlir --convert-memref-args-to-ptr-args %s | FileCheck %s

// Verifies the safety guards of --convert-memref-args-to-ptr-args: arguments are left as
// unranked memrefs (and their reinterpret_casts kept) whenever the promotion
// would break an in-module call site, the unranked type is observed by a
// non-cast user, or a cast cannot be rebuilt as an LLVM descriptor from a
// pointer.

// A non-cast use of the argument (here memref.rank observes the unranked type)
// blocks the promotion: the argument must stay unranked.
// CHECK-LABEL: func.func private @mixed_use(
// CHECK-SAME:    %{{[^:]+}}: memref<*xf32>
// CHECK:         memref.reinterpret_cast
// CHECK:         memref.rank
func.func private @mixed_use(%arg0: memref<*xf32>, %arg1: i32) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = arith.index_cast %arg1 : i32 to index
  %a = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0], strides: [1] : memref<*xf32> to memref<?xf32, strided<[1]>>
  %u = vector.transfer_read %a[%0], %cst : memref<?xf32, strided<[1]>>, vector<4xf32>
  %rnk = memref.rank %arg0 : memref<*xf32>
  return
}

// @callee is called from within the module, so changing its signature would
// break the call. It is left untouched; @caller passes its argument to a
// non-cast user (the call), so it is not promoted either.
// CHECK-LABEL: func.func private @callee(
// CHECK-SAME:    %{{[^:]+}}: memref<*xf32>
// CHECK:         memref.reinterpret_cast
func.func private @callee(%arg0: memref<*xf32>, %arg1: i32) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = arith.index_cast %arg1 : i32 to index
  %a = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0], strides: [1] : memref<*xf32> to memref<?xf32, strided<[1]>>
  %u = vector.transfer_read %a[%0], %cst : memref<?xf32, strided<[1]>>, vector<4xf32>
  return
}

// CHECK-LABEL: func.func private @caller(
// CHECK-SAME:    %{{[^:]+}}: memref<*xf32>
// CHECK:         call @callee
func.func private @caller(%arg0: memref<*xf32>, %arg1: i32) {
  func.call @callee(%arg0, %arg1) : (memref<*xf32>, i32) -> ()
  return
}

// A plain cast reads the strides from the descriptor, so a cast to dynamic
// strides or to a non-strided layout blocks the promotion.
// CHECK-LABEL: func.func private @dynamic_stride_cast(
// CHECK-SAME:    %{{[^:]+}}: memref<*xf32>
// CHECK:         memref.cast
func.func private @dynamic_stride_cast(%arg0: memref<*xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %v = memref.cast %arg0 : memref<*xf32> to memref<4x4xf32, strided<[?, 1]>>
  %x = memref.load %v[%c1, %c0] : memref<4x4xf32, strided<[?, 1]>>
  return %x : f32
}

// A plain cast with a dynamic size reads it from the unranked descriptor.
// A pointer cannot supply that size to memref.dim.
// CHECK-LABEL: func.func private @dynamic_size_cast(
// CHECK-SAME:    %{{[^:]+}}: memref<*xf32>
// CHECK:         memref.cast
// CHECK:         memref.dim
func.func private @dynamic_size_cast(%arg0: memref<*xf32>) -> index {
  %c0 = arith.constant 0 : index
  %v = memref.cast %arg0 : memref<*xf32> to memref<?xf32, strided<[1], offset: ?>>
  %size = memref.dim %v, %c0 : memref<?xf32, strided<[1], offset: ?>>
  return %size : index
}

// CHECK-LABEL: func.func private @non_strided_cast(
// CHECK-SAME:    %{{[^:]+}}: memref<*xf32>
// CHECK:         memref.cast
func.func private @non_strided_cast(%arg0: memref<*xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %v = memref.cast %arg0 : memref<*xf32> to memref<4x4xf32, affine_map<(d0, d1) -> (d0 * 4 + d1 floordiv 2)>>
  %x = memref.load %v[%c1, %c0] : memref<4x4xf32, affine_map<(d0, d1) -> (d0 * 4 + d1 floordiv 2)>>
  return %x : f32
}

// The memref-to-LLVM lowering has no descriptor for a memref of tf32.
// CHECK-LABEL: func.func private @no_llvm_descriptor(
// CHECK-SAME:    %{{[^:]+}}: memref<*xtf32>
// CHECK:         memref.reinterpret_cast
func.func private @no_llvm_descriptor(%arg0: memref<*xtf32>) -> tf32 {
  %v = memref.reinterpret_cast %arg0 to offset: [0], sizes: [], strides: [] : memref<*xtf32> to memref<tf32, strided<[], offset: ?>>
  %x = memref.load %v[] : memref<tf32, strided<[], offset: ?>>
  return %x : tf32
}
