// RUN: cudatile-to-mlir --convert-memref-args-to-ptr-args %s | FileCheck %s

// Verifies the safety guards of --convert-memref-args-to-ptr-args: arguments are left as
// unranked memrefs (and their reinterpret_casts kept) whenever the promotion
// would be ambiguous or would break an in-module call site.

// Divergent reinterpretations of the same argument (different ranked result
// types) are ambiguous, so the argument is not promoted.
// CHECK-LABEL: func.func @divergent(
// CHECK-SAME:    %{{[^:]+}}: memref<*xf32>
// CHECK:         memref.reinterpret_cast
// CHECK:         memref.reinterpret_cast
func.func @divergent(%arg0: memref<*xf32>, %arg1: i32) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = arith.index_cast %arg1 : i32 to index
  %a = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0], strides: [1] : memref<*xf32> to memref<?xf32, strided<[1]>>
  %b = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0, %0], strides: [%0, 1] : memref<*xf32> to memref<?x?xf32, strided<[?, 1]>>
  %u = vector.transfer_read %a[%0], %cst : memref<?xf32, strided<[1]>>, vector<4xf32>
  %v = vector.transfer_read %b[%0, %0], %cst : memref<?x?xf32, strided<[?, 1]>>, vector<4x4xf32>
  return
}

// A non-cast use of the argument (here memref.rank observes the unranked type)
// blocks the promotion: the argument must stay unranked.
// CHECK-LABEL: func.func @mixed_use(
// CHECK-SAME:    %{{[^:]+}}: memref<*xf32>
// CHECK:         memref.reinterpret_cast
// CHECK:         memref.rank
func.func @mixed_use(%arg0: memref<*xf32>, %arg1: i32) {
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
// CHECK-LABEL: func.func @callee(
// CHECK-SAME:    %{{[^:]+}}: memref<*xf32>
// CHECK:         memref.reinterpret_cast
func.func @callee(%arg0: memref<*xf32>, %arg1: i32) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = arith.index_cast %arg1 : i32 to index
  %a = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0], strides: [1] : memref<*xf32> to memref<?xf32, strided<[1]>>
  %u = vector.transfer_read %a[%0], %cst : memref<?xf32, strided<[1]>>, vector<4xf32>
  return
}

// CHECK-LABEL: func.func @caller(
// CHECK-SAME:    %{{[^:]+}}: memref<*xf32>
// CHECK:         call @callee
func.func @caller(%arg0: memref<*xf32>, %arg1: i32) {
  func.call @callee(%arg0, %arg1) : (memref<*xf32>, i32) -> ()
  return
}
