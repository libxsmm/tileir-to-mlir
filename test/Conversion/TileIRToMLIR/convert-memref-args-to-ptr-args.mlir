// RUN: tileir-to-mlir --convert-memref-args-to-ptr-args %s | FileCheck %s

// Verifies that --convert-memref-args-to-ptr-args promotes unranked-memref
// function arguments to `!llvm.ptr`, and -- in place of each redundant
// `memref.reinterpret_cast` -- rebuilds the ranked memref by packing an LLVM
// memref descriptor from the pointer and casting it back. The body keeps using
// the recovered ranked memref. The pass is generic over FunctionOpInterface, so
// it covers both func.func and gpu.func.

// CHECK-LABEL: func.func @add_kernel(
//   All three pointer inputs become !llvm.ptr; the scalar i32 args are untouched.
// CHECK-SAME:    %[[A0:[^:]+]]: !llvm.ptr, %[[A1:[^:]+]]: !llvm.ptr, %[[A2:[^:]+]]: !llvm.ptr,
// CHECK-SAME:    %{{[^:]+}}: i32, %{{[^:]+}}: i32)
// CHECK-NOT:     memref.reinterpret_cast
//   Each pointer arg is packed into a memref descriptor and cast back to a ranked memref.
// CHECK:         llvm.insertvalue %[[A2]]
// CHECK:         %[[M2:.*]] = builtin.unrealized_conversion_cast %{{.*}} to memref<?xf32, strided<[1]>>
// CHECK:         llvm.insertvalue %[[A1]]
// CHECK:         %[[M1:.*]] = builtin.unrealized_conversion_cast %{{.*}} to memref<?xf32, strided<[1]>>
// CHECK:         llvm.insertvalue %[[A0]]
// CHECK:         %[[M0:.*]] = builtin.unrealized_conversion_cast %{{.*}} to memref<?xf32, strided<[1]>>
// CHECK:         %[[R0:.*]] = vector.transfer_read %[[M0]]
// CHECK:         %[[R1:.*]] = vector.transfer_read %[[M1]]
// CHECK:         %[[SUM:.*]] = arith.addf %[[R0]], %[[R1]]
// CHECK:         vector.transfer_write %[[SUM]], %[[M2]]
// CHECK:         return
func.func @add_kernel(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32, %arg4: i32) {
  %cst = arith.constant 0.000000e+00 : f32
  %c4096 = arith.constant 4096 : index
  // %0 only feeds the reinterpret_casts, so it must be cleaned up with them.
  %0 = arith.index_cast %arg3 : i32 to index
  %rc2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%0], strides: [1] : memref<*xf32> to memref<?xf32, strided<[1]>>
  %rc1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%0], strides: [1] : memref<*xf32> to memref<?xf32, strided<[1]>>
  %rc0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0], strides: [1] : memref<*xf32> to memref<?xf32, strided<[1]>>
  %1 = arith.index_cast %arg4 : i32 to index
  %2 = arith.muli %1, %c4096 : index
  %3 = vector.transfer_read %rc0[%2], %cst : memref<?xf32, strided<[1]>>, vector<4096xf32>
  %4 = vector.transfer_read %rc1[%2], %cst : memref<?xf32, strided<[1]>>, vector<4096xf32>
  %5 = arith.addf %3, %4 : vector<4096xf32>
  vector.transfer_write %5, %rc2[%2] : vector<4096xf32>, memref<?xf32, strided<[1]>>
  return
}

// A single argument promoted to a 2-D strided memref: shows the pass is not
// limited to rank-1 and preserves a non-trivial recovered layout. Here the
// dynamic index operand (%0) is still used by the load.
// CHECK-LABEL: func.func @rank2(
// CHECK-SAME:    %[[B0:[^:]+]]: !llvm.ptr, %{{[^:]+}}: i32)
// CHECK-NOT:     memref.reinterpret_cast
// CHECK:         llvm.insertvalue %[[B0]]
// CHECK:         %[[M:.*]] = builtin.unrealized_conversion_cast %{{.*}} to memref<?x?xf32, strided<[?, 1]>>
// CHECK:         vector.transfer_read %[[M]]
func.func @rank2(%arg0: memref<*xf32>, %arg1: i32) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = arith.index_cast %arg1 : i32 to index
  %rc = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0, %0], strides: [%0, 1] : memref<*xf32> to memref<?x?xf32, strided<[?, 1]>>
  %v = vector.transfer_read %rc[%0, %0], %cst : memref<?x?xf32, strided<[?, 1]>>, vector<4x4xf32>
  return
}

// The same promotion works on a gpu.func (FunctionOpInterface), independent of
// the func dialect.
// CHECK-LABEL: gpu.func @gpu_kernel(
// CHECK-SAME:    %[[G0:[^:]+]]: !llvm.ptr, %{{[^:]+}}: i32)
// CHECK-NOT:     memref.reinterpret_cast
// CHECK:         llvm.insertvalue %[[G0]]
// CHECK:         %[[GM:.*]] = builtin.unrealized_conversion_cast %{{.*}} to memref<?xf32, strided<[1]>>
// CHECK:         vector.transfer_read %[[GM]]
// CHECK:         gpu.return
module attributes {gpu.container_module} {
  gpu.module @m {
    gpu.func @gpu_kernel(%arg0: memref<*xf32>, %arg1: i32) kernel {
      %cst = arith.constant 0.000000e+00 : f32
      %0 = arith.index_cast %arg1 : i32 to index
      %rc = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0], strides: [1] : memref<*xf32> to memref<?xf32, strided<[1]>>
      %v = vector.transfer_read %rc[%0], %cst : memref<?xf32, strided<[1]>>, vector<8xf32>
      gpu.return
    }
  }
}

// An argument reinterpreted several different ways is still promoted: each cast
// is rebuilt independently from the recovered pointer, so divergent ranks /
// layouts are fine.
// CHECK-LABEL: func.func @divergent(
// CHECK-SAME:    %[[D0:[^:]+]]: !llvm.ptr, %{{[^:]+}}: i32)
// CHECK-NOT:     memref.reinterpret_cast
// CHECK:         %[[DA:.*]] = builtin.unrealized_conversion_cast %{{.*}} to memref<?xf32, strided<[1]>>
// CHECK:         %[[DB:.*]] = builtin.unrealized_conversion_cast %{{.*}} to memref<?x?xf32, strided<[?, 1]>>
// CHECK:         vector.transfer_read %[[DA]]
// CHECK:         vector.transfer_read %[[DB]]
func.func @divergent(%arg0: memref<*xf32>, %arg1: i32) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = arith.index_cast %arg1 : i32 to index
  %a = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0], strides: [1] : memref<*xf32> to memref<?xf32, strided<[1]>>
  %b = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%0, %0], strides: [%0, 1] : memref<*xf32> to memref<?x?xf32, strided<[?, 1]>>
  %u = vector.transfer_read %a[%0], %cst : memref<?xf32, strided<[1]>>, vector<4xf32>
  %v = vector.transfer_read %b[%0, %0], %cst : memref<?x?xf32, strided<[?, 1]>>, vector<4x4xf32>
  return
}

// Coupled casts: a first reinterpret_cast (with a non-zero absolute offset)
// feeds only `extract_strided_metadata`, whose recovered offset drives a second
// reinterpret_cast of the same argument. Because the offset is kept in the
// descriptor's offset field (not folded into the pointer), the metadata still
// recovers the original offset, so the second view keeps its correct address.
// CHECK-LABEL: func.func @coupled_offset(
// CHECK-SAME:    %[[C0:[^:]+]]: !llvm.ptr, %{{[^:]+}}: i32, %{{[^:]+}}: i32)
// CHECK-NOT:     memref.reinterpret_cast
//   First cast: the absolute offset lands in the descriptor's offset field (index 2),
//   and the argument pointer stays the base buffer.
// CHECK:         %[[OFF:.*]] = arith.index_cast %{{.*}} : index to i64
// CHECK:         llvm.insertvalue %[[C0]], %{{.*}}[0]
// CHECK:         llvm.insertvalue %[[C0]], %{{.*}}[1]
// CHECK:         llvm.insertvalue %[[OFF]], %{{.*}}[2]
// CHECK:         %[[M1:.*]] = builtin.unrealized_conversion_cast %{{.*}} to memref<1xf32, strided<[1], offset: ?>>
//   The recovered offset feeds the real view, which the load uses.
// CHECK:         %{{.*}}, %[[ROFF:.*]], %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[M1]]
// CHECK:         %[[M2:.*]] = builtin.unrealized_conversion_cast %{{.*}} to memref<?xf32, strided<[1], offset: ?>>
// CHECK:         vector.transfer_read %[[M2]]
func.func @coupled_offset(%arg0: memref<*xf32>, %arg1: i32, %arg2: i32) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %off = arith.index_cast %arg1 : i32 to index
  %n = arith.index_cast %arg2 : i32 to index
  %probe = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
  %base, %roff, %sz, %st = memref.extract_strided_metadata %probe : memref<1xf32, strided<[1], offset: ?>> -> memref<f32>, index, index, index
  %view = memref.reinterpret_cast %arg0 to offset: [%roff], sizes: [%n], strides: [1] : memref<*xf32> to memref<?xf32, strided<[1], offset: ?>>
  %v = vector.transfer_read %view[%c0], %cst : memref<?xf32, strided<[1], offset: ?>>, vector<4xf32>
  return
}
