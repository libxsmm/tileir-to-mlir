// RUN: cudatile-to-gpu --convert-cuda-tile-to-gpu %s | FileCheck %s
// RUN: cudatile-to-gpu --convert-cuda-tile-to-gpu %s | mlir-opt --loop-invariant-code-motion -canonicalize -cse > /dev/null

// Additional tests that complement cuda_tile_ir_ops.mlir.
// Focuses on: 2D array constants, vector iter_args, assume chaining,
// additional cmp predicates, dim_map swizzling, dynamic memref load/store.

// CHECK-LABEL: gpu.module @m {
cuda_tile.module @m {

  // --- 2D array constant (not covered by cuda_tile_ir_ops) ---
  // CHECK-LABEL: gpu.func @test_constant_2d_array
  entry @test_constant_2d_array() {
    // CHECK: arith.constant dense<{{.*}}> : vector<2x2xi32>
    %c3 = constant <i32: [[1, 2], [3, 4]]> : tile<2x2xi32>
  }

  // --- for with vector iter_arg ---
  // CHECK-LABEL: gpu.func @test_for_vector_iter_arg
  entry @test_for_vector_iter_arg() {
    %lb = constant <i32: 0> : tile<i32>
    %ub = constant <i32: 4> : tile<i32>
    %st = constant <i32: 1> : tile<i32>
    %init = constant <f32: 0.000000e+00> : tile<2x2xf32>

    // CHECK: arith.index_cast {{.*}} : i32 to index
    // CHECK: arith.index_cast {{.*}} : i32 to index
    // CHECK: arith.index_cast {{.*}} : i32 to index
    // CHECK: scf.for
    // CHECK: arith.index_cast {{.*}} : index to i32
    // CHECK: scf.yield
    %res = for %iv in (%lb to %ub, step %st) : tile<i32>
      iter_values(%acc = %init) -> (tile<2x2xf32>) {
      continue %acc : tile<2x2xf32>
    }
  }

  // --- assume passthrough in chain ---
  // CHECK-LABEL: gpu.func @test_assume_passthrough
  entry @test_assume_passthrough() {
    %a0 = constant <i32: 5> : tile<i32>
    %a1 = assume #cuda_tile.div_by<8>, %a0 : tile<i32>
    %b = constant <i32: 9> : tile<i32>
    // CHECK: arith.muli
    // CHECK-NOT: cuda_tile.assume
    %m = muli %a1, %b : tile<i32>
  }

  // --- additional cmp predicates and vector mulhii ---
  // CHECK-LABEL: gpu.func @test_cmp_and_mulhii_variants
  entry @test_cmp_and_mulhii_variants() {
    %xf = constant <f32: [1.000000e+00, -1.000000e+00, 0.000000e+00, 2.000000e+00]> : tile<4xf32>
    %yf = constant <f32: [1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00]> : tile<4xf32>
    %zi = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
    %wi = constant <i32: [4, 5, 6, 7]> : tile<4xi32>

    // CHECK: arith.cmpf olt
    %cmpf = cmpf less_than ordered %xf, %yf : tile<4xf32> -> tile<4xi1>
    // CHECK: arith.cmpf ueq
    %cmpf_unordered = cmpf equal unordered %xf, %yf : tile<4xf32> -> tile<4xi1>
    // CHECK: arith.cmpi ult
    %cmpi_u = cmpi less_than %zi, %wi, unsigned : tile<4xi32> -> tile<4xi1>
    // CHECK: arith.mului_extended
    %mulhi = mulhii %zi, %wi : tile<4xi32>
  }

  // --- get_index_space_shape with swizzled dim_map ---
  // CHECK-LABEL: gpu.func @test_shape_swizzled
  // CHECK-SAME: %[[SSW_UPTR:[a-zA-Z0-9_]+]]: memref<*xf16>
  entry @test_shape_swizzled(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %m: !cuda_tile.tile<i32>, %n: !cuda_tile.tile<i32>, %s: !cuda_tile.tile<i32>) {
    %tv = make_tensor_view %p, shape = [%m, %n], strides = [%s, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %pv = make_partition_view %tv : partition_view<tile=(16x8), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
    // CHECK: %[[SSW_PTR:.*]] = memref.reinterpret_cast %[[SSW_UPTR]]
    // CHECK: %[[SSW_D0:.*]] = memref.dim %[[SSW_PTR]], {{.*}}1
    // CHECK: %[[SSW_C16:.*]] = arith.constant 16 : index
    // CHECK: %[[SSW_Q0:.*]] = arith.ceildivui %[[SSW_D0]], %[[SSW_C16]] : index
    // CHECK: arith.index_cast %[[SSW_Q0]] : index to i32
    // CHECK: %[[SSW_D1:.*]] = memref.dim %[[SSW_PTR]], {{.*}}0
    // CHECK: %[[SSW_C8:.*]] = arith.constant 8 : index
    // CHECK: %[[SSW_Q1:.*]] = arith.ceildivui %[[SSW_D1]], %[[SSW_C8]] : index
    // CHECK: arith.index_cast %[[SSW_Q1]] : index to i32
    %dims:2 = get_index_space_shape %pv : partition_view<tile=(16x8), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]> -> tile<i32>
  }

  // --- load with dynamic memref, identity dim_map, ub.poison padding ---
  // CHECK-LABEL: gpu.func @test_load_identity_no_padding
  // CHECK-SAME: %[[LID_UPTR:[a-zA-Z0-9_]+]]: memref<*xf16>
  entry @test_load_identity_no_padding(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %m: !cuda_tile.tile<i32>, %n: !cuda_tile.tile<i32>, %s: !cuda_tile.tile<i32>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %tv = make_tensor_view %p, shape = [%m, %n], strides = [%s, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %pv = make_partition_view %tv : partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>
    // CHECK: %[[LID_PTR:.*]] = memref.reinterpret_cast %[[LID_UPTR]]
    // CHECK: %[[LID_PAD:.*]] = ub.poison : f16
    // CHECK: %[[LID_TILE:.*]] = vector.transfer_read %[[LID_PTR]]{{.*}}, %[[LID_PAD]] : memref<?x?xf16, strided<[?, 1]>>, vector<4x2xf16>
    // CHECK-NOT: permutation_map
    %tile, %tok = load_view_tko weak %pv[%c0, %c1] : partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> tile<4x2xf16>, token
  }

  // --- load with swizzled dim_map and zero padding ---
  // CHECK-LABEL: gpu.func @test_load_swizzled_with_padding
  // CHECK-SAME: %[[LSW_UPTR:[a-zA-Z0-9_]+]]: memref<*xf16>
  entry @test_load_swizzled_with_padding(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %m: !cuda_tile.tile<i32>, %n: !cuda_tile.tile<i32>, %s: !cuda_tile.tile<i32>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %tv = make_tensor_view %p, shape = [%m, %n], strides = [%s, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %pv = make_partition_view %tv : partition_view<tile=(4x2), padding_value = zero, tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
    // CHECK: %[[LSW_PTR:.*]] = memref.reinterpret_cast %[[LSW_UPTR]]
    // CHECK: %[[LSW_PAD:.*]] = arith.constant 0.000000e+00 : f16
    // CHECK: %[[LSW_TILE:.*]] = vector.transfer_read %[[LSW_PTR]][%{{.*}}, %{{.*}}], %[[LSW_PAD]] {permutation_map = #{{.*}}} : memref<?x?xf16, strided<[?, 1]>>, vector<4x2xf16>
    %tile, %tok = load_view_tko weak %pv[%c0, %c1] : partition_view<tile=(4x2), padding_value = zero, tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<4x2xf16>, token
  }

  // --- store with dynamic memref, identity dim_map ---
  // CHECK-LABEL: gpu.func @test_store_identity
  // CHECK-SAME: %[[STI_UPTR:[a-zA-Z0-9_]+]]: memref<*xf16>
  entry @test_store_identity(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %m: !cuda_tile.tile<i32>, %n: !cuda_tile.tile<i32>, %s: !cuda_tile.tile<i32>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %tile = constant <f16: 1.000000e+00> : tile<4x2xf16>
    %tv = make_tensor_view %p, shape = [%m, %n], strides = [%s, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %pv = make_partition_view %tv : partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>
    // CHECK: %[[STI_TILE:.*]] = arith.constant 1.000000e+00 : f16
    // CHECK: %[[STI_BCAST:.*]] = vector.broadcast %[[STI_TILE]] : f16 to vector<4x2xf16>
    // CHECK: %[[STI_PTR:.*]] = memref.reinterpret_cast %[[STI_UPTR]]
    // CHECK: vector.transfer_write %[[STI_BCAST]], %[[STI_PTR]]
    // CHECK-NOT: permutation_map
    %tok = store_view_tko weak %tile, %pv[%c0, %c1] : tile<4x2xf16>, partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> token
  }

  // --- store with swizzled dim_map ---
  // CHECK-LABEL: gpu.func @test_store_swizzled
  // CHECK-SAME: %[[STS_UPTR:[a-zA-Z0-9_]+]]: memref<*xf16>
  entry @test_store_swizzled(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %m: !cuda_tile.tile<i32>, %n: !cuda_tile.tile<i32>, %s: !cuda_tile.tile<i32>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %tile = constant <f16: 1.000000e+00> : tile<4x2xf16>
    %tv = make_tensor_view %p, shape = [%m, %n], strides = [%s, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %pv = make_partition_view %tv : partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
    // CHECK: %[[STS_TILE:.*]] = arith.constant 1.000000e+00 : f16
    // CHECK: %[[STS_BCAST:.*]] = vector.broadcast %[[STS_TILE]] : f16 to vector<4x2xf16>
    // CHECK: %[[STS_PTR:.*]] = memref.reinterpret_cast %[[STS_UPTR]]
    // CHECK: vector.transfer_write %[[STS_BCAST]], %[[STS_PTR]][%{{.*}}, %{{.*}}] {permutation_map = #{{.*}}} : vector<4x2xf16>, memref<?x?xf16, strided<[?, 1]>>
    %tok = store_view_tko weak %tile, %pv[%c0, %c1] : tile<4x2xf16>, partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> token
  }
}