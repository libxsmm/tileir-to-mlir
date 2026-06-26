// RUN: cudatile-to-mlir --convert-cuda-tile-to-mlir %s | FileCheck %s
// RUN: cudatile-to-mlir --convert-cuda-tile-to-mlir %s | mlir-opt --loop-invariant-code-motion -canonicalize -cse > /dev/null

// Additional tests that complement cuda_tile_ir_ops.mlir.
// Focuses on: 2D array constants, vector iter_args, assume chaining,
// additional cmp predicates, dim_map swizzling, dynamic memref load/store.

// CHECK-LABEL: gpu.module @m {
cuda_tile.module @m {

  // --- global/get_global edge cases ---
  // CHECK: memref.global @g_f32_aligned : memref<4xf32> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> {alignment = 256 : i64}
  global @g_f32_aligned alignment = 256 <f32: [1.0, 2.0, 3.0, 4.0]> : tile<4xf32>

  // CHECK: memref.global @g_i32_single : memref<1xi32> = dense<7>
  global @g_i32_single <i32: [7]> : tile<1xi32>

  // CHECK-LABEL: gpu.func @test_get_global_edges
  entry @test_get_global_edges() {
    // CHECK: %[[GF:.*]] = memref.get_global @g_f32_aligned : memref<4xf32>
    // CHECK: %[[PF:.*]] = memref.cast %[[GF]] : memref<4xf32> to memref<*xf32>
    %pf = get_global @g_f32_aligned : tile<ptr<f32>>

    // CHECK: %[[GI:.*]] = memref.get_global @g_i32_single : memref<1xi32>
    // CHECK: %[[PI:.*]] = memref.cast %[[GI]] : memref<1xi32> to memref<*xi32>
    %pi = get_global @g_i32_single : tile<ptr<i32>>
    return
  }

  // --- alloca edge cases ---
  // Covers a single-element allocation at the element's natural alignment, a
  // wide allocation with large alignment, a non-f32 element type, and a
  // zero-length allocation.
  // CHECK-LABEL: gpu.func @test_alloca_edges
  entry @test_alloca_edges() {
    // CHECK: %[[A1:.*]] = memref.alloca() {alignment = 2 : i64} : memref<1xf16>
    // CHECK: %[[P1:.*]] = memref.cast %[[A1]] : memref<1xf16> to memref<*xf16>
    %single = alloca num_elem = 1, alignment = 2 : tile<ptr<f16>>

    // CHECK: %[[A2:.*]] = memref.alloca() {alignment = 256 : i64} : memref<1024xi32>
    // CHECK: %[[P2:.*]] = memref.cast %[[A2]] : memref<1024xi32> to memref<*xi32>
    %wide = alloca num_elem = 1024, alignment = 256 : tile<ptr<i32>>

    // CHECK: %[[A3:.*]] = memref.alloca() {alignment = 8 : i64} : memref<0xf64>
    // CHECK: %[[P3:.*]] = memref.cast %[[A3]] : memref<0xf64> to memref<*xf64>
    %empty = alloca num_elem = 0, alignment = 8 : tile<ptr<f64>>
    return
  }

  // --- ptr_to_ptr ---
  // CHECK-LABEL: gpu.func @test_ptr_to_ptr_example
  // CHECK-SAME: %[[PTP_IN:[a-zA-Z0-9_]+]]: memref<*xf32>
  entry @test_ptr_to_ptr_example(%source: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // CHECK-NOT: memref.cast
    %cast = ptr_to_ptr %source : tile<ptr<f32>> -> tile<ptr<f32>>
    return
  }

  // --- ptr_to_ptr edge cases ---
  // Covers chaining with get_global and use as a make_tensor_view base.
  // CHECK-LABEL: gpu.func @test_ptr_to_ptr_get_global_chain
  entry @test_ptr_to_ptr_get_global_chain() {
    // CHECK: %[[PTPG_G:.*]] = memref.get_global @g_f32_aligned : memref<4xf32>
    // CHECK: %[[PTPG_P:.*]] = memref.cast %[[PTPG_G]] : memref<4xf32> to memref<*xf32>
    %p = get_global @g_f32_aligned : tile<ptr<f32>>
    %q = ptr_to_ptr %p : tile<ptr<f32>> -> tile<ptr<f32>>
    // CHECK-NOT: memref.cast {{.*}} : memref<\*xf32> to memref<\*xf32>
    // CHECK: %[[PTPG_V:.*]] = memref.reinterpret_cast %[[PTPG_P]] to offset: [%{{.*}}], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
    %tv = make_tensor_view %q, shape = [4], strides = [1] : tensor_view<4xf32, strides=[1]>
    return
  }

  // CHECK-LABEL: gpu.func @test_ptr_to_ptr_dynamic_base
  // CHECK-SAME: %[[PTPD_BASE:[a-zA-Z0-9_]+]]: memref<*xf16>, %[[PTPD_M:[a-zA-Z0-9_]+]]: i32, %[[PTPD_S:[a-zA-Z0-9_]+]]: i32
  entry @test_ptr_to_ptr_dynamic_base(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %m: !cuda_tile.tile<i32>, %s: !cuda_tile.tile<i32>) {
    // CHECK-NOT: memref.cast
    %q = ptr_to_ptr %p : tile<ptr<f16>> -> tile<ptr<f16>>
    // CHECK: %[[PTPD_MI:.*]] = arith.index_cast %[[PTPD_M]] : i32 to index
    // CHECK: %[[PTPD_SI:.*]] = arith.index_cast %[[PTPD_S]] : i32 to index
    // CHECK: %[[PTPD_V:.*]] = memref.reinterpret_cast %[[PTPD_BASE]] to offset: [0], sizes: [%[[PTPD_MI]], 8], strides: [%[[PTPD_SI]], 1] : memref<*xf16> to memref<?x8xf16, strided<[?, 1], offset: ?>>
    %tv = make_tensor_view %q, shape = [%m, 8], strides = [%s, 1] : tile<i32> -> tensor_view<?x8xf16, strides=[?,1]>
    return
  }

  // --- scalar offset on a pointer tile ---
  // CHECK-LABEL: gpu.func @test_offset_scalar_i32
  // CHECK-SAME: %[[OFS_BASE:[a-zA-Z0-9_]+]]: memref<*xf16>, %[[OFS_O:[a-zA-Z0-9_]+]]: i32
  entry @test_offset_scalar_i32(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %o: !cuda_tile.tile<i32>) {
    // CHECK: %[[OFS_I:.*]] = arith.index_cast %[[OFS_O]] : i32 to index
    // CHECK: %[[OFS_R:.*]] = memref.reinterpret_cast %[[OFS_BASE]] to offset: [%[[OFS_I]]], sizes: [1], strides: [1] : memref<*xf16> to memref<1xf16, strided<[1], offset: ?>>
    // CHECK: memref.cast %[[OFS_R]] : memref<1xf16, strided<[1], offset: ?>> to memref<*xf16>
    %q = offset %p, %o : tile<ptr<f16>>, tile<i32> -> tile<ptr<f16>>
    return
  }

  // --- scalar offset with i64 index width ---
  // CHECK-LABEL: gpu.func @test_offset_scalar_i64
  // CHECK-SAME: %[[OFS64_BASE:[a-zA-Z0-9_]+]]: memref<*xf32>, %[[OFS64_O:[a-zA-Z0-9_]+]]: i64
  entry @test_offset_scalar_i64(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %o: !cuda_tile.tile<i64>) {
    // CHECK: %[[OFS64_I:.*]] = arith.index_cast %[[OFS64_O]] : i64 to index
    // CHECK: memref.reinterpret_cast %[[OFS64_BASE]] to offset: [%[[OFS64_I]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    %q = offset %p, %o : tile<ptr<f32>>, tile<i64> -> tile<ptr<f32>>
    return
  }

  // --- scalar offset feeding make_tensor_view (composition) ---
  // The view's reinterpret_cast must recover and re-apply the offset-shifted
  // memref's descriptor offset, because reinterpret_cast offsets are absolute:
  // using offset 0 would discard the pointer shift introduced by the offset op.
  // CHECK-LABEL: gpu.func @test_offset_then_view
  // CHECK-SAME: %[[OTV_BASE:[a-zA-Z0-9_]+]]: memref<*xf16>, %[[OTV_O:[a-zA-Z0-9_]+]]: i32, %[[OTV_M:[a-zA-Z0-9_]+]]: i32
  entry @test_offset_then_view(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %o: !cuda_tile.tile<i32>, %m: !cuda_tile.tile<i32>) {
    // CHECK: %[[OTV_I:.*]] = arith.index_cast %[[OTV_O]] : i32 to index
    // CHECK: %[[OTV_R:.*]] = memref.reinterpret_cast %[[OTV_BASE]] to offset: [%[[OTV_I]]], sizes: [1], strides: [1] : memref<*xf16> to memref<1xf16, strided<[1], offset: ?>>
    // CHECK: %[[OTV_CAST:.*]] = memref.cast %[[OTV_R]] : memref<1xf16, strided<[1], offset: ?>> to memref<*xf16>
    %q = offset %p, %o : tile<ptr<f16>>, tile<i32> -> tile<ptr<f16>>
    // CHECK: %[[OTV_MI:.*]] = arith.index_cast %[[OTV_M]] : i32 to index
    // CHECK: memref.reinterpret_cast %[[OTV_CAST]] to offset: [%{{.*}}], sizes: [%[[OTV_MI]], 8], strides: [8, 1] : memref<*xf16> to memref<?x8xf16, strided<[8, 1], offset: ?>>
    %tv = make_tensor_view %q, shape = [%m, 8], strides = [8, 1] : tile<i32> -> tensor_view<?x8xf16, strides=[8,1]>
    return
  }

  // --- offset across a ptr_to_ptr chain (asserts unranked-only path) ---
  // CHECK-LABEL: gpu.func @test_offset_after_ptr_to_ptr
  // CHECK-SAME: %[[OPP_BASE:[a-zA-Z0-9_]+]]: memref<*xf16>, %[[OPP_O:[a-zA-Z0-9_]+]]: i32
  entry @test_offset_after_ptr_to_ptr(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %o: !cuda_tile.tile<i32>) {
    %q = ptr_to_ptr %p : tile<ptr<f16>> -> tile<ptr<f16>>
    // CHECK: %[[OPP_I:.*]] = arith.index_cast %[[OPP_O]] : i32 to index
    // CHECK: memref.reinterpret_cast %[[OPP_BASE]] to offset: [%[[OPP_I]]], sizes: [1], strides: [1]
    %r = offset %q, %o : tile<ptr<f16>>, tile<i32> -> tile<ptr<f16>>
    return
  }

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

  // --- get_tensor_shape with dynamic tensor_view dims ---
  // CHECK-LABEL: gpu.func @test_get_tensor_shape_dynamic
  // CHECK-SAME: %[[GTSD_UPTR:[a-zA-Z0-9_]+]]: memref<*xf16>
  entry @test_get_tensor_shape_dynamic(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %m: !cuda_tile.tile<i32>, %n: !cuda_tile.tile<i32>, %s: !cuda_tile.tile<i32>) {
    %tv = make_tensor_view %p, shape = [%m, %n], strides = [%s, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    // CHECK: %[[GTSD_VIEW:.*]] = memref.reinterpret_cast %[[GTSD_UPTR]]{{.*}} : memref<*xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
    // CHECK: %[[GTSD_D0:.*]] = memref.dim %[[GTSD_VIEW]], {{.*}}0
    // CHECK: %[[GTSD_R0:.*]] = arith.index_castui %[[GTSD_D0]] : index to i32
    // CHECK: %[[GTSD_D1:.*]] = memref.dim %[[GTSD_VIEW]], {{.*}}1
    // CHECK: %[[GTSD_R1:.*]] = arith.index_castui %[[GTSD_D1]] : index to i32
    %d0, %d1 = get_tensor_shape %tv : tensor_view<?x?xf16, strides=[?,1]> -> tile<i32>
    return
  }

  // --- get_tensor_shape with mixed static/dynamic dims ---
  // CHECK-LABEL: gpu.func @test_get_tensor_shape_mixed
  // CHECK-SAME: %[[GTSM_UPTR:[a-zA-Z0-9_]+]]: memref<*xf16>
  entry @test_get_tensor_shape_mixed(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %n: !cuda_tile.tile<i32>, %s: !cuda_tile.tile<i32>) {
    %tv = make_tensor_view %p, shape = [64, %n], strides = [%s, 1] : tile<i32> -> tensor_view<64x?xf16, strides=[?,1]>
    // CHECK: %[[GTSM_VIEW:.*]] = memref.reinterpret_cast %[[GTSM_UPTR]]{{.*}} : memref<*xf16> to memref<64x?xf16, strided<[?, 1], offset: ?>>
    // CHECK: %[[GTSM_C64:.*]] = arith.constant 64 : index
    // CHECK: %[[GTSM_R0:.*]] = arith.index_castui %[[GTSM_C64]] : index to i64
    // CHECK: %[[GTSM_D1:.*]] = memref.dim %[[GTSM_VIEW]], {{.*}}1
    // CHECK: %[[GTSM_R1:.*]] = arith.index_castui %[[GTSM_D1]] : index to i64
    %d0, %d1 = get_tensor_shape %tv : tensor_view<64x?xf16, strides=[?,1]> -> tile<i64>
    return
  }

  // --- iota edge cases ---
  // CHECK-LABEL: gpu.func @test_iota_i8
  entry @test_iota_i8() {
    // CHECK: %[[I8_STEP:.*]] = vector.step : vector<4xindex>
    // CHECK: %[[I8_IOTA:.*]] = arith.index_castui %[[I8_STEP]] : vector<4xindex> to vector<4xi8>
    %idx = iota : tile<4xi8>
    return
  }

  // CHECK-LABEL: gpu.func @test_iota_i64
  entry @test_iota_i64() {
    // CHECK: %[[I64_STEP:.*]] = vector.step : vector<16xindex>
    // CHECK: %[[I64_IOTA:.*]] = arith.index_castui %[[I64_STEP]] : vector<16xindex> to vector<16xi64>
    %idx = iota : tile<16xi64>
    return
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
    // CHECK: %[[LID_TILE:.*]] = vector.transfer_read %[[LID_PTR]]{{.*}}, %[[LID_PAD]] : memref<?x?xf16, strided<[?, 1], offset: ?>>, vector<4x2xf16>
    // CHECK-NOT: permutation_map
    %tile, %tok = load_view_tko weak %pv[%c0, %c1] : partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> tile<4x2xf16>, !cuda_tile.token
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
    // CHECK: %[[LSW_TILE:.*]] = vector.transfer_read %[[LSW_PTR]][%{{.*}}, %{{.*}}], %[[LSW_PAD]] {permutation_map = #{{.*}}} : memref<?x?xf16, strided<[?, 1], offset: ?>>, vector<4x2xf16>
    %tile, %tok = load_view_tko weak %pv[%c0, %c1] : partition_view<tile=(4x2), padding_value = zero, tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<4x2xf16>, !cuda_tile.token
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
    // CHECK: %[[STI_BCAST:.*]] = arith.constant dense<1.000000e+00> : vector<4x2xf16>
    // CHECK: %[[STI_PTR:.*]] = memref.reinterpret_cast %[[STI_UPTR]]
    // CHECK: vector.transfer_write %[[STI_BCAST]], %[[STI_PTR]]
    // CHECK-NOT: permutation_map
    %tok = store_view_tko weak %tile, %pv[%c0, %c1] : tile<4x2xf16>, partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> !cuda_tile.token
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
    // CHECK: %[[STS_BCAST:.*]] = arith.constant dense<1.000000e+00> : vector<4x2xf16>
    // CHECK: %[[STS_PTR:.*]] = memref.reinterpret_cast %[[STS_UPTR]]
    // CHECK: vector.transfer_write %[[STS_BCAST]], %[[STS_PTR]][%{{.*}}, %{{.*}}] {permutation_map = #{{.*}}} : vector<4x2xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
    %tok = store_view_tko weak %tile, %pv[%c0, %c1] : tile<4x2xf16>, partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> !cuda_tile.token
  }

  // --- scalar load_ptr_tko: lowers directly to reinterpret_cast + memref.load ---
  // CHECK-LABEL: gpu.func @test_scalar_load_ptr
  // CHECK-SAME: %[[SLP_UPTR:[a-zA-Z0-9_]+]]: memref<*xf32>
  entry @test_scalar_load_ptr(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // CHECK: %[[SLP_R0:.*]] = memref.reinterpret_cast %[[SLP_UPTR]] to offset: [0], sizes: [], strides: [] : memref<*xf32> to memref<f32, strided<[], offset: ?>>
    // CHECK: %[[SLP_VAL:.*]] = memref.load %[[SLP_R0]][] : memref<f32, strided<[], offset: ?>>
    // CHECK-NOT: load_ptr_tko
    %v, %tok = load_ptr_tko weak %p : tile<ptr<f32>> -> tile<f32>, !cuda_tile.token
    return
  }

  // --- scalar store_ptr_tko: lowers directly to reinterpret_cast + memref.store ---
  // CHECK-LABEL: gpu.func @test_scalar_store_ptr
  // CHECK-SAME: %[[SSP_UPTR:[a-zA-Z0-9_]+]]: memref<*xf32>
  entry @test_scalar_store_ptr(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %v = constant <f32: 1.000000e+00> : tile<f32>
    // CHECK: %[[SSP_VAL:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK: %[[SSP_R0:.*]] = memref.reinterpret_cast %[[SSP_UPTR]] to offset: [0], sizes: [], strides: [] : memref<*xf32> to memref<f32, strided<[], offset: ?>>
    // CHECK: memref.store %[[SSP_VAL]], %[[SSP_R0]][] : memref<f32, strided<[], offset: ?>>
    // CHECK-NOT: store_ptr_tko
    %tok = store_ptr_tko weak %p, %v : tile<ptr<f32>>, tile<f32> -> !cuda_tile.token
    return
  }

  // --- reduce (1D -> scalar, mulf) ---
  // CHECK-LABEL: gpu.func @test_reduce_mulf_1d
  entry @test_reduce_mulf_1d() {
    // CHECK: %[[RMUL_IN:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    %input = constant <f32: 1.0> : tile<4xf32>
    // CHECK: %[[RMUL_ACC:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK: %[[RMUL_R:.*]] = vector.reduction <mul>, %[[RMUL_IN]], %[[RMUL_ACC]] : vector<4xf32> into f32
    %0 = reduce %input dim=0 identities=[1.000000e+00 : f32] : tile<4xf32> -> tile<f32>
      (%input_arg: tile<f32>, %input_accum: tile<f32>) {
        %mul_result = mulf %input_arg, %input_accum : tile<f32>
        yield %mul_result : tile<f32>
      }
    return
  }

  // --- reduce (1D -> scalar, maxf with propagate_nan) ---
  // CHECK-LABEL: gpu.func @test_reduce_maxf_1d
  entry @test_reduce_maxf_1d() {
    // CHECK: %[[RMAX_IN:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
    %input = constant <f32: 0.0> : tile<8xf32>
    // CHECK: %[[RMAX_ACC:.*]] = arith.constant 0xFF800000 : f32
    // CHECK: %[[RMAX_R:.*]] = vector.reduction <maximumf>, %[[RMAX_IN]], %[[RMAX_ACC]] : vector<8xf32> into f32
    %0 = reduce %input dim=0 identities=[0xFF800000 : f32] : tile<8xf32> -> tile<f32>
      (%input_arg: tile<f32>, %input_accum: tile<f32>) {
        %max_result = maxf %input_arg, %input_accum propagate_nan : tile<f32>
        yield %max_result : tile<f32>
      }
    return
  }

  // --- reduce (1D -> scalar, minf without propagate_nan) ---
  // CHECK-LABEL: gpu.func @test_reduce_minf_1d
  entry @test_reduce_minf_1d() {
    // CHECK: %[[RMIN_IN:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
    %input = constant <f32: 0.0> : tile<8xf32>
    // CHECK: %[[RMIN_ACC:.*]] = arith.constant 0x7F800000 : f32
    // CHECK: %[[RMIN_R:.*]] = vector.reduction <minnumf>, %[[RMIN_IN]], %[[RMIN_ACC]] : vector<8xf32> into f32
    %0 = reduce %input dim=0 identities=[0x7F800000 : f32] : tile<8xf32> -> tile<f32>
      (%input_arg: tile<f32>, %input_accum: tile<f32>) {
        %min_result = minf %input_arg, %input_accum : tile<f32>
        yield %min_result : tile<f32>
      }
    return
  }

  // --- reduce (1D -> scalar, addi) ---
  // CHECK-LABEL: gpu.func @test_reduce_addi_1d
  entry @test_reduce_addi_1d() {
    // CHECK: %[[RADDI_IN:.*]] = arith.constant dense<0> : vector<8xi32>
    %input = constant <i32: 0> : tile<8xi32>
    // CHECK: %[[RADDI_ACC:.*]] = arith.constant 0 : i32
    // CHECK: %[[RADDI_R:.*]] = vector.reduction <add>, %[[RADDI_IN]], %[[RADDI_ACC]] : vector<8xi32> into i32
    %0 = reduce %input dim=0 identities=[0 : i32] : tile<8xi32> -> tile<i32>
      (%input_arg: tile<i32>, %input_accum: tile<i32>) {
        %add_result = addi %input_arg, %input_accum : tile<i32>
        yield %add_result : tile<i32>
      }
    return
  }

  // --- reshape (vector -> scalar via vector.extract) ---
  // CHECK-LABEL: gpu.func @test_reshape_vector_to_scalar
  entry @test_reshape_vector_to_scalar() {
    // CHECK: %[[RV2S_IN:.*]] = arith.constant dense<7> : vector<1x1xi32>
    %t = constant <i32: [[7]]> : tile<1x1xi32>
    // CHECK: %[[RV2S_R:.*]] = vector.extract %[[RV2S_IN]][0, 0] : i32 from vector<1x1xi32>
    %s = reshape %t : tile<1x1xi32> -> tile<i32>
    return
  }

  // --- reshape (scalar -> scalar identity) ---
  // CHECK-LABEL: gpu.func @test_reshape_scalar_to_scalar
  entry @test_reshape_scalar_to_scalar() {
    // CHECK: %[[S:.*]] = arith.constant 7 : i32
    %t = constant <i32: 7> : tile<i32>
    // CHECK-NOT: vector.
    // CHECK-NOT: reshape
    %r = reshape %t : tile<i32> -> tile<i32>
    return
  }

  // --- maxf with propagate_nan -> arith.maximumf ---
  // CHECK-LABEL: gpu.func @test_maxf_propagate_nan
  entry @test_maxf_propagate_nan() {
    // CHECK: %[[MAXP_A:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    %a = constant <f32: 0.0> : tile<4xf32>
    // CHECK: %[[MAXP_B:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    %b = constant <f32: 1.0> : tile<4xf32>
    // CHECK: %[[MAXP_R:.*]] = arith.maximumf %[[MAXP_A]], %[[MAXP_B]] : vector<4xf32>
    %r = maxf %a, %b propagate_nan : tile<4xf32>
    return
  }

  // --- minf with propagate_nan -> arith.minimumf ---
  // CHECK-LABEL: gpu.func @test_minf_propagate_nan
  entry @test_minf_propagate_nan() {
    // CHECK: %[[MINP_A:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    %a = constant <f32: 0.0> : tile<4xf32>
    // CHECK: %[[MINP_B:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    %b = constant <f32: 1.0> : tile<4xf32>
    // CHECK: %[[MINP_R:.*]] = arith.minimumf %[[MINP_A]], %[[MINP_B]] : vector<4xf32>
    %r = minf %a, %b propagate_nan : tile<4xf32>
    return
  }

  // --- scan: 1D inclusive add (initial_value is rank-0 vector) ---
  // CHECK-LABEL: gpu.func @test_scan_addf_1d
  entry @test_scan_addf_1d() {
    // CHECK: %[[SADD1_IN:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
    %input = constant <f32: 0.0> : tile<8xf32>
    // CHECK: %[[SADD1_INIT:.*]] = arith.constant dense<0.000000e+00> : vector<f32>
    // CHECK: %[[SADD1_R:.*]], %{{.*}} = vector.scan <add>, %[[SADD1_IN]], %[[SADD1_INIT]] {inclusive = true, reduction_dim = 0 : i64} : vector<8xf32>, vector<f32>
    %0 = scan %input dim=0 reverse=false identities=[0.000000e+00 : f32] : tile<8xf32> -> tile<8xf32>
      (%a: tile<f32>, %b: tile<f32>) {
        %s = addf %a, %b : tile<f32>
        yield %s : tile<f32>
      }
    return
  }

  // --- scan: 2D inclusive add along dim 0 ---
  // CHECK-LABEL: gpu.func @test_scan_addf_2d_dim0
  entry @test_scan_addf_2d_dim0() {
    // CHECK: %[[SADD2_IN:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
    %input = constant <f32: 0.0> : tile<8x16xf32>
    // CHECK: %[[SADD2_INIT:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
    // CHECK: %[[SADD2_R:.*]], %{{.*}} = vector.scan <add>, %[[SADD2_IN]], %[[SADD2_INIT]] {inclusive = true, reduction_dim = 0 : i64} : vector<8x16xf32>, vector<16xf32>
    %0 = scan %input dim=0 reverse=false identities=[0.000000e+00 : f32] : tile<8x16xf32> -> tile<8x16xf32>
      (%a: tile<f32>, %b: tile<f32>) {
        %s = addf %a, %b : tile<f32>
        yield %s : tile<f32>
      }
    return
  }

  // --- scan: 1D integer add ---
  // CHECK-LABEL: gpu.func @test_scan_addi_1d
  entry @test_scan_addi_1d() {
    // CHECK: %[[SADDI_IN:.*]] = arith.constant dense<0> : vector<8xi32>
    %input = constant <i32: 0> : tile<8xi32>
    // CHECK: %[[SADDI_INIT:.*]] = arith.constant dense<0> : vector<i32>
    // CHECK: %[[SADDI_R:.*]], %{{.*}} = vector.scan <add>, %[[SADDI_IN]], %[[SADDI_INIT]] {inclusive = true, reduction_dim = 0 : i64} : vector<8xi32>, vector<i32>
    %0 = scan %input dim=0 reverse=false identities=[0 : i32] : tile<8xi32> -> tile<8xi32>
      (%a: tile<i32>, %b: tile<i32>) {
        %s = addi %a, %b : tile<i32>
        yield %s : tile<i32>
      }
    return
  }

  // --- scan: 2D max along dim 1 (propagate_nan -> maximumf) ---
  // CHECK-LABEL: gpu.func @test_scan_maxf_propagate_nan
  entry @test_scan_maxf_propagate_nan() {
    // CHECK: %[[SMAX_IN:.*]] = arith.constant dense<0.000000e+00> : vector<4x8xf32>
    %input = constant <f32: 0.0> : tile<4x8xf32>
    // CHECK: %[[SMAX_INIT:.*]] = arith.constant dense<0xFF800000> : vector<4xf32>
    // CHECK: %[[SMAX_R:.*]], %{{.*}} = vector.scan <maximumf>, %[[SMAX_IN]], %[[SMAX_INIT]] {inclusive = true, reduction_dim = 1 : i64} : vector<4x8xf32>, vector<4xf32>
    %0 = scan %input dim=1 reverse=false identities=[0xFF800000 : f32] : tile<4x8xf32> -> tile<4x8xf32>
      (%a: tile<f32>, %b: tile<f32>) {
        %s = maxf %a, %b propagate_nan : tile<f32>
        yield %s : tile<f32>
      }
    return
  }

  // --- scan: signed integer min ---
  // CHECK-LABEL: gpu.func @test_scan_minsi_1d
  entry @test_scan_minsi_1d() {
    // CHECK: %[[SMINSI_IN:.*]] = arith.constant dense<0> : vector<8xi32>
    %input = constant <i32: 0> : tile<8xi32>
    // CHECK: %[[SMINSI_INIT:.*]] = arith.constant dense<2147483647> : vector<i32>
    // CHECK: %[[SMINSI_R:.*]], %{{.*}} = vector.scan <minsi>, %[[SMINSI_IN]], %[[SMINSI_INIT]] {inclusive = true, reduction_dim = 0 : i64} : vector<8xi32>, vector<i32>
    %0 = scan %input dim=0 reverse=false identities=[2147483647 : i32] : tile<8xi32> -> tile<8xi32>
      (%a: tile<i32>, %b: tile<i32>) {
        %s = mini %a, %b signed : tile<i32>
        yield %s : tile<i32>
      }
    return
  }

  // --- select: scalar (rank-0) cond and values ---
  // CHECK-LABEL: gpu.func @test_select_scalar
  entry @test_select_scalar() {
    // CHECK: %[[SELS_BID:.*]] = gpu.block_id x
    // CHECK: %[[SELS_BIDI:.*]] = arith.index_cast %[[SELS_BID]] : index to i32
    %bidx, %bidy, %bidz = get_tile_block_id : tile<i32>
    // CHECK: %[[SELS_ZERO:.*]] = arith.constant 0 : i32
    %zero = constant <i32: 0> : tile<i32>
    // Derive a non-constant i1 so arith.select is not folded away.
    // CHECK: %[[SELS_COND:.*]] = arith.cmpi eq, %[[SELS_BIDI]], %[[SELS_ZERO]] : i32
    %cond = cmpi equal %bidx, %zero, signed : tile<i32> -> tile<i1>
    // CHECK: %[[SELS_T:.*]] = arith.constant 42 : i32
    %t = constant <i32: 42> : tile<i32>
    // CHECK: %[[SELS_F:.*]] = arith.constant 7 : i32
    %f = constant <i32: 7> : tile<i32>
    // CHECK: %[[SELS_R:.*]] = arith.select %[[SELS_COND]], %[[SELS_T]], %[[SELS_F]] : i32
    %r = select %cond, %t, %f : tile<i1>, tile<i32>
    return
  }

  // --- select: 2D integer ---
  // CHECK-LABEL: gpu.func @test_select_2d_i32
  entry @test_select_2d_i32() {
    // Non-splat cond avoids arith.select folding to a single operand.
    // CHECK: %[[SEL2D_COND:.*]] = arith.constant dense<{{\[}}[true, false, true, false], [false, true, false, true]]> : vector<2x4xi1>
    %cond = constant <i1: [[1, 0, 1, 0], [0, 1, 0, 1]]> : tile<2x4xi1>
    // CHECK: %[[SEL2D_T:.*]] = arith.constant dense<0> : vector<2x4xi32>
    %t = constant <i32: 0> : tile<2x4xi32>
    // CHECK: %[[SEL2D_F:.*]] = arith.constant dense<1> : vector<2x4xi32>
    %f = constant <i32: 1> : tile<2x4xi32>
    // CHECK: %[[SEL2D_R:.*]] = arith.select %[[SEL2D_COND]], %[[SEL2D_T]], %[[SEL2D_F]] : vector<2x4xi1>, vector<2x4xi32>
    %r = select %cond, %t, %f : tile<2x4xi1>, tile<2x4xi32>
    return
  }

  // --- select: 1D f16 (different element type) ---
  // CHECK-LABEL: gpu.func @test_select_1d_f16
  entry @test_select_1d_f16() {
    // CHECK: %[[SEL16_COND:.*]] = arith.constant dense<[false, true, false, true, false, true, false, true]> : vector<8xi1>
    %cond = constant <i1: [0, 1, 0, 1, 0, 1, 0, 1]> : tile<8xi1>
    // CHECK: %[[SEL16_T:.*]] = arith.constant dense<1.000000e+00> : vector<8xf16>
    %t = constant <f16: 1.0> : tile<8xf16>
    // CHECK: %[[SEL16_F:.*]] = arith.constant dense<2.000000e+00> : vector<8xf16>
    %f = constant <f16: 2.0> : tile<8xf16>
    // CHECK: %[[SEL16_R:.*]] = arith.select %[[SEL16_COND]], %[[SEL16_T]], %[[SEL16_F]] : vector<8xi1>, vector<8xf16>
    %r = select %cond, %t, %f : tile<8xi1>, tile<8xf16>
    return
  }

  // --- conversion ops (no mlirExamples in Ops.td; edge coverage lives here) ---

  // CHECK-LABEL: gpu.func @test_conv_bitcast
  entry @test_conv_bitcast() {
    // CHECK: %[[BC_IN:.*]] = arith.constant dense<1> : vector<4xi32>
    %x = constant <i32: 1> : tile<4xi32>
    // CHECK: %[[BC_R:.*]] = arith.bitcast %[[BC_IN]] : vector<4xi32> to vector<4xf32>
    %r = bitcast %x : tile<4xi32> -> tile<4xf32>
    return
  }

  // CHECK-LABEL: gpu.func @test_conv_bitcast_scalar
  entry @test_conv_bitcast_scalar() {
    // CHECK: %[[BCS_IN:.*]] = arith.constant 1065353216 : i32
    %x = constant <i32: 1065353216> : tile<i32>
    // CHECK: %[[BCS_R:.*]] = arith.bitcast %[[BCS_IN]] : i32 to f32
    %r = bitcast %x : tile<i32> -> tile<f32>
    return
  }

  // CHECK-LABEL: gpu.func @test_conv_exti
  entry @test_conv_exti() {
    // CHECK: %[[EXT_IN:.*]] = arith.constant dense<5> : vector<4xi8>
    %x = constant <i8: 5> : tile<4xi8>
    // CHECK: %[[EXT_S:.*]] = arith.extsi %[[EXT_IN]] : vector<4xi8> to vector<4xi16>
    %s = exti %x signed : tile<4xi8> -> tile<4xi16>
    // CHECK: %[[EXT_U:.*]] = arith.extui %[[EXT_IN]] : vector<4xi8> to vector<4xi16>
    %u = exti %x unsigned : tile<4xi8> -> tile<4xi16>
    return
  }

  // CHECK-LABEL: gpu.func @test_conv_exti_scalar
  entry @test_conv_exti_scalar() {
    // CHECK: %[[EXTS_IN:.*]] = arith.constant 5 : i8
    %x = constant <i8: 5> : tile<i8>
    // CHECK: %[[EXTS_S:.*]] = arith.extsi %[[EXTS_IN]] : i8 to i16
    %s = exti %x signed : tile<i8> -> tile<i16>
    // CHECK: %[[EXTS_U:.*]] = arith.extui %[[EXTS_IN]] : i8 to i16
    %u = exti %x unsigned : tile<i8> -> tile<i16>
    return
  }

  // CHECK-LABEL: gpu.func @test_conv_ftof
  entry @test_conv_ftof() {
    // CHECK: %[[FTOF_IN:.*]] = arith.constant dense<1.250000e+00> : vector<4xf32>
    %x = constant <f32: 1.25> : tile<4xf32>
    // CHECK: %[[FTOF_TR:.*]] = arith.truncf %[[FTOF_IN]] to_nearest_even : vector<4xf32> to vector<4xf16>
    %tr = ftof %x rounding<nearest_even> : tile<4xf32> -> tile<4xf16>
    // CHECK: %[[FTOF_EX:.*]] = arith.extf %[[FTOF_TR]] {{.*}}tir-dropped-rounding{{.*}} : vector<4xf16> to vector<4xf64>
    %ex = ftof %tr rounding<nearest_even> : tile<4xf16> -> tile<4xf64>
    return
  }

  // CHECK-LABEL: gpu.func @test_conv_ftoi
  entry @test_conv_ftoi() {
    // CHECK: %[[FTOI_IN:.*]] = arith.constant dense<3.500000e+00> : vector<4xf32>
    %x = constant <f32: 3.5> : tile<4xf32>
    // CHECK: %[[FTOI_S:.*]] = arith.fptosi %[[FTOI_IN]] : vector<4xf32> to vector<4xi32>
    %s = ftoi %x signed rounding<nearest_int_to_zero> : tile<4xf32> -> tile<4xi32>
    // CHECK: %[[FTOI_U:.*]] = arith.fptoui %[[FTOI_IN]] : vector<4xf32> to vector<4xi32>
    %u = ftoi %x unsigned rounding<nearest_int_to_zero> : tile<4xf32> -> tile<4xi32>
    return
  }

  // CHECK-LABEL: gpu.func @test_conv_ftoi_scalar
  entry @test_conv_ftoi_scalar() {
    // CHECK: %[[FTOIS_IN:.*]] = arith.constant 3.500000e+00 : f32
    %x = constant <f32: 3.5> : tile<f32>
    // CHECK: %[[FTOIS_S:.*]] = arith.fptosi %[[FTOIS_IN]] : f32 to i32
    %s = ftoi %x signed rounding<nearest_int_to_zero> : tile<f32> -> tile<i32>
    // CHECK: %[[FTOIS_U:.*]] = arith.fptoui %[[FTOIS_IN]] : f32 to i32
    %u = ftoi %x unsigned rounding<nearest_int_to_zero> : tile<f32> -> tile<i32>
    return
  }

  // CHECK-LABEL: gpu.func @test_conv_trunci
  entry @test_conv_trunci() {
    // CHECK: %[[TRI_IN:.*]] = arith.constant dense<42> : vector<4xi32>
    %x = constant <i32: 42> : tile<4xi32>
    // CHECK: %[[TRI_NW:.*]] = arith.trunci %[[TRI_IN]] overflow<nsw, nuw> : vector<4xi32> to vector<4xi16>
    %nw = trunci %x overflow<no_wrap> : tile<4xi32> -> tile<4xi16>
    // CHECK: %[[TRI_NSW:.*]] = arith.trunci %[[TRI_IN]] overflow<nsw> : vector<4xi32> to vector<4xi16>
    %nsw = trunci %x overflow<no_signed_wrap> : tile<4xi32> -> tile<4xi16>
    // CHECK: %[[TRI_NUW:.*]] = arith.trunci %[[TRI_IN]] overflow<nuw> : vector<4xi32> to vector<4xi16>
    %nuw = trunci %x overflow<no_unsigned_wrap> : tile<4xi32> -> tile<4xi16>
    // CHECK: %[[TRI_NONE:.*]] = arith.trunci %[[TRI_IN]] : vector<4xi32> to vector<4xi16>
    %none = trunci %x overflow<none> : tile<4xi32> -> tile<4xi16>
    return
  }

  // CHECK-LABEL: gpu.func @test_conv_itof
  entry @test_conv_itof() {
    // CHECK: %[[ITOF_IN:.*]] = arith.constant dense<7> : vector<4xi32>
    %x = constant <i32: 7> : tile<4xi32>
    // CHECK: %[[ITOF_S:.*]] = arith.sitofp %[[ITOF_IN]] : vector<4xi32> to vector<4xf32>
    %s = itof %x signed rounding<nearest_even> : tile<4xi32> -> tile<4xf32>
    // CHECK: %[[ITOF_U:.*]] = arith.uitofp %[[ITOF_IN]] : vector<4xi32> to vector<4xf32>
    %u = itof %x unsigned rounding<nearest_even> : tile<4xi32> -> tile<4xf32>
    return
  }

  // CHECK-LABEL: gpu.func @test_conv_itof_scalar
  entry @test_conv_itof_scalar() {
    // CHECK: %[[ITOFS_IN:.*]] = arith.constant 7 : i32
    %x = constant <i32: 7> : tile<i32>
    // CHECK: %[[ITOFS_S:.*]] = arith.sitofp %[[ITOFS_IN]] : i32 to f32
    %s = itof %x signed rounding<nearest_even> : tile<i32> -> tile<f32>
    // CHECK: %[[ITOFS_U:.*]] = arith.uitofp %[[ITOFS_IN]] : i32 to f32
    %u = itof %x unsigned rounding<nearest_even> : tile<i32> -> tile<f32>
    return
  }

  // --- absf ---
  // CHECK-LABEL: gpu.func @test_absf
  entry @test_absf() {
    // CHECK: %[[ABSF_IN:.*]] = arith.constant dense<[1.000000e+00, -1.000000e+00, 0.000000e+00, 2.000000e+00]> : vector<4xf32>
    %in = constant <f32: [1.0, -1.0, 0.0, 2.0]> : tile<4xf32>
    // CHECK: %[[ABSF_R:.*]] = math.absf %[[ABSF_IN]] : vector<4xf32>
    %res = absf %in : tile<4xf32>
    return
  }

  // --- absi ---
  // CHECK-LABEL: gpu.func @test_absi
  entry @test_absi() {
    // CHECK: %[[ABSI_IN:.*]] = arith.constant dense<[0, -1, 2, -3]> : vector<4xi32>
    %in = constant <i32: [0, -1, 2, -3]> : tile<4xi32>
    // CHECK: %[[ABSI_R:.*]] = math.absi %[[ABSI_IN]] : vector<4xi32>
    %res = absi %in : tile<4xi32>
    return
  }

  // --- log ---
  // CHECK-LABEL: gpu.func @test_log
  entry @test_log() {
    // CHECK: %[[LOG_IN:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %in = constant <f32: [1.0, 2.0, 3.0, 4.0]> : tile<4xf32>
    // CHECK: %[[LOG_R:.*]] = math.log %[[LOG_IN]] : vector<4xf32>
    %res = log %in : tile<4xf32>
    return
  }

  // --- tan ---
  // CHECK-LABEL: gpu.func @test_tan
  entry @test_tan() {
    // CHECK: %[[TAN_IN:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
    // CHECK: %[[TAN_R:.*]] = math.tan %[[TAN_IN]] : vector<4xf32>
    %res = tan %in : tile<4xf32>
    return
  }

  // --- sinh ---
  // CHECK-LABEL: gpu.func @test_sinh
  entry @test_sinh() {
    // CHECK: %[[SINH_IN:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
    // CHECK: %[[SINH_R:.*]] = math.sinh %[[SINH_IN]] : vector<4xf32>
    %res = sinh %in : tile<4xf32>
    return
  }

  // --- cosh ---
  // CHECK-LABEL: gpu.func @test_cosh
  entry @test_cosh() {
    // CHECK: %[[COSH_IN:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
    // CHECK: %[[COSH_R:.*]] = math.cosh %[[COSH_IN]] : vector<4xf32>
    %res = cosh %in : tile<4xf32>
    return
  }

  // --- sqrt ---
  // CHECK-LABEL: gpu.func @test_sqrt
  entry @test_sqrt() {
    // CHECK: %[[SQRT_IN:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %in = constant <f32: [1.0, 4.0, 9.0, 16.0]> : tile<4xf32>
    // CHECK: %[[SQRT_R:.*]] = math.sqrt %[[SQRT_IN]] {"tir-dropped-rounding" = "nearest_even"} : vector<4xf32>
    %res = sqrt %in : tile<4xf32>
    return
  }

    // --- sqrt with Triton-style approx + flush_to_zero ---
    // CHECK-LABEL: gpu.func @test_sqrt_approx_ftz
    entry @test_sqrt_approx_ftz() {
      // CHECK: %[[SQRTA_IN:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
      %in = constant <f32: [1.0, 4.0, 9.0, 16.0]> : tile<4xf32>
      // CHECK: %[[SQRTA_R:.*]] = math.sqrt %[[SQRTA_IN]] fastmath<afn> {"tir-dropped-flush-to-zero", "tir-dropped-rounding" = "approx"} : vector<4xf32>
      %res = sqrt %in rounding<approx> flush_to_zero : tile<4xf32>
      return
    }

  // --- andi ---
  // CHECK-LABEL: gpu.func @test_andi
  entry @test_andi() {
    // CHECK: %[[ANDI_LHS:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
    %lhs = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
    // CHECK: %[[ANDI_RHS:.*]] = arith.constant dense<[4, 5, 6, 7]> : vector<4xi32>
    %rhs = constant <i32: [4, 5, 6, 7]> : tile<4xi32>
    // CHECK: %[[ANDI_R:.*]] = arith.andi %[[ANDI_LHS]], %[[ANDI_RHS]] : vector<4xi32>
    %result = andi %lhs, %rhs : tile<4xi32>
    return
  }

  // --- ori ---
  // CHECK-LABEL: gpu.func @test_ori
  entry @test_ori() {
    // CHECK: %[[ORI_LHS:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
    %lhs = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
    // CHECK: %[[ORI_RHS:.*]] = arith.constant dense<[4, 5, 6, 7]> : vector<4xi32>
    %rhs = constant <i32: [4, 5, 6, 7]> : tile<4xi32>
    // CHECK: %[[ORI_R:.*]] = arith.ori %[[ORI_LHS]], %[[ORI_RHS]] : vector<4xi32>
    %result = ori %lhs, %rhs : tile<4xi32>
    return
  }

  // --- remf ---
  // CHECK-LABEL: gpu.func @test_remf
  entry @test_remf() {
    // CHECK: %[[REMF_LHS:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %lhs = constant <f32: [5.0, 7.0, 3.0, 9.0]> : tile<4xf32>
    // CHECK: %[[REMF_RHS:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %rhs = constant <f32: [2.0, 3.0, 2.0, 4.0]> : tile<4xf32>
    // CHECK: %[[REMF_R:.*]] = arith.remf %[[REMF_LHS]], %[[REMF_RHS]] : vector<4xf32>
    %result = remf %lhs, %rhs : tile<4xf32>
    return
  }

  // --- addi ---
  // CHECK-LABEL: gpu.func @test_addi
  entry @test_addi() {
    // CHECK: %[[ADDI_LHS:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
    %lhs = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
    // CHECK: %[[ADDI_RHS:.*]] = arith.constant dense<[4, 5, 6, 7]> : vector<4xi32>
    %rhs = constant <i32: [4, 5, 6, 7]> : tile<4xi32>
    // CHECK: %[[ADDI_R:.*]] = arith.addi %[[ADDI_LHS]], %[[ADDI_RHS]] : vector<4xi32>
    %result = addi %lhs, %rhs : tile<4xi32>
    return
  }

  // --- subi ---
  // CHECK-LABEL: gpu.func @test_subi
  entry @test_subi() {
    // CHECK: %[[SUBI_LHS:.*]] = arith.constant dense<[4, 5, 6, 7]> : vector<4xi32>
    %lhs = constant <i32: [4, 5, 6, 7]> : tile<4xi32>
    // CHECK: %[[SUBI_RHS:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
    %rhs = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
    // CHECK: %[[SUBI_R:.*]] = arith.subi %[[SUBI_LHS]], %[[SUBI_RHS]] : vector<4xi32>
    %result = subi %lhs, %rhs : tile<4xi32>
    return
  }

  // --- shli ---
  // CHECK-LABEL: gpu.func @test_shli
  entry @test_shli() {
    // CHECK: %[[SHLI_LHS:.*]] = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
    %lhs = constant <i32: [1, 2, 3, 4]> : tile<4xi32>
    // CHECK: %[[SHLI_RHS:.*]] = arith.constant dense<1> : vector<4xi32>
    %rhs = constant <i32: 1> : tile<4xi32>
    // CHECK: %[[SHLI_R:.*]] = arith.shli %[[SHLI_LHS]], %[[SHLI_RHS]] : vector<4xi32>
    %result = shli %lhs, %rhs : tile<4xi32>
    return
  }

  // --- divi ---
  // CHECK-LABEL: gpu.func @test_divi
  entry @test_divi() {
    // CHECK: %[[DIVI_LHS:.*]] = arith.constant dense<[8, 9, 10, 11]> : vector<4xi32>
    %lhs = constant <i32: [8, 9, 10, 11]> : tile<4xi32>
    // CHECK: %[[DIVI_RHS:.*]] = arith.constant dense<[2, 3, 4, 5]> : vector<4xi32>
    %rhs = constant <i32: [2, 3, 4, 5]> : tile<4xi32>
    // CHECK: %[[DIVI_S:.*]] = arith.divsi %[[DIVI_LHS]], %[[DIVI_RHS]] : vector<4xi32>
    %s = divi %lhs, %rhs signed : tile<4xi32>
    // CHECK: %[[DIVI_U:.*]] = arith.divui %[[DIVI_LHS]], %[[DIVI_RHS]] : vector<4xi32>
    %u = divi %lhs, %rhs unsigned : tile<4xi32>
    return
  }

  // --- remi ---
  // CHECK-LABEL: gpu.func @test_remi
  entry @test_remi() {
    // CHECK: %[[REMI_LHS:.*]] = arith.constant dense<[7, 8, 9, 10]> : vector<4xi32>
    %lhs = constant <i32: [7, 8, 9, 10]> : tile<4xi32>
    // CHECK: %[[REMI_RHS:.*]] = arith.constant dense<[3, 3, 4, 4]> : vector<4xi32>
    %rhs = constant <i32: [3, 3, 4, 4]> : tile<4xi32>
    // CHECK: %[[REMI_S:.*]] = arith.remsi %[[REMI_LHS]], %[[REMI_RHS]] : vector<4xi32>
    %s = remi %lhs, %rhs signed : tile<4xi32>
    // CHECK: %[[REMI_U:.*]] = arith.remui %[[REMI_LHS]], %[[REMI_RHS]] : vector<4xi32>
    %u = remi %lhs, %rhs unsigned : tile<4xi32>
    return
  }

  // --- shri ---
  // CHECK-LABEL: gpu.func @test_shri
  entry @test_shri() {
    // CHECK: %[[SHRI_LHS:.*]] = arith.constant dense<[8, 16, 32, 64]> : vector<4xi32>
    %lhs = constant <i32: [8, 16, 32, 64]> : tile<4xi32>
    // CHECK: %[[SHRI_RHS:.*]] = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
    %rhs = constant <i32: [1, 2, 3, 4]> : tile<4xi32>
    // CHECK: %[[SHRI_S:.*]] = arith.shrsi %[[SHRI_LHS]], %[[SHRI_RHS]] : vector<4xi32>
    %s = shri %lhs, %rhs signed : tile<4xi32>
    // CHECK: %[[SHRI_U:.*]] = arith.shrui %[[SHRI_LHS]], %[[SHRI_RHS]] : vector<4xi32>
    %u = shri %lhs, %rhs unsigned : tile<4xi32>
    return
  }

  // --- addf (unspecified rounding -> default nearest_even; explicit zero) ---
  // CHECK-LABEL: gpu.func @test_addf
  entry @test_addf() {
    // CHECK: %[[ADDF_LHS:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %lhs = constant <f32: [1.0, 2.0, 3.0, 4.0]> : tile<4xf32>
    // CHECK: %[[ADDF_RHS:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %rhs = constant <f32: [5.0, 6.0, 7.0, 8.0]> : tile<4xf32>
    // Unspecified rounding defaults to nearest_even.
    // CHECK: %[[ADDF_R:.*]] = arith.addf %[[ADDF_LHS]], %[[ADDF_RHS]] : vector<4xf32>
    %result = addf %lhs, %rhs : tile<4xf32>
    // Explicit rounding<zero> -> toward_zero.
    // CHECK: %[[ADDF_RZ:.*]] = arith.addf %[[ADDF_LHS]], %[[ADDF_RHS]] : vector<4xf32>
    %result_z = addf %lhs, %rhs rounding<zero> : tile<4xf32>
    return
  }

  // --- subf (unspecified rounding -> default nearest_even; explicit negative_inf) ---
  // CHECK-LABEL: gpu.func @test_subf
  entry @test_subf() {
    // CHECK: %[[SUBF_LHS:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %lhs = constant <f32: [5.0, 6.0, 7.0, 8.0]> : tile<4xf32>
    // CHECK: %[[SUBF_RHS:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %rhs = constant <f32: [1.0, 2.0, 3.0, 4.0]> : tile<4xf32>
    // CHECK: %[[SUBF_R:.*]] = arith.subf %[[SUBF_LHS]], %[[SUBF_RHS]] : vector<4xf32>
    %result = subf %lhs, %rhs : tile<4xf32>
    // Explicit rounding<negative_inf> -> downward.
    // CHECK: %[[SUBF_RN:.*]] = arith.subf %[[SUBF_LHS]], %[[SUBF_RHS]] : vector<4xf32>
    %result_n = subf %lhs, %rhs rounding<negative_inf> : tile<4xf32>
    return
  }

  // --- mulf (unspecified rounding -> default nearest_even; explicit positive_inf) ---
  // CHECK-LABEL: gpu.func @test_mulf
  entry @test_mulf() {
    // CHECK: %[[MULF_LHS:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %lhs = constant <f32: [1.0, 2.0, 3.0, 4.0]> : tile<4xf32>
    // CHECK: %[[MULF_RHS:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %rhs = constant <f32: [2.0, 3.0, 4.0, 5.0]> : tile<4xf32>
    // CHECK: %[[MULF_R:.*]] = arith.mulf %[[MULF_LHS]], %[[MULF_RHS]] : vector<4xf32>
    %result = mulf %lhs, %rhs : tile<4xf32>
    // Explicit rounding<positive_inf> -> upward.
    // CHECK: %[[MULF_RP:.*]] = arith.mulf %[[MULF_LHS]], %[[MULF_RHS]] : vector<4xf32>
    %result_p = mulf %lhs, %rhs rounding<positive_inf> : tile<4xf32>
    return
  }

  // --- divf (unspecified rounding -> default nearest_even; explicit nearest_even) ---
  // CHECK-LABEL: gpu.func @test_divf
  entry @test_divf() {
    // CHECK: %[[DIVF_LHS:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %lhs = constant <f32: [4.0, 9.0, 16.0, 25.0]> : tile<4xf32>
    // CHECK: %[[DIVF_RHS:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %rhs = constant <f32: [2.0, 3.0, 4.0, 5.0]> : tile<4xf32>
    // CHECK: %[[DIVF_R:.*]] = arith.divf %[[DIVF_LHS]], %[[DIVF_RHS]] : vector<4xf32>
    %result = divf %lhs, %rhs : tile<4xf32>
    // Explicit rounding<nearest_even> -> to_nearest_even.
    // CHECK: %[[DIVF_RE:.*]] = arith.divf %[[DIVF_LHS]], %[[DIVF_RHS]] : vector<4xf32>
    %result_e = divf %lhs, %rhs rounding<nearest_even> : tile<4xf32>
    return
  }

  // --- fma ---
  // CHECK-LABEL: gpu.func @test_fma
  entry @test_fma() {
    // CHECK: %[[FMA_LHS:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %lhs = constant <f32: [1.0, 2.0, 3.0, 4.0]> : tile<4xf32>
    // CHECK: %[[FMA_RHS:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %rhs = constant <f32: [2.0, 3.0, 4.0, 5.0]> : tile<4xf32>
    // CHECK: %[[FMA_ACC:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
    %acc = constant <f32: [0.5, 0.5, 0.5, 0.5]> : tile<4xf32>
    // CHECK: %[[FMA_R:.*]] = math.fma %[[FMA_LHS]], %[[FMA_RHS]], %[[FMA_ACC]] {"tir-dropped-rounding" = "nearest_even"} : vector<4xf32>
    %result = fma %lhs, %rhs, %acc : tile<4xf32>
    return
  }

  // --- addi with overflow flags ---
  // CHECK-LABEL: gpu.func @test_addi_overflow
  entry @test_addi_overflow() {
    // CHECK: %[[AOV_LHS:.*]] = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
    %lhs = constant <i32: [1, 2, 3, 4]> : tile<4xi32>
    // CHECK: %[[AOV_RHS:.*]] = arith.constant dense<[5, 6, 7, 8]> : vector<4xi32>
    %rhs = constant <i32: [5, 6, 7, 8]> : tile<4xi32>
    // CHECK: %[[AOV_NSW:.*]] = arith.addi %[[AOV_LHS]], %[[AOV_RHS]] overflow<nsw> : vector<4xi32>
    %nsw = addi %lhs, %rhs overflow<no_signed_wrap> : tile<4xi32>
    // CHECK: %[[AOV_NUW:.*]] = arith.addi %[[AOV_LHS]], %[[AOV_RHS]] overflow<nuw> : vector<4xi32>
    %nuw = addi %lhs, %rhs overflow<no_unsigned_wrap> : tile<4xi32>
    // CHECK: %[[AOV_NW:.*]] = arith.addi %[[AOV_LHS]], %[[AOV_RHS]] overflow<nsw, nuw> : vector<4xi32>
    %nw = addi %lhs, %rhs overflow<no_wrap> : tile<4xi32>
    return
  }

  // --- subi with overflow flags ---
  // CHECK-LABEL: gpu.func @test_subi_overflow
  entry @test_subi_overflow() {
    // CHECK: %[[SOV_LHS:.*]] = arith.constant dense<[5, 6, 7, 8]> : vector<4xi32>
    %lhs = constant <i32: [5, 6, 7, 8]> : tile<4xi32>
    // CHECK: %[[SOV_RHS:.*]] = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
    %rhs = constant <i32: [1, 2, 3, 4]> : tile<4xi32>
    // CHECK: %[[SOV_NSW:.*]] = arith.subi %[[SOV_LHS]], %[[SOV_RHS]] overflow<nsw> : vector<4xi32>
    %nsw = subi %lhs, %rhs overflow<no_signed_wrap> : tile<4xi32>
    return
  }

  // --- shli with overflow flags ---
  // CHECK-LABEL: gpu.func @test_shli_overflow
  entry @test_shli_overflow() {
    // CHECK: %[[SHLOV_LHS:.*]] = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
    %lhs = constant <i32: [1, 2, 3, 4]> : tile<4xi32>
    // CHECK: %[[SHLOV_RHS:.*]] = arith.constant dense<1> : vector<4xi32>
    %rhs = constant <i32: 1> : tile<4xi32>
    // CHECK: %[[SHLOV_NUW:.*]] = arith.shli %[[SHLOV_LHS]], %[[SHLOV_RHS]] overflow<nuw> : vector<4xi32>
    %nuw = shli %lhs, %rhs overflow<no_unsigned_wrap> : tile<4xi32>
    return
  }

  // --- scalar math ops ---
  // CHECK-LABEL: gpu.func @test_scalar_math_ops
  entry @test_scalar_math_ops() {
    // CHECK: %[[SM_F:.*]] = arith.constant 4.000000e+00 : f32
    %f = constant <f32: 4.0> : tile<f32>
    // CHECK: %[[SM_I:.*]] = arith.constant -3 : i32
    %i = constant <i32: -3> : tile<i32>
    // CHECK: %[[SM_SQRT:.*]] = math.sqrt %[[SM_F]] {"tir-dropped-rounding" = "nearest_even"} : f32
    %sq = sqrt %f : tile<f32>
    // CHECK: %[[SM_ABSF:.*]] = math.absf %[[SM_F]] : f32
    %af = absf %f : tile<f32>
    // CHECK: %[[SM_ABSI:.*]] = math.absi %[[SM_I]] : i32
    %ai = absi %i : tile<i32>
    // CHECK: %[[SM_LOG:.*]] = math.log %[[SM_F]] : f32
    %lg = log %f : tile<f32>
    return
  }

  // --- scalar binary ops ---
  // CHECK-LABEL: gpu.func @test_scalar_binary_ops
  entry @test_scalar_binary_ops() {
    // CHECK: %[[SB_A:.*]] = arith.constant 10 : i32
    %a = constant <i32: 10> : tile<i32>
    // CHECK: %[[SB_B:.*]] = arith.constant 3 : i32
    %b = constant <i32: 3> : tile<i32>
    // CHECK: %[[SB_ADDI:.*]] = arith.addi %[[SB_A]], %[[SB_B]] : i32
    %add = addi %a, %b : tile<i32>
    // CHECK: %[[SB_SUBI:.*]] = arith.subi %[[SB_A]], %[[SB_B]] : i32
    %sub = subi %a, %b : tile<i32>
    // CHECK: %[[SB_DIVS:.*]] = arith.divsi %[[SB_A]], %[[SB_B]] : i32
    %ds = divi %a, %b signed : tile<i32>
    // CHECK: %[[SB_DIVU:.*]] = arith.divui %[[SB_A]], %[[SB_B]] : i32
    %du = divi %a, %b unsigned : tile<i32>
    // CHECK: %[[SB_REMS:.*]] = arith.remsi %[[SB_A]], %[[SB_B]] : i32
    %rs = remi %a, %b signed : tile<i32>
    // CHECK: %[[SB_REMU:.*]] = arith.remui %[[SB_A]], %[[SB_B]] : i32
    %ru = remi %a, %b unsigned : tile<i32>
    return
  }

  // --- scalar float ops ---
  // CHECK-LABEL: gpu.func @test_scalar_float_ops
  entry @test_scalar_float_ops() {
    // CHECK: %[[SF_A:.*]] = arith.constant 6.000000e+00 : f32
    %a = constant <f32: 6.0> : tile<f32>
    // CHECK: %[[SF_B:.*]] = arith.constant 2.000000e+00 : f32
    %b = constant <f32: 2.0> : tile<f32>
    // CHECK: %[[SF_C:.*]] = arith.constant 1.000000e+00 : f32
    %c = constant <f32: 1.0> : tile<f32>
    // CHECK: %[[SF_ADDF:.*]] = arith.addf %[[SF_A]], %[[SF_B]] : f32
    %add = addf %a, %b : tile<f32>
    // CHECK: %[[SF_SUBF:.*]] = arith.subf %[[SF_A]], %[[SF_B]] : f32
    %sub = subf %a, %b : tile<f32>
    // CHECK: %[[SF_MULF:.*]] = arith.mulf %[[SF_A]], %[[SF_B]] : f32
    %mul = mulf %a, %b : tile<f32>
    // CHECK: %[[SF_DIVF:.*]] = arith.divf %[[SF_A]], %[[SF_B]] : f32
    %div = divf %a, %b : tile<f32>
    // CHECK: %[[SF_FMA:.*]] = math.fma %[[SF_A]], %[[SF_B]], %[[SF_C]] {"tir-dropped-rounding" = "nearest_even"} : f32
    %fm = fma %a, %b, %c : tile<f32>
    return
  }

  // --- muli with overflow flags ---
  // CHECK-LABEL: gpu.func @test_muli_overflow
  entry @test_muli_overflow() {
    // CHECK: %[[MOV_LHS:.*]] = arith.constant dense<[2, 3, 4, 5]> : vector<4xi32>
    %lhs = constant <i32: [2, 3, 4, 5]> : tile<4xi32>
    // CHECK: %[[MOV_RHS:.*]] = arith.constant dense<[6, 7, 8, 9]> : vector<4xi32>
    %rhs = constant <i32: [6, 7, 8, 9]> : tile<4xi32>
    // CHECK: %[[MOV_NSW:.*]] = arith.muli %[[MOV_LHS]], %[[MOV_RHS]] overflow<nsw> : vector<4xi32>
    %nsw = muli %lhs, %rhs overflow<no_signed_wrap> : tile<4xi32>
    // CHECK: %[[MOV_NUW:.*]] = arith.muli %[[MOV_LHS]], %[[MOV_RHS]] overflow<nuw> : vector<4xi32>
    %nuw = muli %lhs, %rhs overflow<no_unsigned_wrap> : tile<4xi32>
    // CHECK: %[[MOV_NW:.*]] = arith.muli %[[MOV_LHS]], %[[MOV_RHS]] overflow<nsw, nuw> : vector<4xi32>
    %nw = muli %lhs, %rhs overflow<no_wrap> : tile<4xi32>
    return
  }

  // --- permute: 2D transpose ---
  // CHECK-LABEL: gpu.func @test_permute_2d
  entry @test_permute_2d() {
    // CHECK: %[[P2D_IN:.*]] = arith.constant dense<1> : vector<2x4xi32>
    %a = constant <i32: 1> : tile<2x4xi32>
    // CHECK: %[[P2D_R:.*]] = vector.transpose %[[P2D_IN]], [1, 0] : vector<2x4xi32> to vector<4x2xi32>
    %t = permute %a [1, 0] : tile<2x4xi32> -> tile<4x2xi32>
    return
  }

  // --- permute: identity permutation (no-op reordering) ---
  // CHECK-LABEL: gpu.func @test_permute_identity
  entry @test_permute_identity() {
    // CHECK: %[[PID_IN:.*]] = arith.constant dense<0.000000e+00> : vector<2x4x8xf32>
    %a = constant <f32: 0.0> : tile<2x4x8xf32>
    // CHECK: %[[PID_R:.*]] = vector.transpose %[[PID_IN]], [0, 1, 2] : vector<2x4x8xf32> to vector<2x4x8xf32>
    %t = permute %a [0, 1, 2] : tile<2x4x8xf32> -> tile<2x4x8xf32>
    return
  }

  // --- permute: 4D permutation ---
  // CHECK-LABEL: gpu.func @test_permute_4d
  entry @test_permute_4d() {
    // CHECK: %[[P4D_IN:.*]] = arith.constant dense<1.000000e+00> : vector<2x4x8x16xf16>
    %a = constant <f16: 1.0> : tile<2x4x8x16xf16>
    // CHECK: %[[P4D_R:.*]] = vector.transpose %[[P4D_IN]], [3, 2, 1, 0] : vector<2x4x8x16xf16> to vector<16x8x4x2xf16>
    %t = permute %a [3, 2, 1, 0] : tile<2x4x8x16xf16> -> tile<16x8x4x2xf16>
    return
  }

  // --- broadcast: 1-D expand from size 1 ---
  // CHECK-LABEL: gpu.func @test_broadcast_1d
  entry @test_broadcast_1d() {
    // CHECK: %[[B1D_IN:.*]] = arith.constant dense<3.000000e+00> : vector<1xf32>
    %a = constant <f32: 3.0> : tile<1xf32>
    // CHECK: %[[B1D_R:.*]] = vector.broadcast %[[B1D_IN]] : vector<1xf32> to vector<8xf32>
    %r = broadcast %a : tile<1xf32> -> tile<8xf32>
    return
  }

  // --- broadcast: 2-D expand both dimensions ---
  // CHECK-LABEL: gpu.func @test_broadcast_2d
  entry @test_broadcast_2d() {
    // CHECK: %[[B2D_IN:.*]] = arith.constant dense<2.000000e+00> : vector<1x1xf16>
    %a = constant <f16: 2.0> : tile<1x1xf16>
    // CHECK: %[[B2D_R:.*]] = vector.broadcast %[[B2D_IN]] : vector<1x1xf16> to vector<4x8xf16>
    %r = broadcast %a : tile<1x1xf16> -> tile<4x8xf16>
    return
  }

  // --- broadcast: 2-D expand only second dimension ---
  // CHECK-LABEL: gpu.func @test_broadcast_partial
  entry @test_broadcast_partial() {
    // CHECK: %[[BP_IN:.*]] = arith.constant dense<1.000000e+00> : vector<4x1xf32>
    %a = constant <f32: 1.0> : tile<4x1xf32>
    // CHECK: %[[BP_R:.*]] = vector.broadcast %[[BP_IN]] : vector<4x1xf32> to vector<4x8xf32>
    %r = broadcast %a : tile<4x1xf32> -> tile<4x8xf32>
    return
  }

  // --- broadcast: 3-D expand selected dimensions ---
  // CHECK-LABEL: gpu.func @test_broadcast_3d
  entry @test_broadcast_3d() {
    // CHECK: %[[B3D_IN:.*]] = arith.constant dense<5.000000e-01> : vector<1x4x1xf32>
    %a = constant <f32: 0.5> : tile<1x4x1xf32>
    // CHECK: %[[B3D_R:.*]] = vector.broadcast %[[B3D_IN]] : vector<1x4x1xf32> to vector<8x4x16xf32>
    %r = broadcast %a : tile<1x4x1xf32> -> tile<8x4x16xf32>
    return
  }

  // --- extract: 1-D extraction (no transpose needed) ---
  // CHECK-LABEL: gpu.func @test_extract_1d
  entry @test_extract_1d() {
    // CHECK-DAG: %[[E1D_IDX:.*]] = arith.constant 2 : i32
    %idx = constant <i32: 2> : tile<i32>
    // CHECK-DAG: %[[E1D_SRC:.*]] = arith.constant dense<1.000000e+00> : vector<16xf32>
    %src = constant <f32: 1.0> : tile<16xf32>
    // CHECK: %[[E1D_RESHAPE:.*]] = vector.shape_cast %[[E1D_SRC]] : vector<16xf32> to vector<4x4xf32>
    // CHECK: %[[E1D_CAST:.*]] = arith.index_castui %[[E1D_IDX]] : i32 to index
    // CHECK: %[[E1D_R:.*]] = vector.extract %[[E1D_RESHAPE]][%[[E1D_CAST]]] : vector<4xf32> from vector<4x4xf32>
    %r = extract %src[%idx] : tile<16xf32> -> tile<4xf32>
    return
  }

  // --- extract: trivial identity (source == result shape) ---
  // CHECK-LABEL: gpu.func @test_extract_identity
  entry @test_extract_identity() {
    // CHECK-DAG: %[[EID_IDX:.*]] = arith.constant 0 : i32
    %idx0 = constant <i32: 0> : tile<i32>
    // CHECK-DAG: %[[EID_SRC:.*]] = arith.constant dense<2.000000e+00> : vector<4x8xf32>
    %src = constant <f32: 2.0> : tile<4x8xf32>
    // Source and result shapes match → identity (no reshape/transpose/extract).
    // CHECK-NOT: vector.shape_cast
    // CHECK-NOT: vector.transpose
    // CHECK-NOT: vector.extract
    %r = extract %src[%idx0, %idx0] : tile<4x8xf32> -> tile<4x8xf32>
    return
  }

  // --- extract: 2-D extraction with integer type ---
  // CHECK-LABEL: gpu.func @test_extract_2d_int
  entry @test_extract_2d_int() {
    // CHECK-DAG: %[[E2I_I:.*]] = arith.constant 1 : i32
    %i = constant <i32: 1> : tile<i32>
    // CHECK-DAG: %[[E2I_J:.*]] = arith.constant 0 : i32
    %j = constant <i32: 0> : tile<i32>
    // CHECK-DAG: %[[E2I_SRC:.*]] = arith.constant dense<42> : vector<8x4xi16>
    %src = constant <i16: 42> : tile<8x4xi16>
    // 8/2=4 slices in dim0, 4/2=2 slices in dim1.
    // CHECK: %[[E2I_RESHAPE:.*]] = vector.shape_cast %[[E2I_SRC]] : vector<8x4xi16> to vector<4x2x2x2xi16>
    // CHECK: %[[E2I_TRANS:.*]] = vector.transpose %[[E2I_RESHAPE]], [0, 2, 1, 3] : vector<4x2x2x2xi16> to vector<4x2x2x2xi16>
    // CHECK: %[[E2I_IDX0:.*]] = arith.index_castui %[[E2I_I]] : i32 to index
    // CHECK: %[[E2I_IDX1:.*]] = arith.index_castui %[[E2I_J]] : i32 to index
    // CHECK: %[[E2I_R:.*]] = vector.extract %[[E2I_TRANS]][%[[E2I_IDX0]], %[[E2I_IDX1]]] : vector<2x2xi16> from vector<4x2x2x2xi16>
    %r = extract %src[%i, %j] : tile<8x4xi16> -> tile<2x2xi16>
    return
  }

  // --- extract: 1-D with halving (extract half) ---
  // CHECK-LABEL: gpu.func @test_extract_half
  entry @test_extract_half() {
    // CHECK-DAG: %[[EH_IDX:.*]] = arith.constant 1 : i32
    %idx = constant <i32: 1> : tile<i32>
    // CHECK-DAG: %[[EH_SRC:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
    %src = constant <f32: 0.0> : tile<8xf32>
    // 8/4=2 slices.
    // CHECK: %[[EH_RESHAPE:.*]] = vector.shape_cast %[[EH_SRC]] : vector<8xf32> to vector<2x4xf32>
    // CHECK: %[[EH_CAST:.*]] = arith.index_castui %[[EH_IDX]] : i32 to index
    // CHECK: %[[EH_R:.*]] = vector.extract %[[EH_RESHAPE]][%[[EH_CAST]]] : vector<4xf32> from vector<2x4xf32>
    %r = extract %src[%idx] : tile<8xf32> -> tile<4xf32>
    return
  }

  // --- cat: 1D concatenation ---
  // CHECK-LABEL: gpu.func @test_cat_1d
  entry @test_cat_1d() {
    // CHECK-DAG: %[[C1D_L:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    %l = constant <f32: 1.0> : tile<4xf32>
    // CHECK-DAG: %[[C1D_R:.*]] = arith.constant dense<2.000000e+00> : vector<4xf32>
    %r = constant <f32: 2.0> : tile<4xf32>
    // dim=0, lhs.shape[0]=4, result = <8xf32>
    // CHECK: %[[C1D_P:.*]] = ub.poison : vector<8xf32>
    // CHECK: %[[C1D_INS0:.*]] = vector.insert_strided_slice %[[C1D_L]], %[[C1D_P]] {offsets = [0], strides = [1]} : vector<4xf32> into vector<8xf32>
    // CHECK: %[[C1D_INS1:.*]] = vector.insert_strided_slice %[[C1D_R]], %[[C1D_INS0]] {offsets = [4], strides = [1]} : vector<4xf32> into vector<8xf32>
    %0 = cat %l, %r dim = 0 : tile<4xf32>, tile<4xf32> -> tile<8xf32>
    return
  }

  // --- cat: asymmetric 2D along dim 0 ---
  // CHECK-LABEL: gpu.func @test_cat_2d_dim0
  entry @test_cat_2d_dim0() {
    // CHECK-DAG: %[[C2D0_L:.*]] = arith.constant dense<0> : vector<2x8xi16>
    %l = constant <i16: 0> : tile<2x8xi16>
    // CHECK-DAG: %[[C2D0_R:.*]] = arith.constant dense<1> : vector<2x8xi16>
    %r = constant <i16: 1> : tile<2x8xi16>
    // result = <4x8xi16>
    // CHECK: %[[C2D0_P:.*]] = ub.poison : vector<4x8xi16>
    // CHECK: %[[C2D0_INS0:.*]] = vector.insert_strided_slice %[[C2D0_L]], %[[C2D0_P]] {offsets = [0, 0], strides = [1, 1]} : vector<2x8xi16> into vector<4x8xi16>
    // CHECK: %[[C2D0_INS1:.*]] = vector.insert_strided_slice %[[C2D0_R]], %[[C2D0_INS0]] {offsets = [2, 0], strides = [1, 1]} : vector<2x8xi16> into vector<4x8xi16>
    %0 = cat %l, %r dim = 0 : tile<2x8xi16>, tile<2x8xi16> -> tile<4x8xi16>
    return
  }

  // --- cat: 3D concatenation along middle dim ---
  // CHECK-LABEL: gpu.func @test_cat_3d_dim1
  entry @test_cat_3d_dim1() {
    // CHECK-DAG: %[[C3D_L:.*]] = arith.constant dense<0.000000e+00> : vector<2x4x8xf16>
    %l = constant <f16: 0.0> : tile<2x4x8xf16>
    // CHECK-DAG: %[[C3D_R:.*]] = arith.constant dense<0.000000e+00> : vector<2x4x8xf16>
    %r = constant <f16: 0.0> : tile<2x4x8xf16>
    // result = <2x8x8xf16>
    // CHECK: %[[C3D_P:.*]] = ub.poison : vector<2x8x8xf16>
    // CHECK: %[[C3D_INS0:.*]] = vector.insert_strided_slice %[[C3D_L]], %[[C3D_P]] {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<2x4x8xf16> into vector<2x8x8xf16>
    // CHECK: %[[C3D_INS1:.*]] = vector.insert_strided_slice %[[C3D_R]], %[[C3D_INS0]] {offsets = [0, 4, 0], strides = [1, 1, 1]} : vector<2x4x8xf16> into vector<2x8x8xf16>
    %0 = cat %l, %r dim = 1 : tile<2x4x8xf16>, tile<2x4x8xf16> -> tile<2x8x8xf16>
    return
  }

  // --- cat: integer type, last dim ---
  // CHECK-LABEL: gpu.func @test_cat_i32_last_dim
  entry @test_cat_i32_last_dim() {
    // CHECK-DAG: %[[CI_L:.*]] = arith.constant dense<0> : vector<4x2xi32>
    %l = constant <i32: 0> : tile<4x2xi32>
    // CHECK-DAG: %[[CI_R:.*]] = arith.constant dense<0> : vector<4x2xi32>
    %r = constant <i32: 0> : tile<4x2xi32>
    // result = <4x4xi32>
    // CHECK: %[[CI_P:.*]] = ub.poison : vector<4x4xi32>
    // CHECK: %[[CI_INS0:.*]] = vector.insert_strided_slice %[[CI_L]], %[[CI_P]] {offsets = [0, 0], strides = [1, 1]} : vector<4x2xi32> into vector<4x4xi32>
    // CHECK: %[[CI_INS1:.*]] = vector.insert_strided_slice %[[CI_R]], %[[CI_INS0]] {offsets = [0, 2], strides = [1, 1]} : vector<4x2xi32> into vector<4x4xi32>
    %0 = cat %l, %r dim = 1 : tile<4x2xi32>, tile<4x2xi32> -> tile<4x4xi32>
    return
  }

  // --- addf: rounding<zero> propagated, flush_to_zero dropped ---
  // CHECK-LABEL: gpu.func @test_addf_dropped_flags
  entry @test_addf_dropped_flags() {
    // CHECK-DAG: %[[DF_LHS:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    %lhs = constant <f32: 1.0> : tile<4xf32>
    // CHECK-DAG: %[[DF_RHS:.*]] = arith.constant dense<2.000000e+00> : vector<4xf32>
    %rhs = constant <f32: 2.0> : tile<4xf32>
    // CHECK: arith.addf %[[DF_LHS]], %[[DF_RHS]] {"tir-dropped-flush-to-zero"} : vector<4xf32>
    %r = addf %lhs, %rhs rounding<zero> flush_to_zero : tile<4xf32>
    return
  }

  // --- divf rounding<approx> + flush_to_zero: rounding captured as arcp, ftz dropped ---
  // CHECK-LABEL: gpu.func @test_divf_approx_ftz
  entry @test_divf_approx_ftz() {
    // CHECK-DAG: %[[DA_LHS:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    %lhs = constant <f32: 1.0> : tile<4xf32>
    // CHECK-DAG: %[[DA_RHS:.*]] = arith.constant dense<2.000000e+00> : vector<4xf32>
    %rhs = constant <f32: 2.0> : tile<4xf32>
    // CHECK: arith.divf %[[DA_LHS]], %[[DA_RHS]] fastmath<arcp> {"tir-dropped-flush-to-zero"} : vector<4xf32>
    %r = divf %lhs, %rhs rounding<approx> flush_to_zero : tile<4xf32>
    return
  }

  // --- divf rounding<zero>: rounding propagated, no arcp in output ---
  // CHECK-LABEL: gpu.func @test_divf_rounding_dropped
  entry @test_divf_rounding_dropped() {
    // CHECK-DAG: %[[DR_LHS:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    %lhs = constant <f32: 1.0> : tile<4xf32>
    // CHECK-DAG: %[[DR_RHS:.*]] = arith.constant dense<2.000000e+00> : vector<4xf32>
    %rhs = constant <f32: 2.0> : tile<4xf32>
    // CHECK: arith.divf %[[DR_LHS]], %[[DR_RHS]] : vector<4xf32>
    %r = divf %lhs, %rhs rounding<zero> : tile<4xf32>
    return
  }

  // --- fma: rounding<zero> + flush_to_zero both dropped ---
  // CHECK-LABEL: gpu.func @test_fma_dropped_flags
  entry @test_fma_dropped_flags() {
    // CHECK-DAG: %[[FF_A:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    %a = constant <f32: 1.0> : tile<4xf32>
    // CHECK-DAG: %[[FF_B:.*]] = arith.constant dense<2.000000e+00> : vector<4xf32>
    %b = constant <f32: 2.0> : tile<4xf32>
    // CHECK-DAG: %[[FF_C:.*]] = arith.constant dense<3.000000e+00> : vector<4xf32>
    %c = constant <f32: 3.0> : tile<4xf32>
    // CHECK: math.fma %[[FF_A]], %[[FF_B]], %[[FF_C]] {"tir-dropped-flush-to-zero", "tir-dropped-rounding" = "zero"} : vector<4xf32>
    %r = fma %a, %b, %c rounding<zero> flush_to_zero : tile<4xf32>
    return
  }

  // --- sqrt rounding<zero>: rounding dropped, no afn ---
  // CHECK-LABEL: gpu.func @test_sqrt_dropped_rounding
  entry @test_sqrt_dropped_rounding() {
    // CHECK: %[[SD_IN:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    %in = constant <f32: 1.0> : tile<4xf32>
    // CHECK: math.sqrt %[[SD_IN]] {"tir-dropped-rounding" = "zero"} : vector<4xf32>
    %r = sqrt %in rounding<zero> : tile<4xf32>
    return
  }

  // --- tanh rounding<approx>: captured as fastmath<afn>, no tir-dropped attr ---
  // CHECK-LABEL: gpu.func @test_tanh_approx
  entry @test_tanh_approx() {
    // CHECK: %[[TA_IN:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    %in = constant <f32: 1.0> : tile<4xf32>
    // CHECK: math.tanh %[[TA_IN]] fastmath<afn> {"tir-dropped-rounding" = "approx"} : vector<4xf32>
    %r = tanh %in rounding<approx> : tile<4xf32>
    return
  }

  // --- negi overflow<no_signed_wrap>: overflow dropped ---
  // CHECK-LABEL: gpu.func @test_negi_dropped_overflow
  entry @test_negi_dropped_overflow() {
    // CHECK: %[[NO_IN:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
    %source = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
    // CHECK: %[[NO_ZERO:.*]] = arith.constant dense<0> : vector<4xi32>
    // CHECK: arith.subi %[[NO_ZERO]], %[[NO_IN]] {"tir-dropped-overflow" = "no_signed_wrap"} : vector<4xi32>
    %r = negi %source overflow<no_signed_wrap> : tile<4xi32>
    return
  }

  // --- maxf flush_to_zero: ftz dropped ---
  // CHECK-LABEL: gpu.func @test_maxf_ftz
  entry @test_maxf_ftz() {
    // CHECK-DAG: %[[MF_LHS:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    %lhs = constant <f32: 1.0> : tile<4xf32>
    // CHECK-DAG: %[[MF_RHS:.*]] = arith.constant dense<2.000000e+00> : vector<4xf32>
    %rhs = constant <f32: 2.0> : tile<4xf32>
    // CHECK: arith.maxnumf %[[MF_LHS]], %[[MF_RHS]] {"tir-dropped-flush-to-zero"} : vector<4xf32>
    %r = maxf %lhs, %rhs flush_to_zero : tile<4xf32>
    return
  }

  // --- scalar atomic_rmw_tko mode xchg: maps to memref.atomic_rmw assign ---
  // CHECK-LABEL: gpu.func @test_atomic_rmw_scalar_xchg
  // CHECK-SAME: %[[AX_UPTR:[a-zA-Z0-9_]+]]: memref<*xi32>
  entry @test_atomic_rmw_scalar_xchg(%p: !cuda_tile.tile<!cuda_tile.ptr<i32>>) {
    %v = constant <i32: 7> : tile<i32>
    // CHECK: %[[AX_V:.*]] = arith.constant 7 : i32
    // CHECK: %[[AX_R0:.*]] = memref.reinterpret_cast %[[AX_UPTR]] to offset: [0], sizes: [], strides: [] : memref<*xi32> to memref<i32, strided<[], offset: ?>>
    // CHECK: %[[AX_OLD:.*]] = memref.atomic_rmw assign %[[AX_V]], %[[AX_R0]][] {{.*dropped-memory-ordering.*acq_rel.*dropped-memory-scope.*sys.*}} : (i32, memref<i32, strided<[], offset: ?>>) -> i32
    %old, %tok = atomic_rmw_tko acq_rel sys %p, xchg, %v : tile<ptr<i32>>, tile<i32> -> tile<i32>, !cuda_tile.token
    return
  }

  // --- pack/unpack edge cases ---
  // CHECK-LABEL: gpu.func @test_pack_unpack_edges
  entry @test_pack_unpack_edges() {
    // Sub-byte source: 8 x i1 (8 bits) packs into a single byte.
    // CHECK: %[[PK_I1:.*]] = arith.constant dense<true> : vector<8xi1>
    %i1s = constant <i1: 1> : tile<8xi1>
    // CHECK: vector.bitcast %[[PK_I1]] : vector<8xi1> to vector<1xi8>
    %p0 = pack %i1s : tile<8xi1> -> tile<1xi8>

    // Wide -> bytes: 4 x f32 (128 bits) packs into 16 bytes.
    // CHECK: %[[PK_F32:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf32>
    %f32s = constant <f32: [1.0, 2.0, 3.0, 4.0]> : tile<4xf32>
    // CHECK: vector.bitcast %[[PK_F32]] : vector<4xf32> to vector<16xi8>
    %p1 = pack %f32s : tile<4xf32> -> tile<16xi8>

    // Bytes -> wider: 8 x i8 unpacks into 4 x i16.
    // CHECK: %[[UP_I8:.*]] = arith.constant dense<0> : vector<8xi8>
    %i8s = constant <i8: 0> : tile<8xi8>
    // CHECK: vector.bitcast %[[UP_I8]] : vector<8xi8> to vector<4xi16>
    %u0 = unpack %i8s : tile<8xi8> -> tile<4xi16>
    return
  }

  // --- strided_view: get_index_space_shape, all-static dims ---
  // Index space along dim i is ceildiv(tensor_dim[dim_map[i]], traversal_strides[i]):
  //   ceildiv(8192,512)=16, ceildiv(8192,1)=8192, ceildiv(64,10)=7.
  // CHECK-LABEL: gpu.func @test_strided_index_space_static
  // CHECK-SAME: %[[SIS_UPTR:[a-zA-Z0-9_]+]]: memref<*xf32>
  entry @test_strided_index_space_static(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %tv = make_tensor_view %p, shape=[8192, 8192, 64], strides=[524288,64,1] : tensor_view<8192x8192x64xf32, strides=[524288,64,1]>
    %sv = make_strided_view %tv : strided_view<tile=(1024x1x32), traversal_strides=[512,1,10], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>
    // The view aliases the tensor_view, so its memref carries the full tensor shape.
    // CHECK: memref.reinterpret_cast %[[SIS_UPTR]] to offset: [0], sizes: [8192, 8192, 64], strides: [524288, 64, 1] : memref<*xf32> to memref<8192x8192x64xf32, strided<[524288, 64, 1], offset: ?>>
    // The three results are the folded index-space extents, in tile-dim order:
    //   dim0 ceildiv(8192,512)=16, dim1 ceildiv(8192,1)=8192, dim2 ceildiv(64,10)=7.
    // CHECK: %[[SIS_C16:.*]] = arith.constant 16 : index
    // CHECK: arith.index_cast %[[SIS_C16]] : index to i32
    // CHECK: %[[SIS_C8192:.*]] = arith.constant 8192 : index
    // CHECK: arith.index_cast %[[SIS_C8192]] : index to i32
    // CHECK: %[[SIS_C7:.*]] = arith.constant 7 : index
    // CHECK: arith.index_cast %[[SIS_C7]] : index to i32
    %d:3 = get_index_space_shape %sv : strided_view<tile=(1024x1x32), traversal_strides=[512,1,10], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>> -> tile<i32>
    return
  }

  // --- strided_view: get_index_space_shape with a dynamic tensor dim ---
  // Dynamic dim 0 -> memref.dim then ceildivui by the traversal stride (512).
  // CHECK-LABEL: gpu.func @test_strided_index_space_dynamic
  // CHECK-SAME: %[[SISD_UPTR:[a-zA-Z0-9_]+]]: memref<*xf32>
  entry @test_strided_index_space_dynamic(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %m: !cuda_tile.tile<i32>, %s: !cuda_tile.tile<i32>) {
    %tv = make_tensor_view %p, shape=[%m, 8192, 64], strides=[%s, 64, 1] : tile<i32> -> tensor_view<?x8192x64xf32, strides=[?,64,1]>
    %sv = make_strided_view %tv : strided_view<tile=(1024x1x32), traversal_strides=[512,1,10], tensor_view<?x8192x64xf32, strides=[?,64,1]>>
    // CHECK: %[[SISD_PTR:.*]] = memref.reinterpret_cast %[[SISD_UPTR]] to offset: [0], sizes: [%{{.*}}, 8192, 64], strides: [%{{.*}}, 64, 1] : memref<*xf32> to memref<?x8192x64xf32, strided<[?, 64, 1], offset: ?>>
    // dim0 is dynamic: ceildivui(dim0, traversal_strides[0]=512).
    // CHECK: %[[SISD_C0:.*]] = arith.constant 0 : index
    // CHECK: %[[SISD_D0:.*]] = memref.dim %[[SISD_PTR]], %[[SISD_C0]] : memref<?x8192x64xf32, strided<[?, 64, 1], offset: ?>>
    // CHECK: %[[SISD_C512:.*]] = arith.constant 512 : index
    // CHECK: %[[SISD_Q0:.*]] = arith.ceildivui %[[SISD_D0]], %[[SISD_C512]] : index
    // CHECK: arith.index_cast %[[SISD_Q0]] : index to i32
    // dim1 static: ceildiv(8192,1)=8192; dim2 static: ceildiv(64,10)=7.
    // CHECK: %[[SISD_C8192:.*]] = arith.constant 8192 : index
    // CHECK: arith.index_cast %[[SISD_C8192]] : index to i32
    // CHECK: %[[SISD_C7:.*]] = arith.constant 7 : index
    // CHECK: arith.index_cast %[[SISD_C7]] : index to i32
    %d:3 = get_index_space_shape %sv : strided_view<tile=(1024x1x32), traversal_strides=[512,1,10], tensor_view<?x8192x64xf32, strides=[?,64,1]>> -> tile<i32>
    return
  }

  // --- strided_view load: 1D overlapping tiles (stride 1 < tile 2) -> not in bounds ---
  // Tile base advances by traversal_strides[i], not tile_shape[i]; here offset = index*1.
  // (ceildiv(8,1)-1)*1 + 2 = 9 > 8, so the last tile is out of bounds: no in_bounds attr.
  // CHECK-LABEL: gpu.func @test_strided_load_overlap
  // CHECK-SAME: %[[SLO_UPTR:[a-zA-Z0-9_]+]]: memref<*xf32>
  entry @test_strided_load_overlap(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %c0 = constant <i32: 0> : tile<i32>
    %tv = make_tensor_view %p, shape=[8], strides=[1] : tensor_view<8xf32, strides=[1]>
    %sv = make_strided_view %tv : strided_view<tile=(2), traversal_strides=[1], tensor_view<8xf32, strides=[1]>>
    // CHECK: %[[SLO_IDXIN:.*]] = arith.constant 0 : i32
    // CHECK: %[[SLO_PTR:.*]] = memref.reinterpret_cast %[[SLO_UPTR]] to offset: [0], sizes: [8], strides: [1] : memref<*xf32> to memref<8xf32, strided<[1], offset: ?>>
    // offset = index * traversal_strides[0] (= 1 here), not tile_shape.
    // CHECK: %[[SLO_IDX:.*]] = arith.index_cast %[[SLO_IDXIN]] : i32 to index
    // CHECK: %[[SLO_C1:.*]] = arith.constant 1 : index
    // CHECK: %[[SLO_OFF:.*]] = arith.muli %[[SLO_IDX]], %[[SLO_C1]] overflow<nsw> : index
    // CHECK: %[[SLO_PAD:.*]] = ub.poison : f32
    // Overlapping tiles (stride 1 < tile 2): the trailing tile spills out, so NO in_bounds.
    // CHECK: vector.transfer_read %[[SLO_PTR]][%[[SLO_OFF]]], %[[SLO_PAD]] : memref<8xf32, strided<[1], offset: ?>>, vector<2xf32>
    // CHECK-NOT: in_bounds
    %t, %tok = load_view_tko weak %sv[%c0] : strided_view<tile=(2), traversal_strides=[1], tensor_view<8xf32, strides=[1]>>, tile<i32> -> tile<2xf32>, !cuda_tile.token
    return
  }

  // --- strided_view load: 1D exact tiling (stride 2 == tile 2) -> in bounds ---
  // (ceildiv(16,2)-1)*2 + 2 = 16 <= 16, so all tiles fit: in_bounds = [true].
  // CHECK-LABEL: gpu.func @test_strided_load_exact
  // CHECK-SAME: %[[SLE_UPTR:[a-zA-Z0-9_]+]]: memref<*xf32>
  entry @test_strided_load_exact(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %c1 = constant <i32: 1> : tile<i32>
    %tv = make_tensor_view %p, shape=[16], strides=[1] : tensor_view<16xf32, strides=[1]>
    %sv = make_strided_view %tv : strided_view<tile=(2), traversal_strides=[2], tensor_view<16xf32, strides=[1]>>
    // CHECK: %[[SLE_IDXIN:.*]] = arith.constant 1 : i32
    // CHECK: %[[SLE_PTR:.*]] = memref.reinterpret_cast %[[SLE_UPTR]] to offset: [0], sizes: [16], strides: [1] : memref<*xf32> to memref<16xf32, strided<[1], offset: ?>>
    // offset = index * traversal_strides[0] (= 2 here).
    // CHECK: %[[SLE_IDX:.*]] = arith.index_cast %[[SLE_IDXIN]] : i32 to index
    // CHECK: %[[SLE_C2:.*]] = arith.constant 2 : index
    // CHECK: %[[SLE_OFF:.*]] = arith.muli %[[SLE_IDX]], %[[SLE_C2]] overflow<nsw> : index
    // Exact tiling (stride 2 == tile 2): every tile fits, so in_bounds = [true].
    // CHECK: vector.transfer_read %[[SLE_PTR]][%[[SLE_OFF]]], %{{.*}} {in_bounds = [true]} : memref<16xf32, strided<[1], offset: ?>>, vector<2xf32>
    %t, %tok = load_view_tko weak %sv[%c1] : strided_view<tile=(2), traversal_strides=[2], tensor_view<16xf32, strides=[1]>>, tile<i32> -> tile<2xf32>, !cuda_tile.token
    return
  }

  // --- strided_view load: 2D swizzled dim_map + nan padding, mixed in_bounds ---
  // dim_map=[1,0]: tile dim0 -> tensor dim1 (size 16, stride 4), tile dim1 -> tensor dim0 (size 64, stride 3).
  //   dim0: (ceildiv(16,4)-1)*4 + 4 = 16 <= 16 -> true; dim1: (ceildiv(64,3)-1)*3 + 2 = 65 > 64 -> false.
  // CHECK-LABEL: gpu.func @test_strided_load_swizzled
  // CHECK-SAME: %[[SLS_UPTR:[a-zA-Z0-9_]+]]: memref<*xf32>
  entry @test_strided_load_swizzled(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %tv = make_tensor_view %p, shape=[64, 16], strides=[16, 1] : tensor_view<64x16xf32, strides=[16,1]>
    %sv = make_strided_view %tv : strided_view<tile=(4x2), traversal_strides=[4,3], padding_value = nan, tensor_view<64x16xf32, strides=[16,1]>, dim_map=[1, 0]>
    // CHECK: %[[SLS_IDX0IN:.*]] = arith.constant 0 : i32
    // CHECK: %[[SLS_IDX1IN:.*]] = arith.constant 1 : i32
    // CHECK: %[[SLS_PTR:.*]] = memref.reinterpret_cast %[[SLS_UPTR]] to offset: [0], sizes: [64, 16], strides: [16, 1] : memref<*xf32> to memref<64x16xf32, strided<[16, 1], offset: ?>>
    // tile dim0 (index %c0) maps to tensor dim1 with traversal stride 4.
    // CHECK: %[[SLS_IDX0:.*]] = arith.index_cast %[[SLS_IDX0IN]] : i32 to index
    // CHECK: %[[SLS_S4:.*]] = arith.constant 4 : index
    // CHECK: %[[SLS_OFF0:.*]] = arith.muli %[[SLS_IDX0]], %[[SLS_S4]] overflow<nsw> : index
    // tile dim1 (index %c1) maps to tensor dim0 with traversal stride 3.
    // CHECK: %[[SLS_IDX1:.*]] = arith.index_cast %[[SLS_IDX1IN]] : i32 to index
    // CHECK: %[[SLS_S3:.*]] = arith.constant 3 : index
    // CHECK: %[[SLS_OFF1:.*]] = arith.muli %[[SLS_IDX1]], %[[SLS_S3]] overflow<nsw> : index
    // CHECK: %[[SLS_PAD:.*]] = arith.constant 0x7FC00000 : f32
    // Swizzle proof: memref indices are in tensor-dim order [dim0=OFF1, dim1=OFF0].
    // in_bounds: dim0 (size16/stride4/tile4) fits -> true; dim1 (size64/stride3/tile2) spills -> false.
    // CHECK: vector.transfer_read %[[SLS_PTR]][%[[SLS_OFF1]], %[[SLS_OFF0]]], %[[SLS_PAD]] {in_bounds = [true, false], permutation_map = #{{.*}}} : memref<64x16xf32, strided<[16, 1], offset: ?>>, vector<4x2xf32>
    %t, %tok = load_view_tko weak %sv[%c0, %c1] : strided_view<tile=(4x2), traversal_strides=[4,3], padding_value = nan, tensor_view<64x16xf32, strides=[16,1]>, dim_map=[1, 0]>, tile<i32> -> tile<4x2xf32>, !cuda_tile.token
    return
  }

  // --- strided_view store: 2D, both dims in bounds ---
  // dim0: (ceildiv(64,4)-1)*4 + 4 = 64 <= 64 -> true; dim1: (ceildiv(16,2)-1)*2 + 2 = 16 <= 16 -> true.
  // CHECK-LABEL: gpu.func @test_strided_store
  // CHECK-SAME: %[[SST_UPTR:[a-zA-Z0-9_]+]]: memref<*xf32>
  entry @test_strided_store(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %tile = constant <f32: 1.0> : tile<4x2xf32>
    %tv = make_tensor_view %p, shape=[64, 16], strides=[16, 1] : tensor_view<64x16xf32, strides=[16,1]>
    %sv = make_strided_view %tv : strided_view<tile=(4x2), traversal_strides=[4,2], tensor_view<64x16xf32, strides=[16,1]>>
    // CHECK: %[[SST_IDX0IN:.*]] = arith.constant 0 : i32
    // CHECK: %[[SST_IDX1IN:.*]] = arith.constant 1 : i32
    // CHECK: %[[SST_BCAST:.*]] = arith.constant dense<1.000000e+00> : vector<4x2xf32>
    // CHECK: %[[SST_PTR:.*]] = memref.reinterpret_cast %[[SST_UPTR]] to offset: [0], sizes: [64, 16], strides: [16, 1] : memref<*xf32> to memref<64x16xf32, strided<[16, 1], offset: ?>>
    // identity dim_map: tile dim0 uses stride 4, tile dim1 uses stride 2.
    // CHECK: %[[SST_IDX0:.*]] = arith.index_cast %[[SST_IDX0IN]] : i32 to index
    // CHECK: %[[SST_S4:.*]] = arith.constant 4 : index
    // CHECK: %[[SST_OFF0:.*]] = arith.muli %[[SST_IDX0]], %[[SST_S4]] overflow<nsw> : index
    // CHECK: %[[SST_IDX1:.*]] = arith.index_cast %[[SST_IDX1IN]] : i32 to index
    // CHECK: %[[SST_S2:.*]] = arith.constant 2 : index
    // CHECK: %[[SST_OFF1:.*]] = arith.muli %[[SST_IDX1]], %[[SST_S2]] overflow<nsw> : index
    // identity dim_map: memref index order is [OFF0, OFF1], both dims fit -> in_bounds = [true, true], no permutation_map.
    // CHECK: vector.transfer_write %[[SST_BCAST]], %[[SST_PTR]][%[[SST_OFF0]], %[[SST_OFF1]]] {in_bounds = [true, true]} : vector<4x2xf32>, memref<64x16xf32, strided<[16, 1], offset: ?>>
    // CHECK-NOT: permutation_map
    %tok = store_view_tko weak %tile, %sv[%c0, %c1] : tile<4x2xf32>, strided_view<tile=(4x2), traversal_strides=[4,2], tensor_view<64x16xf32, strides=[16,1]>>, tile<i32> -> !cuda_tile.token
    return
  }

  // --- make_gather_scatter_view: view aliases the tensor_view memref (no consumer support) ---
  // CHECK-LABEL: gpu.func @test_make_gather_scatter_view
  // CHECK-SAME: %[[GSV_UPTR:[a-zA-Z0-9_]+]]: memref<*xf32>
  entry @test_make_gather_scatter_view(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %tv = make_tensor_view %p, shape=[8192, 8192], strides=[8192, 1] : tensor_view<8192x8192xf32, strides=[8192,1]>
    // CHECK: %[[GSV_PTR:.*]] = memref.reinterpret_cast %[[GSV_UPTR]] to offset: [0], sizes: [8192, 8192], strides: [8192, 1] : memref<*xf32> to memref<8192x8192xf32, strided<[8192, 1], offset: ?>>
    // CHECK-NOT: make_gather_scatter_view
    %gsv = make_gather_scatter_view %tv : gather_scatter_view<tile=(128x128), tensor_view<8192x8192xf32, strides=[8192,1]>, sparse_dim=0>
    return
  }

}