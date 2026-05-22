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
    // CHECK: %[[STI_BCAST:.*]] = arith.constant dense<1.000000e+00> : vector<4x2xf16>
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
    // CHECK: %[[STS_BCAST:.*]] = arith.constant dense<1.000000e+00> : vector<4x2xf16>
    // CHECK: %[[STS_PTR:.*]] = memref.reinterpret_cast %[[STS_UPTR]]
    // CHECK: vector.transfer_write %[[STS_BCAST]], %[[STS_PTR]][%{{.*}}, %{{.*}}] {permutation_map = #{{.*}}} : vector<4x2xf16>, memref<?x?xf16, strided<[?, 1]>>
    %tok = store_view_tko weak %tile, %pv[%c0, %c1] : tile<4x2xf16>, partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> token
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
}