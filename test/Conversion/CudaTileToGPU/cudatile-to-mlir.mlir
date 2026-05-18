// RUN: cudatile-to-gpu --convert-cuda-tile-to-gpu %s | FileCheck %s
// RUN: cudatile-to-gpu --convert-cuda-tile-to-gpu %s | mlir-opt --loop-invariant-code-motion -cse -canonicalize -cse > /dev/null

// CHECK-LABEL: gpu.module @m {
cuda_tile.module @m {
  // CHECK-LABEL: gpu.func @test_module_and_entry(
  // CHECK-SAME: memref<*xf32>
  entry @test_module_and_entry(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // CHECK: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [8], strides: [1] : memref<*xf32> to memref<8xf32>
    // CHECK-NOT: cuda_tile.make_tensor_view
    %tv = make_tensor_view %p, shape = [8], strides = [1]
      : tensor_view<8xf32, strides=[1]>
    // CHECK-NOT: cuda_tile.make_partition_view
    %pv = make_partition_view %tv
      : partition_view<tile=(8), tensor_view<8xf32, strides=[1]>>
  }

  // CHECK-LABEL: gpu.func @test_constant_variants
  entry @test_constant_variants() {
    // CHECK: arith.constant 7 : i32
    %c0 = constant <i32: 7> : tile<i32>

    // CHECK: arith.constant 2.500000e+00 : f32
    %c1 = constant <f32: 2.500000e+00> : tile<f32>

    // CHECK: arith.constant 3 : i32
    // CHECK: vector.broadcast
    %c2 = constant <i32: 3> : tile<2x4xi32>

    // CHECK: arith.constant dense<{{.*}}> : vector<2x2xi32>
    %c3 = constant <i32: [[1, 2], [3, 4]]> : tile<2x2xi32>
  }

  // CHECK-LABEL: gpu.func @test_block_id_and_muli
  entry @test_block_id_and_muli() {
    // CHECK: gpu.block_id x
    // CHECK: arith.index_cast {{.*}} : index to i32
    // CHECK: gpu.block_id y
    // CHECK: arith.index_cast {{.*}} : index to i32
    // CHECK: gpu.block_id z
    // CHECK: arith.index_cast {{.*}} : index to i32
    %x, %y, %z = get_tile_block_id : tile<i32>

    %a = constant <i32: 6> : tile<i32>
    %b = constant <i32: 7> : tile<i32>
    // CHECK: arith.muli
    %m = muli %a, %b : tile<i32>
  }

  // CHECK-LABEL: gpu.func @test_for_continue
  entry @test_for_continue() {
    %lb = constant <i32: 0> : tile<i32>
    %ub = constant <i32: 4> : tile<i32>
    %st = constant <i32: 1> : tile<i32>
    %init = constant <f32: 0.000000e+00> : tile<2x2xf32>

    %res = for %iv in (%lb to %ub, step %st) : tile<i32>
      iter_values(%acc = %init) -> (tile<2x2xf32>) {
      // CHECK: arith.index_cast {{.*}} : i32 to index
      // CHECK: arith.index_cast {{.*}} : i32 to index
      // CHECK: arith.index_cast {{.*}} : i32 to index
      // CHECK: scf.for
      // CHECK: arith.index_cast {{.*}} : index to i32
      // CHECK: scf.yield
      continue %acc : tile<2x2xf32>
    }
  }

  // CHECK-LABEL: gpu.func @test_return
  entry @test_return() {
    // CHECK: gpu.return
    return
  }

  // CHECK-LABEL: gpu.func @test_mmaf_2d
  entry @test_mmaf_2d() {
    %a = constant <f16: 1.000000e+00> : tile<4x8xf16>
    %b = constant <f16: 2.000000e+00> : tile<8x4xf16>
    %c = constant <f32: 0.000000e+00> : tile<4x4xf32>
    // CHECK: %[[MMAF2D_A_C:.*]] = arith.constant 1.000000e+00 : f16
    // CHECK: %[[MMAF2D_A:.*]] = vector.broadcast %[[MMAF2D_A_C]] : f16 to vector<4x8xf16>
    // CHECK: %[[MMAF2D_B_C:.*]] = arith.constant 2.000000e+00 : f16
    // CHECK: %[[MMAF2D_B:.*]] = vector.broadcast %[[MMAF2D_B_C]] : f16 to vector<8x4xf16>
    // CHECK: %[[MMAF2D_ACC_C:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[MMAF2D_ACC:.*]] = vector.broadcast %[[MMAF2D_ACC_C]] : f32 to vector<4x4xf32>
    // CHECK: %[[MMAF2D_R:.*]] = vector.contract
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
    // CHECK-SAME: kind = #vector.kind<add>
    // CHECK-SAME: %[[MMAF2D_A]], %[[MMAF2D_B]], %[[MMAF2D_ACC]]
    // CHECK-SAME: vector<4x8xf16>, vector<8x4xf16> into vector<4x4xf32>
    %r = mmaf %a, %b, %c : tile<4x8xf16>, tile<8x4xf16>, tile<4x4xf32>
  }

  // CHECK-LABEL: gpu.func @test_mmaf_3d
  entry @test_mmaf_3d() {
    %a = constant <f16: 1.000000e+00> : tile<2x4x8xf16>
    %b = constant <f16: 2.000000e+00> : tile<2x8x4xf16>
    %c = constant <f32: 0.000000e+00> : tile<2x4x4xf32>
    // CHECK: %[[MMAF3D_A_C:.*]] = arith.constant 1.000000e+00 : f16
    // CHECK: %[[MMAF3D_A:.*]] = vector.broadcast %[[MMAF3D_A_C]] : f16 to vector<2x4x8xf16>
    // CHECK: %[[MMAF3D_B_C:.*]] = arith.constant 2.000000e+00 : f16
    // CHECK: %[[MMAF3D_B:.*]] = vector.broadcast %[[MMAF3D_B_C]] : f16 to vector<2x8x4xf16>
    // CHECK: %[[MMAF3D_ACC_C:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[MMAF3D_ACC:.*]] = vector.broadcast %[[MMAF3D_ACC_C]] : f32 to vector<2x4x4xf32>
    // CHECK: %[[MMAF3D_R:.*]] = vector.contract
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    // CHECK-SAME: kind = #vector.kind<add>
    // CHECK-SAME: %[[MMAF3D_A]], %[[MMAF3D_B]], %[[MMAF3D_ACC]]
    // CHECK-SAME: vector<2x4x8xf16>, vector<2x8x4xf16> into vector<2x4x4xf32>
    %r = mmaf %a, %b, %c : tile<2x4x8xf16>, tile<2x8x4xf16>, tile<2x4x4xf32>
  }

  // CHECK-LABEL: gpu.func @test_mmai_2d
  entry @test_mmai_2d() {
    %ai = constant <i8: 0> : tile<4x8xi8>
    %bi = constant <i8: 0> : tile<8x2xi8>
    %acci = constant <i32: 0> : tile<4x2xi32>
    // CHECK: %[[MMAI_C0_I8_A:.*]] = arith.constant 0 : i8
    // CHECK: %[[MMAI_A:.*]] = vector.broadcast %[[MMAI_C0_I8_A]] : i8 to vector<4x8xi8>
    // CHECK: %[[MMAI_C0_I8_B:.*]] = arith.constant 0 : i8
    // CHECK: %[[MMAI_B:.*]] = vector.broadcast %[[MMAI_C0_I8_B]] : i8 to vector<8x2xi8>
    // CHECK: %[[MMAI_C0_I32_ACC:.*]] = arith.constant 0 : i32
    // CHECK: %[[MMAI_ACC:.*]] = vector.broadcast %[[MMAI_C0_I32_ACC]] : i32 to vector<4x2xi32>
    // CHECK: %[[MMAI_R:.*]] = vector.contract
    // CHECK-SAME: indexing_maps = [#map, #map1, #map2]
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
    // CHECK-SAME: kind = #vector.kind<add>
    // CHECK-SAME: %[[MMAI_A]], %[[MMAI_B]], %[[MMAI_ACC]]
    %mmai = mmai %ai, %bi, %acci signed signed : tile<4x8xi8>, tile<8x2xi8>, tile<4x2xi32>
  }

  // CHECK-LABEL: gpu.func @test_mmai_3d
  entry @test_mmai_3d() {
    %ai = constant <i8: 0> : tile<2x4x8xi8>
    %bi = constant <i8: 0> : tile<2x8x2xi8>
    %acci = constant <i32: 0> : tile<2x4x2xi32>
    // CHECK: %[[MMAIB_A_C0:.*]] = arith.constant 0 : i8
    // CHECK: %[[MMAIB_A:.*]] = vector.broadcast %[[MMAIB_A_C0]] : i8 to vector<2x4x8xi8>
    // CHECK: %[[MMAIB_B_C0:.*]] = arith.constant 0 : i8
    // CHECK: %[[MMAIB_B:.*]] = vector.broadcast %[[MMAIB_B_C0]] : i8 to vector<2x8x2xi8>
    // CHECK: %[[MMAIB_ACC_C0:.*]] = arith.constant 0 : i32
    // CHECK: %[[MMAIB_ACC:.*]] = vector.broadcast %[[MMAIB_ACC_C0]] : i32 to vector<2x4x2xi32>
    // CHECK: %[[MMAIB_R:.*]] = vector.contract
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    // CHECK-SAME: kind = #vector.kind<add>
    // CHECK-SAME: %[[MMAIB_A]], %[[MMAIB_B]], %[[MMAIB_ACC]]
    // CHECK-SAME: vector<2x4x8xi8>, vector<2x8x2xi8> into vector<2x4x2xi32>
    %mmai = mmai %ai, %bi, %acci signed signed : tile<2x4x8xi8>, tile<2x8x2xi8>, tile<2x4x2xi32>
  }

  // CHECK-LABEL: gpu.func @test_assume_passthrough
  entry @test_assume_passthrough() {
    %a0 = constant <i32: 5> : tile<i32>
    %a1 = assume #cuda_tile.div_by<8>, %a0 : tile<i32>
    %b = constant <i32: 9> : tile<i32>
    // CHECK: arith.muli
    // CHECK-NOT: cuda_tile.assume
    %m = muli %a1, %b : tile<i32>
  }

  // CHECK-LABEL: gpu.func @test_new_math_and_int_ops
  entry @test_new_math_and_int_ops() {
    %xf = constant <f32: [1.000000e+00, -1.000000e+00, 0.000000e+00, 2.000000e+00]> : tile<4xf32>
    %yf = constant <f32: [1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00]> : tile<4xf32>
    %zi = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
    %wi = constant <i32: [4, 5, 6, 7]> : tile<4xi32>

    // CHECK: math.atan2
    %atan2 = atan2 %xf, %yf : tile<4xf32>
    // CHECK: math.ceil
    %ceil = ceil %xf : tile<4xf32>
    // CHECK: arith.cmpf olt
    %cmpf = cmpf less_than ordered %xf, %yf : tile<4xf32> -> tile<4xi1>
    // CHECK: arith.cmpf ueq
    %cmpf_unordered = cmpf equal unordered %xf, %yf : tile<4xf32> -> tile<4xi1>
    // CHECK: math.cos
    %cos = cos %xf : tile<4xf32>
    // CHECK: math.exp2
    %exp2 = exp2 %xf : tile<4xf32>
    // CHECK: math.exp
    %exp = exp %xf : tile<4xf32>
    // CHECK: math.floor
    %floor = floor %xf : tile<4xf32>
    // CHECK: math.log2
    %log2 = log2 %xf : tile<4xf32>
    // CHECK: arith.maxnumf
    %maxf = maxf %xf, %yf : tile<4xf32>
    // CHECK: arith.minnumf
    %minf = minf %xf, %yf : tile<4xf32>
    // CHECK: arith.negf
    %negf = negf %xf : tile<4xf32>
    // CHECK: math.powf
    %pow = pow %xf, %yf : tile<4xf32>
    // CHECK: math.rsqrt
    %rsqrt = rsqrt %xf : tile<4xf32>
    // CHECK: math.sin
    %sin = sin %xf : tile<4xf32>
    // CHECK: math.tanh
    %tanh = tanh %xf rounding<full> : tile<4xf32>

    // CHECK: arith.cmpi slt
    %cmpi = cmpi less_than %zi, %wi, signed : tile<4xi32> -> tile<4xi1>
    // CHECK: arith.cmpi ult
    %cmpi_u = cmpi less_than %zi, %wi, unsigned : tile<4xi32> -> tile<4xi1>
    // CHECK: arith.maxui
    %maxi = maxi %zi, %wi unsigned : tile<4xi32>
    // CHECK: arith.maxsi
    %maxi_s = maxi %zi, %wi signed : tile<4xi32>
    // CHECK: arith.minsi
    %mini = mini %zi, %wi signed : tile<4xi32>
    // CHECK: arith.minui
    %mini_u = mini %zi, %wi unsigned : tile<4xi32>
    // CHECK: arith.mului_extended
    %mulhi = mulhii %zi, %wi : tile<4xi32>
    // CHECK: arith.subi
    %negi = negi %zi : tile<4xi32>
    // CHECK: arith.xori
    %xori = xori %zi, %wi : tile<4xi32>

  }


  // CHECK-LABEL: gpu.func @test_shape_identity
  // CHECK-SAME: %[[SID_UPTR:[a-zA-Z0-9_]+]]: memref<*xf16>
  entry @test_shape_identity(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %m: !cuda_tile.tile<i32>, %n: !cuda_tile.tile<i32>, %s: !cuda_tile.tile<i32>) {
    %tv = make_tensor_view %p, shape = [%m, %n], strides = [%s, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %pv = make_partition_view %tv : partition_view<tile=(16x8), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>
    // CHECK: %[[SID_PTR:.*]] = memref.reinterpret_cast %[[SID_UPTR]]
    // CHECK: %[[SID_D0:.*]] = memref.dim %[[SID_PTR]], {{.*}}0
    // CHECK: %[[SID_C16:.*]] = arith.constant 16 : index
    // CHECK: %[[SID_Q0:.*]] = arith.ceildivui %[[SID_D0]], %[[SID_C16]] : index
    // CHECK: arith.index_cast %[[SID_Q0]] : index to i32
    // CHECK: %[[SID_D1:.*]] = memref.dim %[[SID_PTR]], {{.*}}1
    // CHECK: %[[SID_C8:.*]] = arith.constant 8 : index
    // CHECK: %[[SID_Q1:.*]] = arith.ceildivui %[[SID_D1]], %[[SID_C8]] : index
    // CHECK: arith.index_cast %[[SID_Q1]] : index to i32
    %dims:2 = get_index_space_shape %pv : partition_view<tile=(16x8), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]> -> tile<i32>
  }

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