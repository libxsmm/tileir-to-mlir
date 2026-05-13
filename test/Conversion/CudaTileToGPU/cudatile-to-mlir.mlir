// RUN: cudatile-to-gpu --convert-cuda-tile-to-gpu %s | FileCheck %s

// CHECK-LABEL: gpu.module @m {
cuda_tile.module @m {
  // CHECK-LABEL: gpu.func @test_module_and_entry(
  // CHECK-SAME: memref<8xf32>
  entry @test_module_and_entry(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // CHECK-NOT: cuda_tile.make_tensor_view
    %tv = make_tensor_view %p, shape = [8], strides = [1]
      : tensor_view<8xf32, strides=[1]>
    // CHECK-NOT: cuda_tile.make_partition_view
    %pv = make_partition_view %tv
      : partition_view<tile=(8), tensor_view<8xf32, strides=[1]>>
  }

  // CHECK-LABEL: gpu.func @test_constant_variants
  entry @test_constant_variants() {
    // CHECK: arith.constant 7 : index
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
    // CHECK: gpu.block_id y
    // CHECK: gpu.block_id z
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
      // CHECK: scf.for
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
    // CHECK: vector.contract
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
    // CHECK-SAME: kind = #vector.kind<add>
    %r = mmaf %a, %b, %c : tile<4x8xf16>, tile<8x4xf16>, tile<4x4xf32>
  }

  // CHECK-LABEL: gpu.func @test_mmaf_3d
  entry @test_mmaf_3d() {
    %a = constant <f16: 1.000000e+00> : tile<2x4x8xf16>
    %b = constant <f16: 2.000000e+00> : tile<2x8x4xf16>
    %c = constant <f32: 0.000000e+00> : tile<2x4x4xf32>
    // CHECK: vector.contract
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    // CHECK-SAME: kind = #vector.kind<add>
    %r = mmaf %a, %b, %c : tile<2x4x8xf16>, tile<2x8x4xf16>, tile<2x4x4xf32>
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

  // CHECK-LABEL: gpu.func @test_shape_identity
  entry @test_shape_identity(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %m: !cuda_tile.tile<i32>, %n: !cuda_tile.tile<i32>, %s: !cuda_tile.tile<i32>) {
    %tv = make_tensor_view %p, shape = [%m, %n], strides = [%s, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %pv = make_partition_view %tv : partition_view<tile=(16x8), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>
    // CHECK: memref.dim {{.*}}, {{.*}}0
    // CHECK: arith.constant 16 : index
    // CHECK: memref.dim {{.*}}, {{.*}}1
    // CHECK: arith.constant 8 : index
    // CHECK: arith.ceildivui
    %dims:2 = get_index_space_shape %pv : partition_view<tile=(16x8), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]> -> tile<i32>
  }

  // CHECK-LABEL: gpu.func @test_shape_swizzled
  entry @test_shape_swizzled(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %m: !cuda_tile.tile<i32>, %n: !cuda_tile.tile<i32>, %s: !cuda_tile.tile<i32>) {
    %tv = make_tensor_view %p, shape = [%m, %n], strides = [%s, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %pv = make_partition_view %tv : partition_view<tile=(16x8), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
    // CHECK: memref.dim {{.*}}, {{.*}}1
    // CHECK: arith.constant 16 : index
    // CHECK: memref.dim {{.*}}, {{.*}}0
    // CHECK: arith.constant 8 : index
    // CHECK: arith.ceildivui
    %dims:2 = get_index_space_shape %pv : partition_view<tile=(16x8), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]> -> tile<i32>
  }

  // CHECK-LABEL: gpu.func @test_load_identity_no_padding
  entry @test_load_identity_no_padding(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %m: !cuda_tile.tile<i32>, %n: !cuda_tile.tile<i32>, %s: !cuda_tile.tile<i32>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %tv = make_tensor_view %p, shape = [%m, %n], strides = [%s, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %pv = make_partition_view %tv : partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>
    // CHECK: ub.poison : f16
    // CHECK: vector.transfer_read
    // CHECK-NOT: permutation_map
    %tile, %tok = load_view_tko weak %pv[%c0, %c1] : partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> tile<4x2xf16>, token
  }

  // CHECK-LABEL: gpu.func @test_load_swizzled_with_padding
  entry @test_load_swizzled_with_padding(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %m: !cuda_tile.tile<i32>, %n: !cuda_tile.tile<i32>, %s: !cuda_tile.tile<i32>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %tv = make_tensor_view %p, shape = [%m, %n], strides = [%s, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %pv = make_partition_view %tv : partition_view<tile=(4x2), padding_value = zero, tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
    // CHECK: arith.constant 0.000000e+00 : f16
    // CHECK: vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {permutation_map = #{{.*}}} : memref<?x?xf16, strided<[?, 1]>>, vector<4x2xf16>
    %tile, %tok = load_view_tko weak %pv[%c0, %c1] : partition_view<tile=(4x2), padding_value = zero, tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<4x2xf16>, token
  }

  // CHECK-LABEL: gpu.func @test_store_identity
  entry @test_store_identity(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %m: !cuda_tile.tile<i32>, %n: !cuda_tile.tile<i32>, %s: !cuda_tile.tile<i32>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %tile = constant <f16: 1.000000e+00> : tile<4x2xf16>
    %tv = make_tensor_view %p, shape = [%m, %n], strides = [%s, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %pv = make_partition_view %tv : partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>
    // CHECK: vector.transfer_write
    // CHECK-NOT: permutation_map
    %tok = store_view_tko weak %tile, %pv[%c0, %c1] : tile<4x2xf16>, partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> token
  }

  // CHECK-LABEL: gpu.func @test_store_swizzled
  entry @test_store_swizzled(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %m: !cuda_tile.tile<i32>, %n: !cuda_tile.tile<i32>, %s: !cuda_tile.tile<i32>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %tile = constant <f16: 1.000000e+00> : tile<4x2xf16>
    %tv = make_tensor_view %p, shape = [%m, %n], strides = [%s, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %pv = make_partition_view %tv : partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
    // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] {permutation_map = #{{.*}}} : vector<4x2xf16>, memref<?x?xf16, strided<[?, 1]>>
    %tok = store_view_tko weak %tile, %pv[%c0, %c1] : tile<4x2xf16>, partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> token
  }
}