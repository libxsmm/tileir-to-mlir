// RUN: tileir-to-mlir --convert-tileir-to-mlir %s | FileCheck %s

// A loop whose induction variable indexes a view tile by tile is rescaled to
// iterate over element indices, so that the transfers index with the induction
// variable itself. The rewrite is proven exact with integer range analysis;
// where it cannot be proven, the loop stays in tile space.

// CHECK-LABEL: gpu.func @for_divi_rescale_kernel
cuda_tile.module @for_divi_rescale_module {
  entry @for_divi_rescale_kernel(
      %A_ptr: !cuda_tile.tile<!cuda_tile.ptr<f16>>,
      %M: !cuda_tile.tile<i32>, %K: !cuda_tile.tile<i32>,
      %stride_ak: !cuda_tile.tile<i32>
  ) {
    %A_ptr_assume = assume #cuda_tile.div_by<16>, %A_ptr : tile<ptr<f16>>
    %stride_ak_assume = assume #cuda_tile.div_by<8>, %stride_ak : tile<i32>

    %i0 = constant <i32: 0> : !cuda_tile.tile<i32>
    %i1 = constant <i32: 1> : !cuda_tile.tile<i32>
    %c31 = constant <i32: 31> : !cuda_tile.tile<i32>
    %c32 = constant <i32: 32> : !cuda_tile.tile<i32>
    %cst = constant <f16: 0.000000e+00> : !cuda_tile.tile<128x32xf16>

    %A = make_tensor_view %A_ptr_assume, shape = [%K, %M], strides = [%stride_ak, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %A_block = make_partition_view %A : partition_view<tile=(128x32), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>

    %bidx, %bidy, %bidz = get_tile_block_id : tile<i32>

    // Compute the loop bound as ceildiv(K, 32) using addi + divi by a constant.
    %k_plus = addi %K, %c31 : tile<i32>
    %k_bounded = assume bounded<0, ?>, %k_plus : tile<i32>
    %ub = divi %k_bounded, %c32 signed : tile<i32>

    // CHECK: %[[C32_STEP:.*]] = arith.constant 32 : index
    // CHECK: %{{.*}} = arith.muli %{{.*}}, %[[C32_STEP]] : index
    // CHECK: %{{.*}} = arith.muli %{{.*}}, %[[C32_STEP]] : index
    // CHECK: %[[LOOP_STEP:.*]] = arith.muli %{{.*}}, %[[C32_STEP]] : index
    // CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %[[LOOP_STEP]]
    %result = for %k in (%i0 to %ub, step %i1) : tile<i32>
        iter_values(%acc_prev = %cst) -> (tile<128x32xf16>)
    {
      // Because dim_map=[1, 0], the K (loop) dimension is the first index.
      // CHECK: vector.transfer_read %{{.*}}[%[[IV]], %{{.*}}]
      %A_frag, %t1 = load_view_tko weak %A_block[%bidx, %k] : partition_view<tile=(128x32), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<128x32xf16>, !cuda_tile.token
      continue %A_frag : tile<128x32xf16>
    }
    return
  }
}

// CHECK-LABEL: gpu.func @for_wrapped_iv_index_rescale_kernel
cuda_tile.module @for_wrapped_iv_index_rescale_module {
  entry @for_wrapped_iv_index_rescale_kernel(
      %A_ptr: !cuda_tile.tile<!cuda_tile.ptr<f16>>,
      %M: !cuda_tile.tile<i32>, %K: !cuda_tile.tile<i32>,
      %stride_ak: !cuda_tile.tile<i32>
  ) {
    %A_ptr_assume = assume #cuda_tile.div_by<16>, %A_ptr : tile<ptr<f16>>
    %stride_ak_assume = assume #cuda_tile.div_by<8>, %stride_ak : tile<i32>

    %i0 = constant <i32: 0> : !cuda_tile.tile<i32>
    %i1 = constant <i32: 1> : !cuda_tile.tile<i32>
    %c31 = constant <i32: 31> : !cuda_tile.tile<i32>
    %c32 = constant <i32: 32> : !cuda_tile.tile<i32>
    %cst = constant <f16: 0.000000e+00> : !cuda_tile.tile<128x32xf16>

    %A = make_tensor_view %A_ptr_assume, shape = [%K, %M], strides = [%stride_ak, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %A_block = make_partition_view %A : partition_view<tile=(128x32), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>

    %bidx, %bidy, %bidz = get_tile_block_id : tile<i32>
    %k_plus = addi %K, %c31 : tile<i32>
    %k_bounded = assume bounded<0, ?>, %k_plus : tile<i32>
    %ub = divi %k_bounded, %c32 signed : tile<i32>

    // CHECK: %[[WRAPPED_C32_STEP:.*]] = arith.constant 32 : index
    // CHECK: %{{.*}} = arith.muli %{{.*}}, %[[WRAPPED_C32_STEP]] : index
    // CHECK: %{{.*}} = arith.muli %{{.*}}, %[[WRAPPED_C32_STEP]] : index
    // CHECK: %[[WRAPPED_LOOP_STEP:.*]] = arith.muli %{{.*}}, %[[WRAPPED_C32_STEP]] : index
    // CHECK: scf.for %[[WRAPPED_IV:.*]] = %{{.*}} to %{{.*}} step %[[WRAPPED_LOOP_STEP]]
    %result = for %k in (%i0 to %ub, step %i1) : tile<i32>
        iter_values(%acc_prev = %cst) -> (tile<128x32xf16>)
    {
      // The upper bound keeps %k below 2^26, so `(%k * 32) / 32` equals %k: the
      // transfer indexes with the element-space induction variable and the
      // tile index computation is removed.
      // CHECK-NOT: arith.divsi
      // CHECK: vector.transfer_read %{{.*}}[%[[WRAPPED_IV]], %{{.*}}]
      %scaled_k = muli %k, %c32 : tile<i32>
      %wrapped_k = divi %scaled_k, %c32 signed : tile<i32>
      %A_frag, %t1 = load_view_tko weak %A_block[%bidx, %wrapped_k] : partition_view<tile=(128x32), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<128x32xf16>, !cuda_tile.token
      continue %A_frag : tile<128x32xf16>
    }
    return
  }
}

// Without a bound on %k, `(%k * 32) / 32` may differ from %k because the
// multiplication can overflow, so the loop stays in tile space.

// CHECK-LABEL: gpu.func @for_unbounded_wrapped_iv_kernel
// CHECK: scf.for %[[UNBOUNDED_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK: %[[UNBOUNDED_IV_I32:.*]] = arith.index_cast %[[UNBOUNDED_IV]] : index to i32
// CHECK: arith.muli %[[UNBOUNDED_IV_I32]], %{{.*}} : i32
cuda_tile.module @for_unbounded_wrapped_iv_module {
  entry @for_unbounded_wrapped_iv_kernel(%A: !cuda_tile.tensor_view<?xf32, strides=[1]>,
                                         %n: !cuda_tile.tile<i32>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %c32 = constant <i32: 32> : tile<i32>
    %pv = make_partition_view %A : partition_view<tile=(32), tensor_view<?xf32, strides=[1]>>
    for %k in (%c0 to %n, step %c1) : tile<i32> {
      %scaled_k = muli %k, %c32 : tile<i32>
      %wrapped_k = divi %scaled_k, %c32 signed : tile<i32>
      %x, %t = load_view_tko weak %pv[%wrapped_k] : partition_view<tile=(32), tensor_view<?xf32, strides=[1]>>, tile<i32> -> tile<32xf32>, !cuda_tile.token
      continue
    }
    return
  }
}

// Only transfer indices are rewritten: `(%iv / 3) * 3` computed for a store is
// kept, as it differs from %iv for negative %iv.

// CHECK-LABEL: gpu.func @scalar_div_mul_kept
// CHECK: %[[DIV:.*]] = arith.divui %{{.*}}, %{{.*}} : i32
// CHECK: %[[MUL:.*]] = arith.muli %[[DIV]], %{{.*}} : i32
// CHECK: memref.store %[[MUL]]
cuda_tile.module @scalar_div_mul_module {
  entry @scalar_div_mul_kept(%p: !cuda_tile.tile<!cuda_tile.ptr<i32>>) {
    %lb = constant <i32: -3> : tile<i32>
    %ub = constant <i32: 3> : tile<i32>
    %c3 = constant <i32: 3> : tile<i32>
    for %iv in (%lb to %ub, step %c3) : tile<i32> {
      %d = divi %iv, %c3 unsigned : tile<i32>
      %m = muli %d, %c3 : tile<i32>
      %t = store_ptr_tko weak %p, %m : tile<ptr<i32>>, tile<i32> -> !cuda_tile.token
      continue
    }
    return
  }
}
