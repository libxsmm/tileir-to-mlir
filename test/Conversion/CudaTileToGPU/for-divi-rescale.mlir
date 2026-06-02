// RUN: cudatile-to-gpu --convert-cuda-tile-to-gpu %s | FileCheck %s

// Verifies that when the upper bound of a cuda_tile.for loop is produced by
// `cuda_tile.divi <expr>, <constant N>`, the loop is rescaled so that the
// step is multiplied by N (the tile size) and the redundant `iv * N`
// multiplication used for indexing inside the body is folded away by the
// post-conversion cleanup.

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
      // The transfer_read should index with %[[IV]] directly (the
      // post-conversion cleanup folds `muli(divui(iv, 32), 32) -> iv`).
      // Because dim_map=[1, 0], the K (loop) dimension is the first index.
      // CHECK: vector.transfer_read %{{.*}}[%[[IV]], %{{.*}}]
      %A_frag, %t1 = load_view_tko weak %A_block[%bidx, %k] : partition_view<tile=(128x32), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<128x32xf16>, token
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
      %scaled_k = muli %k, %c32 : tile<i32>
      %wrapped_k = divi %scaled_k, %c32 signed : tile<i32>
      // CHECK: vector.transfer_read %{{.*}}[%[[WRAPPED_IV]], %{{.*}}]
      %A_frag, %t1 = load_view_tko weak %A_block[%bidx, %wrapped_k] : partition_view<tile=(128x32), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<128x32xf16>, token
      continue %A_frag : tile<128x32xf16>
    }
    return
  }
}
