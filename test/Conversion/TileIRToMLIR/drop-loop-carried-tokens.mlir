// RUN: tileir-to-mlir --convert-tileir-to-mlir %s | FileCheck %s

// Tokens only order memory operations, which the lowered IR orders by program
// order, so they convert to no values. A cuda_tile.for loop carrying a token
// iter value (make_token -> loop iter_value -> store_view_tko's `token` operand
// -> its result_token -> continue) lowers to an scf.for without that iter
// value, and so does a token yielded by cuda_tile.if.

// CHECK-LABEL: gpu.func @drop_token_kernel
cuda_tile.module @drop_token_module {
  entry @drop_token_kernel(
      %C: !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>
  ) {
    %i0 = constant <i32: 0> : !cuda_tile.tile<i32>
    %i1 = constant <i32: 1> : !cuda_tile.tile<i32>
    %ub = constant <i32: 8> : !cuda_tile.tile<i32>
    %cst = constant <f32: 0.000000e+00> : !cuda_tile.tile<256x128xf32>

    %C_block = make_partition_view %C : partition_view<tile=(256x128), tensor_view<4096x4096xf32, strides=[4096,1]>, dim_map=[0, 1]>
    %bidx, %bidy, %bidz = get_tile_block_id : tile<i32>

    %tok0 = make_token : !cuda_tile.token

    // The loop must lower to an scf.for with *no* iter args: the token iter
    // value is dropped, so `step` is immediately followed by the loop body `{`.
    // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
    %result = for %k in (%i0 to %ub, step %i1) : tile<i32>
        iter_values(%tok_prev = %tok0) -> (!cuda_tile.token)
    {
      // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]
      %tok_next = store_view_tko weak %cst, %C_block[%bidx, %k] token = %tok_prev : tile<256x128xf32>, partition_view<tile=(256x128), tensor_view<4096x4096xf32, strides=[4096,1]>, dim_map=[0, 1]>, tile<i32> -> !cuda_tile.token
      continue %tok_next : !cuda_tile.token
    }
    // CHECK: gpu.return
    return
  }

  // CHECK-LABEL: gpu.func @drop_straight_line_tokens
  entry @drop_straight_line_tokens(
      %A: !cuda_tile.tensor_view<2x4xf32, strides=[4,1]>,
      %B: !cuda_tile.tensor_view<2x4xf32, strides=[4,1]>
  ) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    %A_block = make_partition_view %A : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>, dim_map=[0, 1]>
    %B_block = make_partition_view %B : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>, dim_map=[0, 1]>
    %tok0 = make_token : !cuda_tile.token

    // CHECK: %[[TILE:.*]] = vector.transfer_read
    %tile, %tok1 = load_view_tko weak %A_block[%c0, %c0] token = %tok0 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>, dim_map=[0, 1]>, tile<i32> -> tile<2x4xf32>, !cuda_tile.token
    // CHECK: vector.transfer_write %[[TILE]]
    %tok2 = store_view_tko weak %tile, %B_block[%c0, %c0] token = %tok1 : tile<2x4xf32>, partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>, dim_map=[0, 1]>, tile<i32> -> !cuda_tile.token
    return
  }

  // CHECK-LABEL: gpu.func @drop_if_token
  entry @drop_if_token(%p: !cuda_tile.tile<!cuda_tile.ptr<i32>>, %c: !cuda_tile.tile<i1>) {
    %v = constant <i32: 1> : tile<i32>
    %t0 = make_token : !cuda_tile.token
    // CHECK: scf.if %{{.*}} {
    // CHECK:   memref.store
    // CHECK: } else {
    // CHECK-NEXT: }
    // CHECK: memref.store
    %r = if %c -> (!cuda_tile.token) {
      %t1 = store_ptr_tko weak %p, %v token = %t0 : tile<ptr<i32>>, tile<i32> -> !cuda_tile.token
      yield %t1 : !cuda_tile.token
    } else {
      yield %t0 : !cuda_tile.token
    }
    %t2 = store_ptr_tko weak %p, %v token = %r : tile<ptr<i32>>, tile<i32> -> !cuda_tile.token
    return
  }

  // Dropping the token iter value keeps the unsigned comparison.
  // CHECK-LABEL: gpu.func @drop_token_unsigned_loop
  entry @drop_token_unsigned_loop(%p: !cuda_tile.tile<!cuda_tile.ptr<i32>>) {
    %c0 = constant <i32: 0> : tile<i32>
    %ub = constant <i32: 16> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %t0 = make_token : !cuda_tile.token
    // CHECK: scf.for unsigned %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
    %r = for unsigned %iv in (%c0 to %ub, step %c1) : tile<i32>
        iter_values(%t = %t0) -> (!cuda_tile.token) {
      %t1 = store_ptr_tko weak %p, %iv token = %t : tile<ptr<i32>>, tile<i32> -> !cuda_tile.token
      continue %t1 : !cuda_tile.token
    }
    return
  }
}
