// RUN: tileir-to-mlir --convert-tileir-to-mlir='target=gpu append-grid-args=false' %s | FileCheck %s

// A select that rewrites lanes with the loaded view's own padding value is
// folded into the transfer_read mask, so those lanes are never loaded. The
// in_bounds flags are unchanged, so lanes the mask leaves enabled keep their
// out-of-bounds protection.

// CHECK-LABEL: gpu.func @fuse_mask_into_load
// CHECK: %[[MASK:.*]] = arith.cmpi slt, %{{.*}} : vector<32xi32>
// CHECK: memref.reinterpret_cast
// CHECK: %[[PAD:.*]] = arith.constant 0xFF800000 : f32
// CHECK-NEXT: vector.transfer_read %{{.*}}[%{{.*}}], %[[PAD]], %[[MASK]] : memref<?xf32, strided<[1], offset: ?>>, vector<32xf32>
// CHECK-NOT: arith.select

// A padding value that differs from the view's padding is not foldable: the
// masked-off lanes would change value.

// CHECK-LABEL: gpu.func @no_fuse_padding_mismatch
// CHECK: %[[RD0:.*]] = vector.transfer_read
// CHECK-NEXT: arith.select %{{.*}}, %[[RD0]], %{{.*}} : vector<32xi1>, vector<32xf32>

// A load feeding more than the select must keep producing the unmasked tile.

// CHECK-LABEL: gpu.func @no_fuse_multi_use
// CHECK: %[[RD1:.*]] = vector.transfer_read
// CHECK-NEXT: arith.select %{{.*}}, %[[RD1]], %{{.*}} : vector<32xi1>, vector<32xf32>

cuda_tile.module @m {
  entry @fuse_mask_into_load(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %n: !cuda_tile.tile<i32>) {
    %c0 = constant <i32: 0> : tile<i32>
    %lim = constant <i32: 64> : tile<i32>
    %neg_inf = constant <f32: 0xFF800000> : tile<f32>
    %r_pad = reshape %neg_inf : tile<f32> -> tile<1xf32>
    %pad = broadcast %r_pad : tile<1xf32> -> tile<32xf32>
    %r_lim = reshape %lim : tile<i32> -> tile<1xi32>
    %b_lim = broadcast %r_lim : tile<1xi32> -> tile<32xi32>
    %iota = iota : tile<32xi32>
    %mask = cmpi less_than %iota, %b_lim, signed : tile<32xi32> -> tile<32xi1>
    %tv = make_tensor_view %p, shape = [%n], strides = [1] : tile<i32> -> tensor_view<?xf32, strides=[1]>
    %pv = make_partition_view %tv : partition_view<tile=(32), padding_value = neg_inf, tensor_view<?xf32, strides=[1]>>
    %tile, %tok0 = load_view_tko weak %pv[%c0] : partition_view<tile=(32), padding_value = neg_inf, tensor_view<?xf32, strides=[1]>>, tile<i32> -> tile<32xf32>, !cuda_tile.token
    %sel = select %mask, %tile, %pad : tile<32xi1>, tile<32xf32>
    %tok1 = store_view_tko weak %sel, %pv[%c0] : tile<32xf32>, partition_view<tile=(32), padding_value = neg_inf, tensor_view<?xf32, strides=[1]>>, tile<i32> -> !cuda_tile.token
    return
  }

  entry @no_fuse_padding_mismatch(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %n: !cuda_tile.tile<i32>) {
    %c0 = constant <i32: 0> : tile<i32>
    %lim = constant <i32: 64> : tile<i32>
    %zero = constant <f32: 0.0> : tile<f32>
    %r_pad = reshape %zero : tile<f32> -> tile<1xf32>
    %pad = broadcast %r_pad : tile<1xf32> -> tile<32xf32>
    %r_lim = reshape %lim : tile<i32> -> tile<1xi32>
    %b_lim = broadcast %r_lim : tile<1xi32> -> tile<32xi32>
    %iota = iota : tile<32xi32>
    %mask = cmpi less_than %iota, %b_lim, signed : tile<32xi32> -> tile<32xi1>
    %tv = make_tensor_view %p, shape = [%n], strides = [1] : tile<i32> -> tensor_view<?xf32, strides=[1]>
    %pv = make_partition_view %tv : partition_view<tile=(32), padding_value = neg_inf, tensor_view<?xf32, strides=[1]>>
    %tile, %tok0 = load_view_tko weak %pv[%c0] : partition_view<tile=(32), padding_value = neg_inf, tensor_view<?xf32, strides=[1]>>, tile<i32> -> tile<32xf32>, !cuda_tile.token
    %sel = select %mask, %tile, %pad : tile<32xi1>, tile<32xf32>
    %tok1 = store_view_tko weak %sel, %pv[%c0] : tile<32xf32>, partition_view<tile=(32), padding_value = neg_inf, tensor_view<?xf32, strides=[1]>>, tile<i32> -> !cuda_tile.token
    return
  }

  entry @no_fuse_multi_use(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %n: !cuda_tile.tile<i32>) {
    %c0 = constant <i32: 0> : tile<i32>
    %lim = constant <i32: 64> : tile<i32>
    %neg_inf = constant <f32: 0xFF800000> : tile<f32>
    %r_pad = reshape %neg_inf : tile<f32> -> tile<1xf32>
    %pad = broadcast %r_pad : tile<1xf32> -> tile<32xf32>
    %r_lim = reshape %lim : tile<i32> -> tile<1xi32>
    %b_lim = broadcast %r_lim : tile<1xi32> -> tile<32xi32>
    %iota = iota : tile<32xi32>
    %mask = cmpi less_than %iota, %b_lim, signed : tile<32xi32> -> tile<32xi1>
    %tv = make_tensor_view %p, shape = [%n], strides = [1] : tile<i32> -> tensor_view<?xf32, strides=[1]>
    %pv = make_partition_view %tv : partition_view<tile=(32), padding_value = neg_inf, tensor_view<?xf32, strides=[1]>>
    %tile, %tok0 = load_view_tko weak %pv[%c0] : partition_view<tile=(32), padding_value = neg_inf, tensor_view<?xf32, strides=[1]>>, tile<i32> -> tile<32xf32>, !cuda_tile.token
    %sel = select %mask, %tile, %pad : tile<32xi1>, tile<32xf32>
    %sum = addf %sel, %tile : tile<32xf32>
    %tok1 = store_view_tko weak %sum, %pv[%c0] : tile<32xf32>, partition_view<tile=(32), padding_value = neg_inf, tensor_view<?xf32, strides=[1]>>, tile<i32> -> !cuda_tile.token
    return
  }
}
