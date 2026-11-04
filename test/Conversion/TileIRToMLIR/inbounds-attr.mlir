// RUN: tileir-to-mlir --convert-tileir-to-mlir='target=gpu append-grid-args=false' %s | FileCheck %s

// Verifies that the `inbounds` attribute of view loads and stores marks the
// declared dimensions of the transfer ops in bounds, while undeclared dynamic
// dimensions stay masked.

// CHECK-LABEL: gpu.func @inbounds_attr_for_transfer_ops
// CHECK: %[[A:.*]] = vector.transfer_read {{.*}} {in_bounds = [true, false]} : memref<?x?xf16, strided<[?, 1], offset: ?>>, vector<4x2xf16>
// CHECK: %[[B:.*]] = vector.transfer_read {{.*}} {in_bounds = [true, true]} : memref<?x?xf16, strided<[?, 1], offset: ?>>, vector<4x2xf16>
// CHECK: %[[C:.*]] = vector.transfer_read {{.*}} : memref<?x?xf16, strided<[?, 1], offset: ?>>, vector<4x2xf16>
// CHECK: vector.transfer_write %[[A]], {{.*}} {in_bounds = [false, true]} : vector<4x2xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>

cuda_tile.module @m {
  entry @inbounds_attr_for_transfer_ops(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %m: !cuda_tile.tile<i32>, %n: !cuda_tile.tile<i32>, %s: !cuda_tile.tile<i32>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %tv = make_tensor_view %p, shape = [%m, %n], strides = [%s, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %pv = make_partition_view %tv : partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>
    %a, %tok0 = load_view_tko weak %pv[%c0, %c1] inbounds = [true, false] : partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> tile<4x2xf16>, !cuda_tile.token
    %b, %tok1 = load_view_tko weak %pv[%c0, %c1] inbounds = [true, true] : partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> tile<4x2xf16>, !cuda_tile.token
    %c, %tok2 = load_view_tko weak %pv[%c0, %c1] : partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> tile<4x2xf16>, !cuda_tile.token
    %tok3 = store_view_tko weak %a, %pv[%c0, %c1] inbounds = [false, true] : tile<4x2xf16>, partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> !cuda_tile.token
    %tok4 = store_view_tko weak %b, %pv[%c0, %c1] : tile<4x2xf16>, partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> !cuda_tile.token
    %tok5 = store_view_tko weak %c, %pv[%c0, %c1] : tile<4x2xf16>, partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> !cuda_tile.token
    return
  }
}
