// RUN: tileir-to-mlir --convert-tileir-to-mlir='target=gpu append-grid-args=false assume-in-bounds=true' %s | FileCheck %s

// Verifies assume-in-bounds=true forces transfer ops to carry fully in-bounds
// flags, even when static divisibility cannot be proven.

// CHECK-LABEL: gpu.func @assume_in_bounds_for_transfer_ops
// CHECK: %[[RD:.*]] = vector.transfer_read {{.*}} {in_bounds = [true, true]} : memref<?x?xf16, strided<[?, 1], offset: ?>>, vector<4x2xf16>
// CHECK: vector.transfer_write %[[RD]], {{.*}} {in_bounds = [true, true]} : vector<4x2xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>

cuda_tile.module @m {
  entry @assume_in_bounds_for_transfer_ops(%p: !cuda_tile.tile<!cuda_tile.ptr<f16>>, %m: !cuda_tile.tile<i32>, %n: !cuda_tile.tile<i32>, %s: !cuda_tile.tile<i32>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %tv = make_tensor_view %p, shape = [%m, %n], strides = [%s, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
    %pv = make_partition_view %tv : partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>
    %tile, %tok0 = load_view_tko weak %pv[%c0, %c1] : partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> tile<4x2xf16>, !cuda_tile.token
    %tok1 = store_view_tko weak %tile, %pv[%c0, %c1] : tile<4x2xf16>, partition_view<tile=(4x2), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> !cuda_tile.token
    return
  }
}
