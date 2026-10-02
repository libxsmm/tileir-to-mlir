// RUN: tileir-to-mlir --convert-tileir-to-mlir='target=gpu' %s | FileCheck %s --check-prefix=GPU
// RUN: tileir-to-mlir --convert-tileir-to-mlir='target=cpu' %s | FileCheck %s --check-prefix=CPU

// The host (CPU) target has no tf32 type, so the type converter lowers tf32
// tile element types to f32.  The GPU target keeps tf32 unchanged.  The tf32
// values feed an mmaf, whose operands make the converted element type
// observable.  On GPU the `ftof f32 -> tf32` lowers to arith.truncf; on CPU,
// where tf32 lowers to f32, the source and result coincide so the cast becomes
// a no-op (no ftof/truncf is emitted) and the contraction runs on f32.

// GPU-LABEL: gpu.func @tf32_mma
// GPU:         arith.truncf %{{.*}} : vector<8x8xf32> to vector<8x8xtf32>
// GPU:         vector.contract {{.*}} vector<8x8xtf32>, vector<8x8xtf32> into vector<8x8xf32>

// CPU-LABEL: func.func @tf32_mma
// CPU-NOT:     tf32
// CPU-NOT:     arith.truncf
// CPU:         vector.contract {{.*}} vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>

// The widening applies to every use of tf32 as an element type: constants,
// globals, pointees and tensor views.

// GPU-LABEL: memref.global @g_tf32 : memref<2xtf32>
// GPU-LABEL: gpu.func @tf32_memory
// GPU:         vector.transfer_read {{.*}} : memref<4xtf32, strided<[1], offset: ?>>, vector<4xtf32>

// CPU-LABEL: memref.global @g_tf32 : memref<2xf32> = dense<[1.000000e+00, 2.000000e+00]>
// CPU-LABEL: func.func @tf32_memory
// CPU-NOT:     tf32
// CPU:         memref.get_global @g_tf32 : memref<2xf32>
// CPU:         arith.constant 1.500000e+00 : f32
// CPU:         arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf32>
// CPU:         memref.store %{{.*}} : memref<f32, strided<[], offset: ?>>
// CPU:         vector.transfer_read {{.*}} : memref<4xf32, strided<[1], offset: ?>>, vector<4xf32>
// CPU:         vector.transfer_write {{.*}} : vector<4xf32>, memref<4xf32, strided<[1], offset: ?>>

cuda_tile.module @m {
  entry @tf32_mma() {
    %a32 = constant <f32: 1.000000e+00> : tile<8x8xf32>
    %b32 = constant <f32: 1.000000e+00> : tile<8x8xf32>
    %a = ftof %a32 : tile<8x8xf32> -> tile<8x8xtf32>
    %b = ftof %b32 : tile<8x8xf32> -> tile<8x8xtf32>
    %c = constant <f32: 0.000000e+00> : tile<8x8xf32>
    %d = mmaf %a, %b, %c : tile<8x8xtf32>, tile<8x8xtf32>, tile<8x8xf32>
    return
  }

  global @g_tf32 <tf32: [1.0, 2.0]> : tile<2xtf32>

  entry @tf32_memory(%p: !cuda_tile.tile<!cuda_tile.ptr<tf32>>,
                     %tv: !cuda_tile.tensor_view<4xtf32, strides=[1]>) {
    %g = get_global @g_tf32 : tile<ptr<tf32>>
    %c0 = constant <i32: 0> : tile<i32>
    %s = constant <tf32: 1.5> : tile<tf32>
    %v = constant <tf32: [1.0, 2.0, 3.0, 4.0]> : tile<4xtf32>
    %t0 = store_ptr_tko weak %p, %s : tile<ptr<tf32>>, tile<tf32> -> token
    %pv = make_partition_view %tv : partition_view<tile=(4), tensor_view<4xtf32, strides=[1]>>
    %x, %t1 = load_view_tko weak %pv[%c0] : partition_view<tile=(4), tensor_view<4xtf32, strides=[1]>>, tile<i32> -> tile<4xtf32>, token
    %t2 = store_view_tko weak %v, %pv[%c0] : tile<4xtf32>, partition_view<tile=(4), tensor_view<4xtf32, strides=[1]>>, tile<i32> -> token
    return
  }
}
