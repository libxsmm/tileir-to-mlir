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
}
