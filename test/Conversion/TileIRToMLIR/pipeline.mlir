// RUN: tileir-to-mlir --tileir-to-mlir-pipeline %s | FileCheck %s --check-prefix=GPU
// RUN: tileir-to-mlir --tileir-to-mlir-pipeline='target=cpu append-grid-args=true' %s | FileCheck %s --check-prefix=CPU
// RUN: not tileir-to-mlir --tileir-to-mlir-pipeline='no-such-option=1' %s 2>&1 | FileCheck %s --check-prefix=ERR

// The pipeline passes its options on to --convert-tileir-to-mlir.

// GPU: gpu.func @coords(%{{.*}}: memref<*xi32>) kernel
// GPU:   gpu.block_id x

// CPU: func.func @coords(%{{.*}}: memref<*xi32>, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32)
// CPU-NOT: gpu.

// ERR: no-such-option

cuda_tile.module @m {
  entry @coords(%p: tile<ptr<i32>>) {
    %x, %y, %z = get_tile_block_id : tile<i32>
    %t = store_ptr_tko weak %p, %x : tile<ptr<i32>>, tile<i32> -> !cuda_tile.token
    return
  }
}
