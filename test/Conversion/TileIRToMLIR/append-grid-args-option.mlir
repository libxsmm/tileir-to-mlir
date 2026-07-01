// RUN: tileir-to-mlir --convert-tileir-to-mlir='target=gpu append-grid-args=true' %s | FileCheck %s --check-prefix=GPU-APPEND

// Verifies append-grid-args=true behavior on GPU target:
// trailing launch args are appended and dim queries read those args instead of
// lowering to gpu.block_id/grid_dim.

// GPU-APPEND:       module attributes {gpu.container_module} {
// GPU-APPEND-NEXT:    gpu.module @m {
// GPU-APPEND:           gpu.func @add(%[[A:.*]]: i32, %[[B:.*]]: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32) kernel {
// GPU-APPEND:             arith.addi %[[A]], %[[B]]
// GPU-APPEND:             gpu.return
// GPU-APPEND:           }
// GPU-APPEND:           gpu.func @mul(%{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32) kernel {
// GPU-APPEND:             arith.muli
// GPU-APPEND:             gpu.return
// GPU-APPEND:           }
// GPU-APPEND:           gpu.func @coords(%[[E0:.*]]: i32, %[[BX:.*]]: i32, %[[BY:.*]]: i32, %[[BZ:.*]]: i32, %[[GX:.*]]: i32, %[[GY:.*]]: i32, %[[GZ:.*]]: i32) kernel {
// GPU-APPEND-NOT:         gpu.block_id
// GPU-APPEND-NOT:         gpu.grid_dim
// GPU-APPEND:             %[[S0:.*]] = arith.addi %[[BX]], %[[GX]]
// GPU-APPEND:             arith.addi %[[BY]], %[[GZ]]
// GPU-APPEND:             arith.addi %[[S0]], %[[E0]]
// GPU-APPEND:             gpu.return
// GPU-APPEND:           }

cuda_tile.module @m {
  entry @add(%arg0: tile<i32>, %arg1: tile<i32>) {
    %c = addi %arg0, %arg1 : tile<i32>
    return
  }

  entry @mul() {
    %a = constant <i32: 3> : tile<i32>
    %b = constant <i32: 4> : tile<i32>
    %c = muli %a, %b : tile<i32>
    return
  }

  entry @rounding(%x: tile<f32>, %y: tile<f32>, %ix: tile<i32>, %iy: tile<i32>) {
    %sum = addf %x, %y : tile<f32>
    %quo = divf %x, %y rounding<zero> : tile<f32>
    %tr = ftof %x rounding<nearest_even> : tile<f32> -> tile<f16>
    %si = ftoi %x signed rounding<nearest_int_to_zero> : tile<f32> -> tile<i32>
    %sf = itof %ix signed rounding<nearest_even> : tile<i32> -> tile<f32>
    %iq = divi %ix, %iy signed : tile<i32>
    return
  }

  entry @coords(%arg0: tile<i32>) {
    %bx, %by, %bz = get_tile_block_id : tile<i32>
    %nx, %ny, %nz = get_num_tile_blocks : tile<i32>
    %s0 = addi %bx, %nx : tile<i32>
    %s1 = addi %by, %nz : tile<i32>
    %s2 = addi %s0, %arg0 : tile<i32>
    return
  }
}
