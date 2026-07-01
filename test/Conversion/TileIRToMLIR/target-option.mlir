// RUN: tileir-to-mlir --convert-tileir-to-mlir='append-grid-args=false' %s | FileCheck %s --check-prefix=GPU
// RUN: tileir-to-mlir --convert-tileir-to-mlir='target=gpu append-grid-args=false' %s | FileCheck %s --check-prefix=GPU
// RUN: tileir-to-mlir --convert-tileir-to-mlir='target=cpu append-grid-args=true' %s | FileCheck %s --check-prefix=CPU

// Verifies the `target` option of convert-tileir-to-mlir.
// append-grid-args is set explicitly per RUN line so this test isolates target
// behavior.
//
//  * target=gpu (the default) wraps the kernels in a gpu.module that is marked
//    as a GPU container module, lowers each `entry` to a `gpu.func` kernel and
//    terminates them with `gpu.return`.
//  * target=cpu dissolves the cuda_tile.module into the enclosing builtin
//    module, lowers each `entry` to a plain `func.func` terminated by
//    `func.return`, and emits no GPU container module / gpu.module / gpu.func.
//    Because the CPU target cannot use gpu dimension-query ops, every function
//    gains six trailing `i32` arguments carrying the launch coordinates (tile
//    block id x/y/z followed by grid dim x/y/z) ahead of the converted entry
//    arguments. `get_tile_block_id` / `get_num_tile_blocks` then read those
//    arguments instead of lowering to `gpu.block_id` / `gpu.grid_dim`.

// GPU:       module attributes {gpu.container_module} {
// GPU-NEXT:    gpu.module @m {
// GPU:           gpu.func @add(%[[A:.*]]: i32, %[[B:.*]]: i32) kernel {
// GPU:             arith.addi %[[A]], %[[B]]
// GPU:             gpu.return
// GPU:           }
// GPU:           gpu.func @mul() kernel {
// GPU:             arith.muli
// GPU:             gpu.return
// GPU:           }
//   On GPU the dimension queries still lower to gpu.block_id / gpu.grid_dim.
// GPU:           gpu.func @coords(%[[E0:.*]]: i32) kernel {
// GPU:             gpu.block_id x
// GPU:             gpu.grid_dim x
// GPU:             gpu.return
// GPU:           }
//   Rounding is kept natively when representable and preserved as
//   tir-dropped-rounding otherwise (same policy on both targets).
// GPU:           gpu.func @rounding(%[[FX:.*]]: f32, %[[FY:.*]]: f32, %[[IX:.*]]: i32, %[[IY:.*]]: i32) kernel {
// GPU:             arith.addf %[[FX]], %[[FY]]
// GPU:             arith.divf %[[FX]], %[[FY]] {"tir-dropped-rounding" = "zero"}
// GPU:             arith.truncf %[[FX]] to_nearest_even
// GPU:             arith.fptosi %[[FX]]
// GPU:             arith.sitofp %[[IX]]
// GPU:             arith.divsi %[[IX]], %[[IY]]
// GPU:             gpu.return
// GPU:           }
// GPU:         }

// CPU-NOT:   gpu.container_module
// CPU:       module {
// CPU-NOT:     gpu.module
//   The six trailing i32 args are the launch coordinates; the leading two are
//   the converted entry arguments that the body actually uses.
// CPU:         func.func @add(%[[A:.*]]: i32, %[[B:.*]]: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32) {
// CPU:           arith.addi %[[A]], %[[B]]
// CPU:           return
// CPU:         }
//   An argument-less entry still receives the six launch-coordinate args.
// CPU:         func.func @mul(%{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32) {
// CPU:           arith.muli
// CPU:           return
// CPU:         }
//   Dimension queries read the launch-coordinate args by position rather than
//   emitting gpu.block_id / gpu.grid_dim. Argument order is
//   [<entry args...>, block x, y, z, grid x, y, z], so block-id x is %[[BX]],
//   grid-dim x is %[[GX]], grid-dim z is %[[GZ]] and the entry arg is %[[E0]].
// CPU:         func.func @coords(%[[E0:.*]]: i32, %[[BX:.*]]: i32, %[[BY:.*]]: i32, %[[BZ:.*]]: i32, %[[GX:.*]]: i32, %[[GY:.*]]: i32, %[[GZ:.*]]: i32) {
// CPU-NOT:       gpu.block_id
// CPU-NOT:       gpu.grid_dim
// CPU:           %[[S0:.*]] = arith.addi %[[BX]], %[[GX]]
// CPU:           arith.addi %[[BY]], %[[GZ]]
// CPU:           arith.addi %[[S0]], %[[E0]]
// CPU:           return
// CPU:         }
//   CPU follows the same representable-vs-dropped rounding policy.
// CPU:         func.func @rounding(%[[FX:.*]]: f32, %[[FY:.*]]: f32, %[[IX:.*]]: i32, %[[IY:.*]]: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32) {
// CPU:           arith.addf %[[FX]], %[[FY]]
// CPU:           arith.divf %[[FX]], %[[FY]] {"tir-dropped-rounding" = "zero"}
// CPU:           arith.truncf %[[FX]] to_nearest_even
// CPU:           arith.fptosi %[[FX]]
// CPU:           arith.sitofp %[[IX]]
// CPU:           arith.divsi %[[IX]], %[[IY]]
// CPU:           return
// CPU:         }
// CPU-NOT:     gpu.func

cuda_tile.module @m {
  // An entry with arguments: exercises signature conversion in both targets
  // (tile<i32> -> i32).
  entry @add(%arg0: tile<i32>, %arg1: tile<i32>) {
    %c = addi %arg0, %arg1 : tile<i32>
    return
  }

  // A second, argument-less entry to verify multiple entries are handled and
  // ordering is preserved.
  entry @mul() {
    %a = constant <i32: 3> : tile<i32>
    %b = constant <i32: 4> : tile<i32>
    %c = muli %a, %b : tile<i32>
    return
  }

  // An entry exercising the launch-coordinate queries. On CPU these must read
  // the trailing function arguments (block-id x = arg0, grid-dim x = arg3,
  // grid-dim z = arg5, entry arg = arg6); on GPU they lower to gpu.block_id /
  // gpu.grid_dim. Selecting distinct dims makes the arg mapping observable.
  entry @coords(%arg0: tile<i32>) {
    %bx, %by, %bz = get_tile_block_id : tile<i32>
    %nx, %ny, %nz = get_num_tile_blocks : tile<i32>
    %s0 = addi %bx, %nx : tile<i32>
    %s1 = addi %by, %nz : tile<i32>
    %s2 = addi %s0, %arg0 : tile<i32>
    return
  }

  // Rounding-mode handling differs by target.
  entry @rounding(%x: tile<f32>, %y: tile<f32>, %ix: tile<i32>, %iy: tile<i32>) {
    %sum = addf %x, %y : tile<f32>
    %quo = divf %x, %y rounding<zero> : tile<f32>
    %tr = ftof %x rounding<nearest_even> : tile<f32> -> tile<f16>
    %si = ftoi %x signed rounding<nearest_int_to_zero> : tile<f32> -> tile<i32>
    %sf = itof %ix signed rounding<nearest_even> : tile<i32> -> tile<f32>
    %iq = divi %ix, %iy signed : tile<i32>
    return
  }
}
