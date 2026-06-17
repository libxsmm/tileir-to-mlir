// RUN: cudatile-to-mlir --convert-cuda-tile-to-mlir %s | FileCheck %s --check-prefix=GPU
// RUN: cudatile-to-mlir --convert-cuda-tile-to-mlir='target=gpu' %s | FileCheck %s --check-prefix=GPU
// RUN: cudatile-to-mlir --convert-cuda-tile-to-mlir='target=cpu' %s | FileCheck %s --check-prefix=CPU

// Verifies the `target` option of convert-cuda-tile-to-mlir.
//
//  * target=gpu (the default) wraps the kernels in a gpu.module that is marked
//    as a GPU container module, lowers each `entry` to a `gpu.func` kernel and
//    terminates them with `gpu.return`.
//  * target=cpu dissolves the cuda_tile.module into the enclosing builtin
//    module, lowers each `entry` to a plain `func.func` terminated by
//    `func.return`, and emits no GPU container module / gpu.module / gpu.func.

// GPU:       module attributes {gpu.container_module} {
// GPU-NEXT:    gpu.module @m {
// GPU:           gpu.func @add(%{{.*}}: i32, %{{.*}}: i32) kernel {
// GPU:             arith.addi
// GPU:             gpu.return
// GPU:           }
// GPU:           gpu.func @mul() kernel {
// GPU:             arith.muli
// GPU:             gpu.return
// GPU:           }
// GPU:         }

// CPU-NOT:   gpu.container_module
// CPU:       module {
// CPU-NOT:     gpu.module
// CPU:         func.func @add(%{{.*}}: i32, %{{.*}}: i32) {
// CPU:           arith.addi
// CPU:           return
// CPU:         }
// CPU:         func.func @mul() {
// CPU:           arith.muli
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
}
