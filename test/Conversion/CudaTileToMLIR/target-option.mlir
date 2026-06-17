// RUN: cudatile-to-mlir --convert-cuda-tile-to-mlir %s | FileCheck %s --check-prefix=GPU
// RUN: cudatile-to-mlir --convert-cuda-tile-to-mlir='target=gpu' %s | FileCheck %s --check-prefix=GPU
// RUN: cudatile-to-mlir --convert-cuda-tile-to-mlir='target=cpu' %s | FileCheck %s --check-prefix=CPU

// Verifies the `target` option of convert-cuda-tile-to-mlir: the default and
// `target=gpu` mark the result as a GPU container module, while `target=cpu`
// omits the GPU container-module marker. The conversion itself happens in both
// cases.

// GPU: module attributes {gpu.container_module}
// GPU: gpu.module @m
// GPU: gpu.func @k

// CPU-NOT: gpu.container_module
// CPU: gpu.module @m
// CPU: gpu.func @k

cuda_tile.module @m {
  entry @k() {
    return
  }
}
