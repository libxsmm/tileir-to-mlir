# VectorToCudaTile

Conversion pass and tooling for lowering CudaTile IR to MLIR GPU/vector/scf/arith/memref dialects.

## Prerequisites

- CMake 3.20+
- A C++17 compiler
- Ninja (recommended)
- An LLVM/MLIR build with CMake package config files (`MLIRConfig.cmake`)
- A `cuda-tile` build with the above LLVM/MLIR build (see https://github.com/intel/cuda-tile)

This project requires `CUDA_TILE_DIR` at configure time.

## Configure

From the repository root:

```bash
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DCUDA_TILE_DIR=/path/to/cuda-tile
```

Notes:

- `CUDA_TILE_DIR` must point to the root of the `cuda-tile` project (the directory that contains `include/` and `build/`).
- If your setup needs it, you can also pass `-DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm`.
- `test/Conversion/CudaTileToGPU/examples_cudatile_appendix.mlir` includes a gemm example

## Build

Build everything:

```bash
cmake --build build
```

Build only the conversion tool:

```bash
cmake --build build --target cudatile-to-gpu
```

## Run Tests

Run the CudaTile-to-GPU regression suite:

```bash
cmake --build build --target check-cudatile-to-gpu
```

Or use the convenience alias:

```bash
cmake --build build --target test
```

## Run the Tool Manually

Example invocation:

```bash
build/tools/cudatile-to-gpu --convert-cuda-tile-to-gpu input.mlir
```

You can pipe output into `FileCheck` for ad-hoc validation:

```bash
build/tools/cudatile-to-gpu --convert-cuda-tile-to-gpu test/Conversion/CudaTileToGPU/cudatile-to-mlir.mlir | FileCheck test/Conversion/CudaTileToGPU/cudatile-to-mlir.mlir
```
