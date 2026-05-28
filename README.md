# CudaTileToGPU

Conversion pass and tooling for lowering CudaTile IR to MLIR GPU/vector/scf/arith/math/memref dialects.

## Prerequisites

- CMake 3.20+
- A C++17 compiler
- Ninja (recommended)
- An LLVM/MLIR build with CMake package config files (`MLIRConfig.cmake`)
- A `cuda-tile` build produced with that same LLVM/MLIR version

The LLVM/MLIR version must match the one used by your `cuda-tile`.
See `cuda-tile`'s `README.md` and its `cmake/IncludeLLVM.cmake` to find the
LLVM version it expects.

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
- `MLIR_DIR` must point to the same LLVM/MLIR build that was used to build `cuda-tile`; mixing different LLVM revisions is not supported.
- If your setup needs it, you can also pass `-DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm`.
- `test/Conversion/CudaTileToGPU/cudatile-appendix.mlir` includes a gemm example.

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

Example: convert the appendix example and run canonical cleanup passes:

```bash
build/tools/cudatile-to-gpu --convert-cuda-tile-to-gpu test/Conversion/CudaTileToGPU/cudatile-appendix.mlir | mlir-opt --loop-invariant-code-motion -cse -canonicalize -cse
```
