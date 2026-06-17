# CudaTileToMLIR

Conversion pass and tooling for lowering CudaTile IR to MLIR GPU/vector/scf/arith/math/memref dialects.

## Prerequisites

- CMake 3.20+
- A C++17 compiler
- Ninja (recommended)
- An LLVM/MLIR build with CMake package config files (`MLIRConfig.cmake`)
  - The LLVM/MLIR version a80153ea4f7dfcd6e0dcf2b415f9ace3cd54015a has been verified to be compatible.
- A `cuda-tile` build produced with that same LLVM/MLIR version
  - You might need to apply a minor patch to `cuda-tile` source to compile it with this version:
    Unqualified uses of the type `TokenType` creates a conflict. Qualify with `cuda_tile::TokenType`.

## Configure

```bash
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DCUDA_TILE_DIR=/path/to/cuda-tile
```

Notes:

- `CUDA_TILE_DIR` must point to the root of the `cuda-tile` project (the directory that contains `include/` and `build/`), or to a cuda-tile install directory (contains `lib/` and `include/include/`).
- `MLIR_DIR` must point to the same LLVM/MLIR build that was used to build `cuda-tile`; mixing different LLVM revisions is not supported.
- If your setup needs it, you can also pass `-DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm`.
- `test/Conversion/CudaTileToMLIR/cudatile-appendix.mlir` includes a gemm example.

## Build

Build everything:

```bash
cmake --build build
```

## Run Tests

Run the CudaTile-to-MLIR regression suite:

```bash
cmake --build build --target check-cudatile-to-mlir
```

Use the convenience alias to run all configured test suites:

```bash
cmake --build build --target test
```

## Run the Tool Manually

Example invocation:

```bash
build/tools/cudatile-to-mlir --convert-cuda-tile-to-mlir input.mlir
```

You can pipe output into `FileCheck` for ad-hoc validation:

```bash
build/tools/cudatile-to-mlir --convert-cuda-tile-to-mlir test/Conversion/CudaTileToMLIR/cudatile-to-mlir.mlir | FileCheck test/Conversion/CudaTileToMLIR/cudatile-to-mlir.mlir
```

Example: convert the appendix example and run canonical cleanup passes:

```bash
build/tools/cudatile-to-mlir --convert-cuda-tile-to-mlir test/Conversion/CudaTileToMLIR/cudatile-appendix.mlir | mlir-opt --loop-invariant-code-motion -canonicalize -cse
```
