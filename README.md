# CudaTileToMLIR

A conversion pass and tooling to lower CudaTile IR to MLIR dialects, including GPU, Vector, SCF, Arith, Math, and MemRef.

* **`--convert-cuda-tile-to-mlir`**: Lowers CudaTile IR to a mix of standard dialects.
  * `target={gpu|cpu}` (default: `gpu`): The `gpu` target wraps the result in a GPU container module; `cpu` lowers without the container marker.
  * `append-grid-args={true|false}` (default: `false`): If `true`, appends six `i32` launch-coordinate arguments (block-id x/y/z, grid-dim x/y/z) to entry functions and sources dimension queries from them. Required on the `cpu` target when dimension queries are present.
* **`--cuda-tile-to-mlir-pipeline`**: A convenience pipeline that runs `--tileir-ptr-to-view` followed by `--convert-cuda-tile-to-mlir`. It forwards the `target` and `append-grid-args` options.
* **`--tileir-ptr-to-view`**: Recognizes Triton-style pointer arithmetic (iota, reshape, broadcast, offset) feeding `load_ptr_tko` or `store_ptr_tko` and lifts them into higher-level "view" operations for more efficient lowering.
* **`--convert-memref-args-to-ptr-args`**: Promotes unranked memref arguments (`memref<*xT>`) to bare `!llvm.ptr` when all uses are identical `reinterpret_cast` operations.
* **`--convert-memref-args-to-ranked-memref`**: Promotes unranked memref kernel arguments and their associated scalar shape/stride arguments into ranked memref arguments. Intended for use after the core conversion.
  * `remove-unused={true|false}` (default: `true`): Removes scalar arguments that become unused after conversion to ranked memrefs.

## Prerequisites

* **CMake 3.20+**
* **C++17 compiler**
* **Ninja** (recommended)
* **LLVM/MLIR build** with CMake package configuration (`MLIRConfig.cmake`).
  * *Note: LLVM/MLIR revision `13c00cbc2aa2ddc9aae2e72b02bc6cb2a482e0e7` is verified compatible.*
* **`cuda-tile` build** using the same LLVM/MLIR version.
  * *Note:* You may need to qualify `TokenType` as `cuda_tile::TokenType` in the `cuda-tile` source to avoid naming conflicts.

## Configuration

```bash
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DCUDA_TILE_DIR=/path/to/cuda-tile
```

### Configuration Notes

* `CUDA_TILE_DIR` should point to either the root of the project (containing `include/` and `build/`) or the installation directory.
* `MLIR_DIR` must match the build used for `cuda-tile`; do not mix different LLVM revisions.
* If necessary, you can also specify `-DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm`.

## Building and Testing

### Build

To build all targets:

```bash
cmake --build build
```

### Run Tests

To run the regression suite:

```bash
cmake --build build --target check-cudatile-to-mlir
```

Alternatively, use the convenience alias for all configured tests:

```bash
cmake --build build --target test
```

## Manual Usage

### Basic Conversion

```bash
build/tools/cudatile-to-mlir --convert-cuda-tile-to-mlir input.mlir
```

### Validation with FileCheck

```bash
build/tools/cudatile-to-mlir --convert-cuda-tile-to-mlir test/Conversion/CudaTileToMLIR/cudatile-to-mlir.mlir | FileCheck test/Conversion/CudaTileToMLIR/cudatile-to-mlir.mlir
```

### Full Pipeline with Post-Processing

Example of a full conversion followed by common optimization passes:

```bash
build/tools/cudatile-to-mlir --convert-cuda-tile-to-mlir test/Conversion/CudaTileToMLIR/cudatile-appendix.mlir | \
mlir-opt --loop-invariant-code-motion -canonicalize -cse
```
