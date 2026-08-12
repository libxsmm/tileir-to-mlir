# TileIRToMLIR
A conversion pass and tooling to lower Tile IR to MLIR dialects, including GPU, Vector, SCF, Arith, Math, and MemRef.

* **`--tileir-to-mlir-pipeline`**: A convenience pipeline that runs `--tileir-ptr-to-view` followed by `--convert-tileir-to-mlir`. It forwards all options of `--convert-tileir-to-mlir` (`target`, `append-grid-args`, `drop-rounding-modes`, `assume-in-bounds`, `known-block-size`).
* **`--convert-tileir-to-mlir`**: Lowers Tile IR to a mix of standard dialects.
  * `target={gpu|cpu}` (default: `gpu`): The `gpu` target wraps the result in a GPU container module; `cpu` lowers without the container marker.
  * `append-grid-args={true|false}` (default: `false`): If `true`, appends six `i32` launch-coordinate arguments (block-id x/y/z, grid-dim x/y/z) to entry functions and sources dimension queries from them. Required on the `cpu` target when dimension queries are present.
  * `drop-rounding-modes={true|false}`(default: `false`): If true, always drop source rounding-mode semantics and preserve them only as tir-dropped-rounding annotations.
  * `assume-in-bounds={true|false}`(default: `false`): If true, assume all load and store ops are in bounds.
  * `known-block-size=x,y,z` (default: unset): If given exactly three values, sets the `known_block_size` attribute on the generated `gpu.func` ops; if omitted, the attribute is not set.
* **`--tileir-ptr-to-view`**: Recognizes gather/scatter-style pointer arithmetic (iota, reshape, broadcast, offset) feeding `load_ptr_tko` or `store_ptr_tko` and lifts them into higher-level "view" operations for more efficient lowering.
* **`--convert-memref-args-to-ptr-args`**: Promotes unranked memref arguments (`memref<*xT>`) to bare `!llvm.ptr` when they have no uses or every use is a direct `memref.cast` or `memref.reinterpret_cast`. Each cast may recover a distinct ranked layout. The function must have no in-module symbol uses. Intended for use after the core conversion.
* **`--convert-memref-args-to-ranked-memref`**: Promotes an unranked memref argument (`memref<*xT>`) to a ranked memref when every use is the same zero-offset `memref.reinterpret_cast`. It rewrites unambiguous dynamic size and stride scalar uses to memref queries. The function must have no in-module symbol uses. Intended for use after the core conversion.
  * `remove-unused={memref-dependent|assumed-memref-dependent|other|all|none}` (default: `memref-dependent`): Controls which unused arguments are removed after promotion.
    * `memref-dependent`: Removes only arguments explicitly used for dynamic sizes or strides of converted memrefs.
    * `assumed-memref-dependent`: Also removes unused arguments in each complete `(memref, size0, ..., sizeN-1, stride0, ..., strideN-1)` sequence after a converted rank-N memref.
    * `other`: Removes unused arguments that are not explicitly memref-dependent.
    * `all`: Removes every unused argument.
    * `none`: Preserves all unused arguments.
  


## Prerequisites

* **CMake 3.20+**
* **C++17 compiler**
* **Ninja** (recommended)
* **LLVM/MLIR build** with CMake package configuration (`MLIRConfig.cmake`).
  * *Note: LLVM/MLIR revision `13c00cbc2aa2ddc9aae2e72b02bc6cb2a482e0e7` is verified compatible.*
* **`cuda-tile` build** using the same LLVM/MLIR version (https://github.com/nvidia/cuda-tile).
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
cmake --build build --target check-tileir-to-mlir
```

Alternatively, use the convenience alias for all configured tests:

```bash
cmake --build build --target test
```

## Manual Usage

### Basic Conversion

```bash
build/tools/tileir-to-mlir --convert-tileir-to-mlir input.mlir
```

### Validation with FileCheck

```bash
build/tools/tileir-to-mlir --convert-tileir-to-mlir test/Conversion/TileIRToMLIR/tileir-to-mlir.mlir | FileCheck test/Conversion/TileIRToMLIR/tileir-to-mlir.mlir
```

### Full Pipeline with Post-Processing

Example of a full conversion followed by common optimization passes:

```bash
build/tools/tileir-to-mlir --convert-tileir-to-mlir test/Conversion/TileIRToMLIR/cudatile-appendix.mlir | \
mlir-opt --loop-invariant-code-motion -canonicalize -cse
```
