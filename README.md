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

## Configure

There are two supported configurations.

### A. Using Triton-to-tile-IR

If you point at a `Triton-to-tile-IR` checkout and its pinned LLVM install,
this project reuses **one** LLVM and the `cuda-tile` that Triton self-builds
against it. No separate `MLIR_DIR` or `CUDA_TILE_DIR` is required:

```bash
cmake -S . -B build -G Ninja \
  -DTRITON_TO_TILEIR_DIR=/path/to/Triton-to-tile-IR-repository \
  -DTRITON_TO_TILEIR_LLVM_SYSPATH=/path/to/llvm-install
```

In this mode:

- `MLIR_DIR` (and `LLVM_DIR`) default to
  `${TRITON_TO_TILEIR_LLVM_SYSPATH}/lib/cmake/{mlir,llvm}` — the pinned LLVM
  install used by Triton-to-tile-IR.
- `cuda-tile` is reused from Triton's self-built install under
  `build/_deps/.../tileir_src/build/install` (libs + public headers) with the
  generated `.h.inc` from the sibling `.../build/include`. It is built on
  demand the first time you build a target that needs it.
- The pinned LLVM **install** does not ship `llvm-lit`. To run the lit-based
  test targets (`check-*`, `test`), point CMake at a `llvm-lit` from any LLVM
  build (lit is version-independent):
  `-DLLVM_EXTERNAL_LIT=/path/to/llvm-project/build/bin/llvm-lit`.

### B. Basic setup (no Triton-to-tile-IR)

Without a Triton checkout, supply both LLVM and `cuda-tile` explicitly:

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

## Run Tests

Run the CudaTile-to-GPU regression suite:

```bash
cmake --build build --target check-cudatile-to-gpu
```

Run the external Triton/tileir chained suite:

```bash
cmake --build build --target check-cudatile-external
```

This external suite is optional.

- By default, if a sibling `../Triton-to-tile-IR` checkout exists, CMake auto-detects it.
- You can explicitly set the source checkout with:

```bash
cmake -S . -B build -G Ninja \
  -DTRITON_TO_TILEIR_DIR=/path/to/Triton-to-tile-IR \
  -DTRITON_TO_TILEIR_LLVM_SYSPATH=/path/to/llvm-project/installed/triton
```

- When `TRITON_TO_TILEIR_DIR` is configured, running `check-cudatile-external`
  auto-builds `triton-cuda-tile-opt` first (target:
  `build-triton-cuda-tile-opt`).

> **One LLVM for the whole project.** With `TRITON_TO_TILEIR_DIR` /
> `TRITON_TO_TILEIR_LLVM_SYSPATH` set, this project and Triton-to-tile-IR share
> the single pinned LLVM install (see *Configure → A*). The pinned LLVM is the
> only one source-compatible with Triton's dialects, and this project's own
> sources build against it as well, so there is exactly one LLVM in play.
>
> `build-triton-cuda-tile-opt` lets Triton build its **own** pinned
> `cuda-tile`/tileir against `TRITON_TO_TILEIR_LLVM_SYSPATH`; this project then
> reuses that same `cuda-tile` install, so no separate `CUDA_TILE_DIR` is
> needed. If `TRITON_TO_TILEIR_DIR` is set but `TRITON_TO_TILEIR_LLVM_SYSPATH`
> is not, configuration fails fast with an explanatory error. If
> `TRITON_TO_TILEIR_DIR` is unset (and no sibling checkout exists), the
> external suite is silently disabled and only the local suite runs; in that
> case supply `-DMLIR_DIR` and `-DCUDA_TILE_DIR` explicitly (see
> *Configure → B*).
>
> The chained harness runs
> `triton-cuda-tile-opt <pinned passes> | cudatile-to-gpu ... | FileCheck`.
> Both tools are built against the same pinned LLVM and the hand-off between
> them is textual MLIR.

Use the convenience alias to run all configured test suites:

```bash
cmake --build build --target test
```

If Triton/tileir is configured, `test` includes both local and external suites.
If not configured, it runs only the local CudaTile-to-GPU suite.

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
build/tools/cudatile-to-gpu --convert-cuda-tile-to-gpu test/Conversion/CudaTileToGPU/cudatile-appendix.mlir | mlir-opt --loop-invariant-code-motion -canonicalize -cse
```
