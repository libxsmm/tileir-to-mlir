//===- Passes.td - TileIRToMLIR pass definitions ---------*- tablegen -*-===//
//
// Pass definitions for the Tile IR to MLIR conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_TILEIRTOMLIR_PASSES
#define MLIR_CONVERSION_TILEIRTOMLIR_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTileIRToMLIRPass : Pass<"convert-tileir-to-mlir", "ModuleOp"> {
  let summary = "Convert Tile IR to GPU/vector/scf/arith ops";
  let description = [{
    Lowers Tile IR to a mix of GPU, vector, scf, arith, math, memref and ub
    operations. The `target` option selects the lowering flavor: `gpu` (the
    default) marks the result as a GPU container module, while `cpu` lowers
    without the GPU container-module marker. `append-grid-args` controls
    whether lowered entry functions receive six trailing `i32`
    launch-coordinate arguments (block-id x/y/z and grid-dim x/y/z), and
    whether dim-query ops read those arguments (`true`) or lower to gpu
    dim-query ops (`false`, GPU target only). `drop-rounding-modes` forces all
    lowering patterns for rounding-carrying ops to preserve source rounding
    only as discardable `tir-dropped-rounding` attributes.
    `assume-in-bounds` forces all generated vector.transfer_read and
    vector.transfer_write ops to carry fully in-bounds flags.
    `known-block-size`, when given three values, sets the `known_block_size`
    attribute (x, y, z) on the generated `gpu.func` ops; when omitted the
    attribute is not set.
  }];
  let dependentDialects = [
    "cuda_tile::CudaTileDialect",
    "arith::ArithDialect",
    "func::FuncDialect",
    "math::MathDialect",
    "vector::VectorDialect",
    "scf::SCFDialect",
    "memref::MemRefDialect",
    "gpu::GPUDialect",
    "ub::UBDialect",
  ];
  let options = [
    Option<"target", "target", "::mlir::TileIRTarget",
           /*default=*/"::mlir::TileIRTarget::GPU",
           "Lowering target ('gpu' or 'cpu')",
           [{::llvm::cl::values(
              clEnumValN(::mlir::TileIRTarget::GPU, "gpu",
                         "Lower to a GPU container module"),
              clEnumValN(::mlir::TileIRTarget::CPU, "cpu",
                         "Lower without the GPU container-module marker"))}]>,
    Option<"appendGridArgs", "append-grid-args", "bool",
           /*default=*/"false",
           "Append six launch-coordinate i32 args to lowered entry signatures and source dim queries from them (required on non-GPU targets when dim-query ops are used)">,
    Option<"dropRoundingModes", "drop-rounding-modes",
           "bool", /*default=*/"false",
           "Always drop source rounding-mode semantics and preserve them only as tir-dropped-rounding annotations">,
    Option<"assumeInBounds", "assume-in-bounds",
           "bool", /*default=*/"false",
           "Assume all load and store ops are in bounds.">,
    ListOption<"knownBlockSize", "known-block-size", "int32_t",
               "Block size (x, y, z) to set as the known_block_size attribute on generated gpu.func ops; must be empty or exactly three values">
  ]; 
}

def ConvertMemrefArgsToPtrArgsPass : Pass<"convert-memref-args-to-ptr-args", "ModuleOp"> {
  let summary = "Promote unranked-memref function arguments to !llvm.ptr";
  let description = [{
    Rewrites eligible unranked-memref arguments (`memref<*xT>`) to bare
    `!llvm.ptr` arguments. An argument is eligible when it has no uses, or when
    every use is a `memref.cast` or `memref.reinterpret_cast` whose source is
    that exact argument and whose result is a ranked memref with the same
    element type. Matching casts need not have the same rank or layout. Each is
    replaced independently by a memref descriptor materialized from the pointer
    and wrapped in an `unrealized_conversion_cast` to its original ranked type.

    The rewrite is conservative. An argument is promoted only when:
      * it is an unranked-memref argument,
      * every use satisfies the cast constraint above, and
      * the containing function is a symbol with no references in its nearest
        symbol table, so no in-module call site is invalidated.
    When any of these conditions fails the argument is left untouched.
  }];
  let dependentDialects = [
    "memref::MemRefDialect",
    "func::FuncDialect",
    "arith::ArithDialect",
    "LLVM::LLVMDialect",
  ];
}

def ConvertMemrefArgsToRankedMemrefPass : Pass<"convert-memref-args-to-ranked-memref", "ModuleOp"> {
  let summary = "Promote unranked memref function args to ranked memrefs";
  let description = [{
    Rewrites function-like signatures produced by `--convert-tileir-to-mlir`
    from unranked memref pointer arguments plus scalar shape/stride arguments
    into ranked memref arguments.

    For each unranked memref argument (`memref<*xT>`), the pass matches the
    canonical pattern where every use is an identical, zero-offset
    `memref.reinterpret_cast` to the same ranked result type. The argument type
    is updated to that ranked memref type and the redundant casts are erased.

    Scalar arguments that were originally used to materialize the matched
    cast's dynamic sizes/strides are rewritten to equivalent memref queries
    when they have an unambiguous replacement recipe:
      * dynamic shape -> `memref.dim`
      * dynamic stride -> `memref.extract_strided_metadata`

    The `remove-unused` option controls removal of unused signature arguments:
      * `memref-dependent` (the default) removes only arguments explicitly
        used for dynamic sizes or strides of converted memrefs,
      * `assumed-memref-dependent` additionally removes unused arguments in a
        complete `(memref, size0, ..., sizeN-1, stride0, ..., strideN-1)`
        calling-convention sequence after each converted rank-N memref,
      * `other` removes only unused arguments that are not explicitly
        memref-dependent,
      * `all` removes every unused argument, and
      * `none` preserves all unused arguments.

    The rewrite is conservative and skips a function unless it is a symbol with
    no references in its nearest symbol table, so changing its signature cannot
    invalidate in-module symbol users.
  }];
  let options = [
    Option<"removeUnused", "remove-unused", "::mlir::MemrefArgRemovalMode",
           /*default=*/"::mlir::MemrefArgRemovalMode::MemrefDependent",
          "Unused argument removal mode ('all', 'none', 'memref-dependent', 'assumed-memref-dependent', or 'other')",
           [{::llvm::cl::values(
              clEnumValN(::mlir::MemrefArgRemovalMode::All, "all",
                         "Remove all unused arguments"),
              clEnumValN(::mlir::MemrefArgRemovalMode::None, "none",
                         "Preserve all unused arguments"),
              clEnumValN(::mlir::MemrefArgRemovalMode::MemrefDependent,
                         "memref-dependent",
                         "Remove only unused memref-dependent arguments"),
              clEnumValN(::mlir::MemrefArgRemovalMode::AssumedMemrefDependent,
                         "assumed-memref-dependent",
                         "Also remove unused assumed size and stride arguments"),
              clEnumValN(::mlir::MemrefArgRemovalMode::Other, "other",
                         "Remove only unused non-memref-dependent arguments"))}]>,
  ];
  let dependentDialects = [
    "memref::MemRefDialect",
    "func::FuncDialect",
    "arith::ArithDialect",
    "gpu::GPUDialect",
  ];
}

def TileIRPtrToViewPass : Pass<"tileir-ptr-to-view", "ModuleOp"> {
  let summary = "Rewrite Triton-style cuda_tile ptr-arithmetic into "
                "make_tensor_view + make_partition_view + load/store_view_tko";
  let description = [{
    Preprocessing pass that attempts to recognize canonical Triton-emitted
    iota+reshape+broadcast+offset pointer-arithmetic chains feeding a
    `cuda_tile.load_ptr_tko` / `cuda_tile.store_ptr_tko` and rewrites them into
    the higher-level `cuda_tile.make_tensor_view` + `cuda_tile.make_partition_view`
    + `cuda_tile.load_view_tko` / `cuda_tile.store_view_tko` form, which the
    `convert-tileir-to-mlir` pass can lower.
  }];
  let dependentDialects = [
    "cuda_tile::CudaTileDialect",
  ];
}

#endif // MLIR_CONVERSION_TILEIRTOMLIR_PASSES
