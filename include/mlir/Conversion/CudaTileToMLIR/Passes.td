//===- Passes.td - CudaTileToMLIR pass definitions ---------*- tablegen -*-===//
//
// Pass definitions for the CudaTile IR to MLIR conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CUDATILETOMLIR_PASSES
#define MLIR_CONVERSION_CUDATILETOMLIR_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTileIRToMLIRPass : Pass<"convert-cuda-tile-to-mlir", "ModuleOp"> {
  let summary = "Convert CudaTile IR to GPU/vector/scf/arith ops";
  let description = [{
    Lowers CudaTile IR to a mix of GPU, vector, scf, arith, math, memref and ub
    operations. The `target` option selects the lowering flavor: `gpu` (the
    default) marks the result as a GPU container module, while `cpu` lowers
    without the GPU container-module marker. `append-grid-args` controls
    whether lowered entry functions receive six trailing `i32`
    launch-coordinate arguments (block-id x/y/z and grid-dim x/y/z), and
    whether dim-query ops read those arguments (`true`) or lower to gpu
    dim-query ops (`false`, GPU target only).
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
    Option<"target", "target", "::mlir::CudaTileTarget",
           /*default=*/"::mlir::CudaTileTarget::GPU",
           "Lowering target ('gpu' or 'cpu')",
           [{::llvm::cl::values(
              clEnumValN(::mlir::CudaTileTarget::GPU, "gpu",
                         "Lower to a GPU container module"),
              clEnumValN(::mlir::CudaTileTarget::CPU, "cpu",
                         "Lower without the GPU container-module marker"))}]>,
    Option<"appendGridArgs", "append-grid-args", "bool",
          /*default=*/"false",
          "Append six launch-coordinate i32 args to lowered entry signatures and source dim queries from them (required on non-GPU targets when dim-query ops are used)">,
  ];
}

def ConvertMemrefArgsToPtrArgsPass : Pass<"convert-memref-args-to-ptr-args", "ModuleOp"> {
  let summary = "Promote unranked-memref function arguments to !llvm.ptr";
  let description = [{
    Detects function arguments of unranked-memref type (`memref<*xT>`) whose
    every use is an identical `memref.reinterpret_cast` that pins down a concrete
    ranked layout, and rewrites the function signature to take a bare `!llvm.ptr`
    instead. The ranked `memref` is recovered once at the top of the body with an
    `unrealized_conversion_cast` from that pointer, and the now-redundant
    `reinterpret_cast` ops (and any address-computation ops that solely fed them)
    are erased.

    The rewrite is conservative. An argument is promoted only when:
      * it is of unranked-memref type and has at least one use,
      * every use is a `memref.reinterpret_cast` whose source is that argument,
      * all those casts are mutually identical (same ranked result type and the
        same offset / size / stride operands), so the promoted type is
        unambiguous, and
      * the function is not referenced (called / launched) from within the
        module, so no in-module call site is invalidated.
    When any of these conditions fails the argument is left untouched.
  }];
  let dependentDialects = [
    "memref::MemRefDialect",
    "func::FuncDialect",
    "arith::ArithDialect",
    "LLVM::LLVMDialect",
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
    `convert-cuda-tile-to-mlir` pass can lower.
  }];
  let dependentDialects = [
    "cuda_tile::CudaTileDialect",
  ];
}

#endif // MLIR_CONVERSION_CUDATILETOMLIR_PASSES
