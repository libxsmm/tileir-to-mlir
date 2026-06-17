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
    without the GPU container-module marker.
  }];
  let dependentDialects = [
    "cuda_tile::CudaTileDialect",
    "arith::ArithDialect",
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
  ];
}

def TileIRPtrToViewPass : Pass<"tileir-ptr-to-view", "ModuleOp"> {
  let summary = "Rewrite Triton-style cuda_tile ptr-arithmetic into "
                "make_tensor_view + make_partition_view + load/store_view_tko";
  let description = [{
    Preprocessing pass that recognizes the canonical Triton-emitted
    iota+reshape+broadcast+offset pointer-arithmetic chain feeding a
    `cuda_tile.load_ptr_tko` / `cuda_tile.store_ptr_tko` and rewrites it into
    the higher-level `cuda_tile.make_tensor_view` + `cuda_tile.make_partition_view`
    + `cuda_tile.load_view_tko` / `cuda_tile.store_view_tko` form, which the
    `convert-cuda-tile-to-mlir` pass can lower.
  }];
  let dependentDialects = [
    "cuda_tile::CudaTileDialect",
  ];
}

#endif // MLIR_CONVERSION_CUDATILETOMLIR_PASSES
