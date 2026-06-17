set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CudaTileToMLIR)
add_public_tablegen_target(MLIRCudaTileToMLIRConversionPassIncGen)
