add_library(CudaTileToMLIR STATIC
  ConvertMemrefArgsToRankedMemref.cpp
  ConvertMemrefArgsToPtrArgs.cpp
  CudaTileToMLIR.cpp
  TileIRPtrToView.cpp
)

add_dependencies(CudaTileToMLIR MLIRCudaTileToMLIRConversionPassIncGen)

target_link_libraries(CudaTileToMLIR
  PRIVATE
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRUBDialect
  MLIRVectorDialect
  MLIRTransforms
  MLIRPass
  MLIRIR
  CudaTileDialect
)
