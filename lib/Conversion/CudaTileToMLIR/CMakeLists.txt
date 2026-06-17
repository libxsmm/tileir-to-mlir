add_library(CudaTileToMLIR STATIC
  CudaTileToMLIR.cpp
  TileIRPtrToView.cpp
)

add_dependencies(CudaTileToMLIR MLIRCudaTileToMLIRConversionPassIncGen)

target_link_libraries(CudaTileToMLIR
  PRIVATE
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRUBDialect
  MLIRVectorDialect
  MLIRTransforms
  MLIRPass
  MLIRIR
  CudaTileDialect
)
