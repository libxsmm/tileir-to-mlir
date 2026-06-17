add_library(CudaTileToMLIR STATIC
  CudaTileToMLIR.cpp
  TileIRPtrToView.cpp
)

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
