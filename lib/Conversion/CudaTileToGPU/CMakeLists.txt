add_library(CudaTileToGPU STATIC
  CudaTileToGPU.cpp
  TileIRPtrToView.cpp
)

target_link_libraries(CudaTileToGPU
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
