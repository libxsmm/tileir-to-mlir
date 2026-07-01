add_library(TileIRToMLIR STATIC
  ConvertMemrefArgsToRankedMemref.cpp
  ConvertMemrefArgsToPtrArgs.cpp
  TileIRToMLIR.cpp
  TileIRPtrToView.cpp
)

add_dependencies(TileIRToMLIR MLIRTileIRToMLIRConversionPassIncGen)

target_link_libraries(TileIRToMLIR
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
