add_library(TileIRToMLIR STATIC
  ConvertMemrefArgsToRankedMemref.cpp
  ConvertMemrefArgsToPtrArgs.cpp
  TileIRToMLIR.cpp
  TileIRPtrToView.cpp
)

add_dependencies(TileIRToMLIR MLIRTileIRToMLIRConversionPassIncGen)
if(TILEIR_TO_MLIR_EXTERNAL_PROJECT_BUILD)
  add_dependencies(TileIRToMLIR CudaTileIncGen)
endif()

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
