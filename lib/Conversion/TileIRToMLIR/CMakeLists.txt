add_library(TileIRToMLIR STATIC
  CanonicalizeFromTileIR.cpp
  ConvertMemrefArgsToRankedMemref.cpp
  ConvertMemrefArgsToPtrArgs.cpp
  PackCPUKernelArguments.cpp
  Pipeline.cpp
  TileIRToMLIR.cpp
  TileIRPtrToView.cpp
)

target_include_directories(TileIRToMLIR PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_BINARY_DIR}/include"
  "${CUDA_TILE_PUBLIC_INCLUDE_DIR}"
  "${CUDA_TILE_GEN_INCLUDE_DIR}"
)

add_dependencies(TileIRToMLIR MLIRTileIRToMLIRConversionPassIncGen)
if(TILEIR_TO_MLIR_EXTERNAL_PROJECT_BUILD)
  add_dependencies(TileIRToMLIR CudaTileIncGen)
endif()

target_link_libraries(TileIRToMLIR
  PRIVATE
  MLIRAnalysis
  MLIRArithDialect
  MLIRArithUtils
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRUBDialect
  MLIRVectorDialect
  MLIRTransforms
  MLIRPass
  MLIRIR
  CudaTileDialect
)
