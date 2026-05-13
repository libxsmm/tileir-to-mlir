cmake_minimum_required(VERSION 3.20)
project(VectorToCudaTile LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Match LLVM/MLIR: no RTTI, no exceptions.
if(NOT LLVM_ENABLE_RTTI)
  if(MSVC)
    string(APPEND CMAKE_CXX_FLAGS " /GR-")
  else()
    string(APPEND CMAKE_CXX_FLAGS " -fno-rtti")
  endif()
endif()

#===----------------------------------------------------------------------===#
# Dependencies
#===----------------------------------------------------------------------===#

# MLIR (also pulls in LLVM).
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# CudaTile dialect.
set(CUDA_TILE_DIR "" CACHE PATH
    "Root directory of the cuda-tile project (contains include/ and build/)")
if(NOT CUDA_TILE_DIR)
  message(FATAL_ERROR
    "CUDA_TILE_DIR is not set. Pass -DCUDA_TILE_DIR=<path-to-cuda-tile-root>")
endif()

set(CUDA_TILE_INCLUDE_DIR "${CUDA_TILE_DIR}/include")
set(CUDA_TILE_BUILD_INCLUDE_DIR "${CUDA_TILE_DIR}/build/include")
set(CUDA_TILE_LIB_DIR "${CUDA_TILE_DIR}/build/lib")

include_directories(${CUDA_TILE_INCLUDE_DIR})
include_directories(${CUDA_TILE_BUILD_INCLUDE_DIR})
link_directories(${CUDA_TILE_LIB_DIR})

#===----------------------------------------------------------------------===#
# Project includes
#===----------------------------------------------------------------------===#

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#===----------------------------------------------------------------------===#
# Subdirectories
#===----------------------------------------------------------------------===#

add_subdirectory(lib)
add_subdirectory(tools)
