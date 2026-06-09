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

# Use a single LLVM/MLIR for the whole project. When the optional
# Triton-to-tile-IR external suite is requested, its pinned LLVM
# (TRITON_TO_TILEIR_LLVM_SYSPATH) is the one everything must build against, so
# default MLIR_DIR to it unless the user overrides MLIR_DIR explicitly.
if(NOT MLIR_DIR AND TRITON_TO_TILEIR_LLVM_SYSPATH)
  set(MLIR_DIR "${TRITON_TO_TILEIR_LLVM_SYSPATH}/lib/cmake/mlir" CACHE PATH
      "Path to MLIRConfig.cmake" FORCE)
endif()

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
#
# There are two ways to provide cuda-tile:
#
#   1. Reuse the cuda-tile that the optional Triton-to-tile-IR external suite
#      self-builds against its pinned LLVM. This is the default whenever a
#      Triton-to-tile-IR checkout is available, and gives a single LLVM and a
#      single cuda-tile across the whole project. The artifacts are produced on
#      demand by the `build-triton-cuda-tile-opt` target, so they need not exist
#      at configure time (build-order dependencies are added in
#      cmake/BuildTritonToTileIR.cmake).
#
#   2. Point CUDA_TILE_DIR at an explicit cuda-tile (source/build root or install
#      dir) built against the same LLVM as this project. Use this when
#      configuring without Triton-to-tile-IR.
#
# CUDA_TILE_DIR may point either at:
#   * a cuda-tile source/build root (contains include/ + build/include + build/lib), or
#   * a cuda-tile install dir (contains lib/ + include/include/). An install does
#     not ship the tablegen-generated *.h.inc headers, so in that case the sibling
#     build tree (<root>/build/include) is used to locate them.
# The detected locations can always be overridden explicitly on the command line.
set(CUDA_TILE_DIR "" CACHE PATH
    "Root directory of the cuda-tile project (source/build root or install dir)")

set(CUDA_TILE_PUBLIC_INCLUDE_DIR "" CACHE PATH
    "Directory containing cuda-tile public headers (cuda_tile/Dialect/...)")
set(CUDA_TILE_GEN_INCLUDE_DIR "" CACHE PATH
    "Directory containing cuda-tile generated headers (*.h.inc)")
set(CUDA_TILE_LIB_DIR "" CACHE PATH
    "Directory containing libCudaTileDialect.a and friends")

# Deterministic locations of the cuda-tile that Triton-to-tile-IR self-builds.
set(_triton_ct_root
    "${CMAKE_BINARY_DIR}/_deps/triton-to-tile-ir-build/third_party/tileir/tileir_src/build")

# Is a Triton-to-tile-IR checkout available (explicit or sibling)?
set(_triton_dir "${TRITON_TO_TILEIR_DIR}")
if(NOT _triton_dir AND EXISTS "${CMAKE_SOURCE_DIR}/../Triton-to-tile-IR/CMakeLists.txt")
  set(_triton_dir "${CMAKE_SOURCE_DIR}/../Triton-to-tile-IR")
endif()

set(CUDA_TILE_FROM_TRITON OFF CACHE INTERNAL "cuda-tile is reused from Triton's self-build")
if(NOT CUDA_TILE_DIR AND _triton_dir)
  set(CUDA_TILE_FROM_TRITON ON CACHE INTERNAL "" FORCE)
endif()

if(CUDA_TILE_FROM_TRITON)
  # Reuse Triton's self-built cuda-tile (install layout under tileir_src/build).
  # These are derived, not user-overridable in this mode; use an explicit
  # CUDA_TILE_DIR (without Triton-to-tile-IR) if you need custom paths.
  set(CUDA_TILE_LIB_DIR "${_triton_ct_root}/install/lib" CACHE PATH "" FORCE)
  set(CUDA_TILE_PUBLIC_INCLUDE_DIR "${_triton_ct_root}/install/include/include" CACHE PATH "" FORCE)
  set(CUDA_TILE_GEN_INCLUDE_DIR "${_triton_ct_root}/include" CACHE PATH "" FORCE)
  message(STATUS "cuda-tile: reusing Triton-to-tile-IR self-built install (built on demand)")
else()
  if(NOT CUDA_TILE_DIR)
    message(FATAL_ERROR
      "CUDA_TILE_DIR is not set and no Triton-to-tile-IR checkout was found.\n"
      "Either pass -DCUDA_TILE_DIR=<cuda-tile-root> (built against the same LLVM "
      "as this project), or pass -DTRITON_TO_TILEIR_DIR=<checkout> to reuse the "
      "cuda-tile that Triton-to-tile-IR self-builds.")
  endif()

  if(NOT CUDA_TILE_PUBLIC_INCLUDE_DIR OR NOT CUDA_TILE_GEN_INCLUDE_DIR OR NOT CUDA_TILE_LIB_DIR)
    if(EXISTS "${CUDA_TILE_DIR}/build/lib/libCudaTileDialect.a")
      # Source/build root layout.
      set(_ct_lib_dir     "${CUDA_TILE_DIR}/build/lib")
      set(_ct_public_dir  "${CUDA_TILE_DIR}/include")
      set(_ct_gen_dir     "${CUDA_TILE_DIR}/build/include")
    elseif(EXISTS "${CUDA_TILE_DIR}/lib/libCudaTileDialect.a")
      # Install layout. Generated headers are not installed; find the sibling build tree.
      set(_ct_lib_dir     "${CUDA_TILE_DIR}/lib")
      set(_ct_public_dir  "${CUDA_TILE_DIR}/include/include")
      get_filename_component(_ct_root "${CUDA_TILE_DIR}" DIRECTORY)
      set(_ct_gen_dir     "${_ct_root}/build/include")
    else()
      message(FATAL_ERROR
        "Could not find libCudaTileDialect.a under CUDA_TILE_DIR='${CUDA_TILE_DIR}'. "
        "Expected either '<dir>/build/lib' (source/build root) or '<dir>/lib' (install). "
        "Override paths with -DCUDA_TILE_LIB_DIR/-DCUDA_TILE_PUBLIC_INCLUDE_DIR/"
        "-DCUDA_TILE_GEN_INCLUDE_DIR.")
    endif()

    if(NOT CUDA_TILE_LIB_DIR)
      set(CUDA_TILE_LIB_DIR "${_ct_lib_dir}" CACHE PATH "" FORCE)
    endif()
    if(NOT CUDA_TILE_PUBLIC_INCLUDE_DIR)
      set(CUDA_TILE_PUBLIC_INCLUDE_DIR "${_ct_public_dir}" CACHE PATH "" FORCE)
    endif()
    if(NOT CUDA_TILE_GEN_INCLUDE_DIR)
      set(CUDA_TILE_GEN_INCLUDE_DIR "${_ct_gen_dir}" CACHE PATH "" FORCE)
    endif()
  endif()

  # The artifacts must already exist when cuda-tile is provided explicitly.
  if(NOT EXISTS "${CUDA_TILE_PUBLIC_INCLUDE_DIR}/cuda_tile/Dialect/CudaTile/IR/Dialect.h")
    message(FATAL_ERROR
      "cuda-tile public headers not found in CUDA_TILE_PUBLIC_INCLUDE_DIR="
      "'${CUDA_TILE_PUBLIC_INCLUDE_DIR}'")
  endif()
  if(NOT EXISTS "${CUDA_TILE_GEN_INCLUDE_DIR}/cuda_tile/Dialect/CudaTile/IR/Ops.h.inc")
    message(FATAL_ERROR
      "cuda-tile generated headers (*.h.inc) not found in CUDA_TILE_GEN_INCLUDE_DIR="
      "'${CUDA_TILE_GEN_INCLUDE_DIR}'. These live only in the cuda-tile build tree; "
      "point CUDA_TILE_DIR at a source/build root or ensure '<root>/build' exists next "
      "to the install dir.")
  endif()
endif()

message(STATUS "cuda-tile public includes:    ${CUDA_TILE_PUBLIC_INCLUDE_DIR}")
message(STATUS "cuda-tile generated includes: ${CUDA_TILE_GEN_INCLUDE_DIR}")
message(STATUS "cuda-tile libraries:          ${CUDA_TILE_LIB_DIR}")

include_directories(${CUDA_TILE_PUBLIC_INCLUDE_DIR})
include_directories(${CUDA_TILE_GEN_INCLUDE_DIR})
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
add_subdirectory(test)
