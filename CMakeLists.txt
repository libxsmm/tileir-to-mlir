cmake_minimum_required(VERSION 3.20)
project(VectorToCudaTile LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Honor -DCMAKE_LINKER=<name-or-path> for compiler-driver links (clang/gcc).
# This also marks CMAKE_LINKER as used, so CMake will not report it as unused.
if(CMAKE_LINKER)
  set(_v2ct_linker_raw "${CMAKE_LINKER}")

  if(IS_ABSOLUTE "${_v2ct_linker_raw}")
    set(_v2ct_linker_path "${_v2ct_linker_raw}")
  else()
    find_program(_v2ct_linker_path
      NAMES "${_v2ct_linker_raw}" "ld.${_v2ct_linker_raw}")
    if(NOT _v2ct_linker_path)
      message(FATAL_ERROR
        "CMAKE_LINKER='${_v2ct_linker_raw}' was requested, but no matching linker "
        "executable was found in PATH.")
    endif()
  endif()

  # Keep CMAKE_LINKER canonicalized to the resolved executable path.
  set(CMAKE_LINKER "${_v2ct_linker_path}" CACHE FILEPATH
      "Path to linker executable" FORCE)

  # Map common linker executable names to -fuse-ld values.
  get_filename_component(_v2ct_linker_name "${_v2ct_linker_path}" NAME)
  set(_v2ct_fuse_ld "${_v2ct_linker_name}")
  if(_v2ct_linker_name STREQUAL "ld.lld")
    set(_v2ct_fuse_ld "lld")
  elseif(_v2ct_linker_name MATCHES "^ld\\.")
    string(REGEX REPLACE "^ld\\." "" _v2ct_fuse_ld "${_v2ct_linker_name}")
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_link_options("-fuse-ld=${_v2ct_fuse_ld}")
    message(STATUS
      "Using requested linker: ${CMAKE_LINKER} (-fuse-ld=${_v2ct_fuse_ld})")
  else()
    message(WARNING
      "CMAKE_LINKER is set to '${CMAKE_LINKER}', but automatic propagation via "
      "-fuse-ld is only implemented for GNU/Clang C++ compilers.")
  endif()

  unset(_v2ct_linker_raw)
  unset(_v2ct_linker_path)
  unset(_v2ct_linker_name)
  unset(_v2ct_fuse_ld)
endif()

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

if(NOT CUDA_TILE_DIR)
  message(FATAL_ERROR
    "CUDA_TILE_DIR is not set.\n"
    "Pass -DCUDA_TILE_DIR=<cuda-tile-root> (built against the same LLVM "
    "as this project).")
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

# The artifacts must already exist.
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
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

#===----------------------------------------------------------------------===#
# Subdirectories
#===----------------------------------------------------------------------===#

add_subdirectory(include/mlir/Conversion/CudaTileToMLIR)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
