cmake_minimum_required(VERSION 3.20)
project(TileIRToMLIR LANGUAGES C CXX)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Honor -DCMAKE_LINKER=<name-or-path> for standalone compiler-driver links
# (clang/gcc). An LLVM external-project build owns linker configuration itself.
if(CMAKE_LINKER AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
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

# MLIR (also pulls in LLVM). When this project is included through
# LLVM_EXTERNAL_PROJECTS, inherit the already-configured MLIR build instead of
# searching for an installed package.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(TILEIR_TO_MLIR_EXTERNAL_PROJECT_BUILD FALSE)
  find_package(MLIR REQUIRED CONFIG)

  # Define the default argument to use by `lit` when testing.
  set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
  set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib")
  set(MLIR_BINARY_DIR "${CMAKE_BINARY_DIR}")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  set(TILEIR_TO_MLIR_EXTERNAL_PROJECT_BUILD TRUE)

  # Use an explicit revision override when supplied; otherwise, use the pin
  # shared with the build scripts and CI. Requiring the exact checkout keeps
  # the in-tree CUDA Tile and conversion APIs aligned.
  if(DEFINED USE_LLVM_REVISION)
    set(TILEIR_TO_MLIR_REQUIRED_LLVM_REVISION "${USE_LLVM_REVISION}")
  else()
    set(TILEIR_TO_MLIR_LLVM_REVISION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../llvm-revision.txt")
    if(NOT EXISTS "${TILEIR_TO_MLIR_LLVM_REVISION_FILE}")
      message(FATAL_ERROR "Missing pinned LLVM revision file: ${TILEIR_TO_MLIR_LLVM_REVISION_FILE}")
    endif()
    # Re-configure whenever the pin changes.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${TILEIR_TO_MLIR_LLVM_REVISION_FILE}")
    file(STRINGS "${TILEIR_TO_MLIR_LLVM_REVISION_FILE}" TILEIR_TO_MLIR_REQUIRED_LLVM_REVISION LIMIT_COUNT 1)
    string(STRIP "${TILEIR_TO_MLIR_REQUIRED_LLVM_REVISION}" TILEIR_TO_MLIR_REQUIRED_LLVM_REVISION)
  endif()

  find_program(_tileir_to_mlir_git_executable git REQUIRED)
  execute_process(
    COMMAND "${_tileir_to_mlir_git_executable}" -C "${LLVM_MAIN_SRC_DIR}" rev-parse HEAD
    RESULT_VARIABLE _tileir_to_mlir_llvm_revision_result
    OUTPUT_VARIABLE _tileir_to_mlir_llvm_revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(NOT _tileir_to_mlir_llvm_revision_result EQUAL 0)
    message(FATAL_ERROR
      "TileIRToMLIR requires an LLVM Git checkout at "
      "${TILEIR_TO_MLIR_REQUIRED_LLVM_REVISION}.")
  endif()
  if(NOT _tileir_to_mlir_llvm_revision STREQUAL
         TILEIR_TO_MLIR_REQUIRED_LLVM_REVISION)
    message(FATAL_ERROR
      "TileIRToMLIR requires LLVM revision "
      "${TILEIR_TO_MLIR_REQUIRED_LLVM_REVISION}, but found "
      "${_tileir_to_mlir_llvm_revision}.")
  endif()
  unset(_tileir_to_mlir_git_executable)
  unset(_tileir_to_mlir_llvm_revision_result)
  unset(_tileir_to_mlir_llvm_revision)

  set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
  set(LLVM_LIBRARY_OUTPUT_INTDIR
      "${CMAKE_BINARY_DIR}/tools/TileIRToMLIR/lib")
  set(MLIR_BINARY_DIR "${CMAKE_BINARY_DIR}")
  set(MLIR_MAIN_SRC_DIR "${LLVM_MAIN_SRC_DIR}/../mlir")
  set(MLIR_INCLUDE_DIR "${MLIR_MAIN_SRC_DIR}/include")
  set(MLIR_GENERATED_INCLUDE_DIR "${LLVM_BINARY_DIR}/tools/mlir/include")
  set(MLIR_INCLUDE_DIRS
      "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_MAIN_SRC_DIR}/cmake/modules")
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# TileIR dialect.
#
# In a monolithic LLVM build, CUDA Tile is added directly from the pinned
# submodule and shares the enclosing MLIR configuration. In standalone mode,
# CUDA_TILE_DIR identifies an already-built CUDA Tile tree.
set(TILEIR_TO_MLIR_CUDA_TILE_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cuda-tile" CACHE PATH
  "CUDA Tile source directory used for an LLVM external-project build")

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

if(TILEIR_TO_MLIR_EXTERNAL_PROJECT_BUILD)
  if(NOT EXISTS "${TILEIR_TO_MLIR_CUDA_TILE_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "CUDA Tile sources are required for an LLVM external-project build.\n"
      "Initialize the third_party/cuda-tile submodule or pass "
      "-DTILEIR_TO_MLIR_CUDA_TILE_SOURCE_DIR=<cuda-tile-source-dir>.")
  endif()

  set(CUDA_TILE_BUILD_IN_LLVM ON CACHE BOOL
      "Build CUDA Tile within the enclosing LLVM build" FORCE)
  set(CUDA_TILE_ENABLE_CAPI OFF CACHE BOOL
      "Disable CUDA Tile C API for TileIRToMLIR's in-tree dependency" FORCE)
  set(CUDA_TILE_ENABLE_TESTING OFF CACHE BOOL
      "Disable CUDA Tile tests for TileIRToMLIR's in-tree dependency" FORCE)
  set(CUDA_TILE_ENABLE_TOOLS OFF CACHE BOOL
      "Disable CUDA Tile tools for TileIRToMLIR's in-tree dependency" FORCE)
  set(CUDA_TILE_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL
      "Disable CUDA Tile Python bindings for TileIRToMLIR's in-tree dependency" FORCE)

  add_subdirectory("${TILEIR_TO_MLIR_CUDA_TILE_SOURCE_DIR}"
                   "${CMAKE_CURRENT_BINARY_DIR}/cuda-tile" EXCLUDE_FROM_ALL)

  # CUDA Tile changes these shared output variables for its own targets.
  set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
  set(LLVM_LIBRARY_OUTPUT_INTDIR
      "${CMAKE_BINARY_DIR}/tools/TileIRToMLIR/lib")

  set(CUDA_TILE_PUBLIC_INCLUDE_DIR
      "${TILEIR_TO_MLIR_CUDA_TILE_SOURCE_DIR}/include")
  set(CUDA_TILE_GEN_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/cuda-tile/include")
else()
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
endif()

if(NOT EXISTS "${CUDA_TILE_PUBLIC_INCLUDE_DIR}/cuda_tile/Dialect/CudaTile/IR/Dialect.h")
  message(FATAL_ERROR
    "cuda-tile public headers not found in CUDA_TILE_PUBLIC_INCLUDE_DIR="
    "'${CUDA_TILE_PUBLIC_INCLUDE_DIR}'")
endif()
if(NOT TILEIR_TO_MLIR_EXTERNAL_PROJECT_BUILD AND
   NOT EXISTS "${CUDA_TILE_GEN_INCLUDE_DIR}/cuda_tile/Dialect/CudaTile/IR/Ops.h.inc")
  message(FATAL_ERROR
    "cuda-tile generated headers (*.h.inc) not found in CUDA_TILE_GEN_INCLUDE_DIR="
    "'${CUDA_TILE_GEN_INCLUDE_DIR}'. These live only in the cuda-tile build tree; "
    "point CUDA_TILE_DIR at a source/build root or ensure '<root>/build' exists next "
    "to the install dir.")
endif()

message(STATUS "cuda-tile public includes:    ${CUDA_TILE_PUBLIC_INCLUDE_DIR}")
message(STATUS "cuda-tile generated includes: ${CUDA_TILE_GEN_INCLUDE_DIR}")
if(TILEIR_TO_MLIR_EXTERNAL_PROJECT_BUILD)
  message(STATUS "cuda-tile libraries:          in-tree targets")
else()
  message(STATUS "cuda-tile libraries:          ${CUDA_TILE_LIB_DIR}")
endif()

include_directories(${CUDA_TILE_PUBLIC_INCLUDE_DIR})
include_directories(${CUDA_TILE_GEN_INCLUDE_DIR})
if(NOT TILEIR_TO_MLIR_EXTERNAL_PROJECT_BUILD)
  link_directories(${CUDA_TILE_LIB_DIR})
endif()

#===----------------------------------------------------------------------===#
# Project includes
#===----------------------------------------------------------------------===#

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

if(TILEIR_TO_MLIR_EXTERNAL_PROJECT_BUILD)
  set(TILEIR_TO_MLIR_TOOL_DIR "${LLVM_RUNTIME_OUTPUT_INTDIR}")
else()
  set(TILEIR_TO_MLIR_TOOL_DIR "${PROJECT_BINARY_DIR}/tools")
endif()

#===----------------------------------------------------------------------===#
# Subdirectories
#===----------------------------------------------------------------------===#

add_subdirectory(include/mlir/Conversion/TileIRToMLIR)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)

install(TARGETS TileIRToMLIR
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(TARGETS tileir-to-mlir
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/mlir/Conversion/TileIRToMLIR"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mlir/Conversion"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.td"
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/include/mlir/Conversion/TileIRToMLIR/Passes.h.inc"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mlir/Conversion/TileIRToMLIR"
)
