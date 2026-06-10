include_guard(GLOBAL)

# Optional integration of the Triton-to-tile-IR external test suite.
#
# When enabled, this builds `triton-cuda-tile-opt` from a Triton-to-tile-IR
# source checkout (letting Triton build its own pinned cuda-tile/tileir against
# the same LLVM) and runs the upstream tileir FileCheck inputs chained through
# our `cudatile-to-gpu` pipeline.
#
# Requirements (only for the external suite):
#   * TRITON_TO_TILEIR_DIR          - a Triton-to-tile-IR source checkout.
#   * TRITON_TO_TILEIR_LLVM_SYSPATH - an LLVM/MLIR install matching the revision
#                                     pinned in
#                                     <TRITON_TO_TILEIR_DIR>/cmake/llvm-hash.txt.
#                                     Triton's dialects only build against that
#                                     specific LLVM commit.
#
# If TRITON_TO_TILEIR_DIR is unset (and no sibling checkout exists) the suite is
# silently disabled and only the local tests run.
function(add_cudatile_external_tileir_tests)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(ARG "" "" "${multiValueArgs}" ${ARGN})

  # Auto-detect a sibling checkout when not specified explicitly.
  set(_default_dir "")
  if(EXISTS "${CMAKE_SOURCE_DIR}/../Triton-to-tile-IR/CMakeLists.txt")
    set(_default_dir "${CMAKE_SOURCE_DIR}/../Triton-to-tile-IR")
  endif()

  set(TRITON_TO_TILEIR_DIR "${_default_dir}" CACHE PATH
      "Path to a Triton-to-tile-IR source checkout (enables the external suite)")
  set(TRITON_TO_TILEIR_LLVM_SYSPATH "" CACHE PATH
      "LLVM/MLIR install matching Triton-to-tile-IR's pinned cmake/llvm-hash.txt")

  if(NOT TRITON_TO_TILEIR_DIR)
    message(STATUS
      "Triton-to-tile-IR external suite disabled (TRITON_TO_TILEIR_DIR not set)")
    return()
  endif()
  if(NOT EXISTS "${TRITON_TO_TILEIR_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "TRITON_TO_TILEIR_DIR does not contain CMakeLists.txt: ${TRITON_TO_TILEIR_DIR}")
  endif()
  if(NOT TRITON_TO_TILEIR_LLVM_SYSPATH)
    message(FATAL_ERROR
      "TRITON_TO_TILEIR_DIR is set but TRITON_TO_TILEIR_LLVM_SYSPATH is not.\n"
      "The external suite needs the LLVM revision pinned in "
      "${TRITON_TO_TILEIR_DIR}/cmake/llvm-hash.txt.\n"
      "Pass -DTRITON_TO_TILEIR_LLVM_SYSPATH=<llvm-install>, or unset "
      "TRITON_TO_TILEIR_DIR to disable the external suite.")
  endif()

  set(_build_dir "${CMAKE_BINARY_DIR}/_deps/triton-to-tile-ir-build")
  set(_cache_dir "${CMAKE_BINARY_DIR}/_deps/triton-cache")
  # Triton emits the tool into its tileir tools subdir, not a top-level bin/.
  set(_tool_dir
      "${_build_dir}/third_party/tileir/tools/triton-cuda-tile-opt")
  set(_tool_path "${_tool_dir}/triton-cuda-tile-opt")
  # cuda-tile that Triton self-builds against the pinned LLVM; this is what the
  # rest of the project reuses (see CUDA_TILE_FROM_TRITON in the top CMakeLists).
  set(_cuda_tile_lib
      "${_build_dir}/third_party/tileir/tileir_src/build/install/lib/libCudaTileDialect.a")
  # Triton clones cuda-tile into this directory during its configure step.
  set(_tileir_src "${_build_dir}/third_party/tileir/tileir_src")

  # Pass the backend list through an initial-cache file: a bare ';' on the
  # command line would be eaten by the shell that runs the custom command.
  set(_init_cache "${CMAKE_BINARY_DIR}/_deps/triton-init-cache.cmake")
  file(WRITE "${_init_cache}"
    "set(TRITON_CODEGEN_BACKENDS \"nvidia;tileir\" CACHE STRING \"\" FORCE)\n")

  # A partial/aborted previous run can leave tileir_src without a .git; Triton's
  # clone guard then tries to clone into a non-empty dir and fails. This script
  # removes such a broken checkout first (a valid one, with .git, is preserved).
  set(_clean_script "${CMAKE_BINARY_DIR}/_deps/triton-clean-stale-tileir.cmake")
  file(WRITE "${_clean_script}"
    "if(EXISTS \"${_tileir_src}\" AND NOT EXISTS \"${_tileir_src}/.git\")\n"
    "  message(STATUS \"Removing stale cuda-tile checkout: ${_tileir_src}\")\n"
    "  file(REMOVE_RECURSE \"${_tileir_src}\")\n"
    "endif()\n")

  # Forward toolchain settings so the Triton sub-build (and the cuda-tile it
  # bootstraps) use the same compilers/linker as this project instead of
  # falling back to the system default cc/c++.
  set(_optional_args)
  if(CMAKE_MAKE_PROGRAM)
    list(APPEND _optional_args -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM})
  endif()
  if(CMAKE_BUILD_TYPE)
    list(APPEND _optional_args -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
  endif()
  if(CMAKE_C_COMPILER)
    list(APPEND _optional_args -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
  endif()
  if(CMAKE_CXX_COMPILER)
    list(APPEND _optional_args -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
  endif()
  if(LLVM_USE_LINKER)
    list(APPEND _optional_args -DLLVM_USE_LINKER=${LLVM_USE_LINKER})
  endif()

  # The cuda-tile that Triton bootstraps is built by a helper shell script
  # (third_party/tileir/scripts/build_cuda_tile.sh) whose `cmake` invocation
  # specifies no compiler, so it would otherwise pick the system default c++.
  # Triton runs that script through `cmake -E env` (which preserves the rest of
  # the environment), and CMake honours the CC/CXX environment variables, so we
  # export them here to steer the cuda-tile build onto the same compilers.
  set(_env_args)
  if(CMAKE_C_COMPILER)
    list(APPEND _env_args CC=${CMAKE_C_COMPILER})
  endif()
  if(CMAKE_CXX_COMPILER)
    list(APPEND _env_args CXX=${CMAKE_CXX_COMPILER})
  endif()

  # Build triton-cuda-tile-opt on demand. Triton clones and builds its own
  # pinned cuda-tile against TRITON_TO_TILEIR_LLVM_SYSPATH, so we pass nothing
  # about our local cuda-tile here. Declaring the tool and cuda-tile library as
  # OUTPUTs means this heavy step runs once and is then skipped on rebuilds.
  add_custom_command(
    OUTPUT ${_tool_path} ${_cuda_tile_lib}
    # Drop a broken (no .git) leftover clone so Triton can re-clone cleanly.
    COMMAND ${CMAKE_COMMAND} -P ${_clean_script}
    COMMAND ${CMAKE_COMMAND} -E env ${_env_args} ${CMAKE_COMMAND}
      -S ${TRITON_TO_TILEIR_DIR}
      -B ${_build_dir}
      -G ${CMAKE_GENERATOR}
      -C ${_init_cache}
      ${_optional_args}
      -DLLVM_SYSPATH=${TRITON_TO_TILEIR_LLVM_SYSPATH}
      -DTRITON_BUILD_PYTHON_MODULE=OFF
      -DTRITON_BUILD_PROTON=OFF
      -DTRITON_BUILD_UT=OFF
      -DTRITON_CACHE_PATH=${_cache_dir}
    COMMAND ${CMAKE_COMMAND} --build ${_build_dir} --target triton-cuda-tile-opt
    USES_TERMINAL
    COMMENT "Building triton-cuda-tile-opt + cuda-tile from ${TRITON_TO_TILEIR_DIR}"
  )
  add_custom_target(build-triton-cuda-tile-opt DEPENDS ${_tool_path})

  # When the project reuses Triton's self-built cuda-tile, our own targets must
  # not compile/link before that cuda-tile (headers + libraries) exists.
  if(CUDA_TILE_FROM_TRITON)
    foreach(_t CudaTileToGPU cudatile-to-gpu)
      if(TARGET ${_t})
        add_dependencies(${_t} build-triton-cuda-tile-opt)
      endif()
    endforeach()
  endif()

  set(TILEIR_EXTERNAL_TEST_DIR
      "${TRITON_TO_TILEIR_DIR}/third_party/tileir/test/FileCheck"
      CACHE PATH "External tileir test inputs")
  set(TILEIR_UPSTREAM_TOOL_DIR "${_tool_dir}" CACHE PATH
      "Directory containing the built triton-cuda-tile-opt" FORCE)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/External/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/External/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/External/lit.cfg.py
  )

  add_lit_testsuite(check-cudatile-external
    "Chaining external tileir tests through cudatile-to-gpu"
    ${CMAKE_CURRENT_BINARY_DIR}/External
    DEPENDS ${ARG_DEPENDS} build-triton-cuda-tile-opt
  )
  set_target_properties(check-cudatile-external PROPERTIES FOLDER "Tests")
endfunction()
