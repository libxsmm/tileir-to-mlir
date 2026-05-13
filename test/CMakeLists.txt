# Configure lit for the project tests.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CUDATILE_TO_GPU_TEST_DEPENDS
  FileCheck
  mlir-opt
  cudatile-to-gpu
)

add_lit_testsuite(check-cudatile-to-gpu
  "Running the CudaTileToGPU regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CUDATILE_TO_GPU_TEST_DEPENDS}
)
set_target_properties(check-cudatile-to-gpu PROPERTIES FOLDER "Tests")

add_lit_testsuites(cudatile-to-gpu
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CUDATILE_TO_GPU_TEST_DEPENDS}
)

# Convenience alias: allow `cmake --build build --target test`.
if(TARGET test)
  add_dependencies(test check-cudatile-to-gpu)
else()
  add_custom_target(test DEPENDS check-cudatile-to-gpu)
endif()

add_subdirectory(Conversion)
