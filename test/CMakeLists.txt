# Configure lit for the project tests.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CUDATILE_TO_MLIR_TEST_DEPENDS
  FileCheck
  mlir-opt
  cudatile-to-mlir
)

add_lit_testsuite(check-cudatile-to-mlir
  "Running the CudaTileToMLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CUDATILE_TO_MLIR_TEST_DEPENDS}
)
set_target_properties(check-cudatile-to-mlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(cudatile-to-mlir
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CUDATILE_TO_MLIR_TEST_DEPENDS}
)

# Convenience alias: allow `cmake --build build --target test`.
if(TARGET test)
  add_dependencies(test check-cudatile-to-mlir)
else()
  add_custom_target(test DEPENDS check-cudatile-to-mlir)
endif()
