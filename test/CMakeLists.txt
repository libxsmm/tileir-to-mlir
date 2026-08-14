# Configure lit for the project tests.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TILEIR_TO_MLIR_TEST_DEPENDS
  FileCheck
  not
  tileir-to-mlir
)

add_lit_testsuite(check-tileir-to-mlir
  "Running the TileIRToMLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TILEIR_TO_MLIR_TEST_DEPENDS}
)
set_target_properties(check-tileir-to-mlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(tileir-to-mlir
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${TILEIR_TO_MLIR_TEST_DEPENDS}
)

# Convenience alias: allow `cmake --build build --target test`.
if(TARGET test)
  add_dependencies(test check-tileir-to-mlir)
else()
  add_custom_target(test DEPENDS check-tileir-to-mlir)
endif()
