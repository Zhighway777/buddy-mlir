configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MIDEND_TEST_DEPENDS
  buddy-opt
)

add_lit_testsuite(check-midend "Checking the buddy-mlir midend tests..."
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MIDEND_TEST_DEPENDS}
)

add_lit_testsuites(BUDDY-MIDEND ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MIDEND_TEST_DEPENDS})

add_subdirectory(Dialect)

