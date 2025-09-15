add_mlir_dialect_library(BuddyCocmh
  CocmhDialect.cpp
  CocmhOps.cpp
  # CocmhTypes.cpp
  # CocmhAttrs.cpp

  DEPENDS
    CocmhDialectIncGen
    CocmhOpsIncGen
    # CocmhTypesIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)