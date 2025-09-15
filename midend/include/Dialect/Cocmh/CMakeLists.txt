set(LLVM_TARGET_DEFINITIONS CocmhOps.td)
mlir_tablegen(CocmhOps.h.inc -gen-op-decls)
mlir_tablegen(CocmhOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(CocmhOpsIncGen)

set(LLVM_TARGET_DEFINITIONS CocmhDialect.td)
mlir_tablegen(CocmhDialect.h.inc -gen-dialect-decls -dialect=cocmh)
mlir_tablegen(CocmhDialect.cpp.inc -gen-dialect-defs -dialect=cocmh)
add_public_tablegen_target(CocmhDialectIncGen)

# 如有 Types/Attrs
# set(LLVM_TARGET_DEFINITIONS CocmhTypes.td)
# mlir_tablegen(CocmhTypes.h.inc -gen-typedef-decls)
# mlir_tablegen(CocmhTypes.cpp.inc -gen-typedef-defs)
# add_public_tablegen_target(CocmhTypesIncGen)

# 文档（可选）
# add_mlir_doc(CocmhOps CocmhOps Dialects/ -gen-op-doc)
