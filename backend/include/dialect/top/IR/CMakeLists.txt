set(LLVM_TARGET_DEFINITIONS TopOps.td)

mlir_tablegen(TopAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(TopAttrs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(TopTypes.h.inc -gen-typedef-decls)
mlir_tablegen(TopTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(TopOps.h.inc -gen-op-decls)
mlir_tablegen(TopOps.cpp.inc -gen-op-defs)
mlir_tablegen(TopDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TopDialect.cpp.inc -gen-dialect-defs)

add_public_tablegen_target(MLIRTopOpsIncGen)

