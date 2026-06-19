set(LLVM_TARGET_DEFINITIONS XVectorOps.td)

mlir_tablegen(XVectorAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(XVectorAttrs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(XVectorEnums.h.inc -gen-enum-decls)
mlir_tablegen(XVectorEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(XVectorTypes.h.inc -gen-typedef-decls)
mlir_tablegen(XVectorTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(XVectorOps.h.inc -gen-op-decls)
mlir_tablegen(XVectorOps.cpp.inc -gen-op-defs)
mlir_tablegen(XVectorDialect.h.inc -gen-dialect-decls)
mlir_tablegen(XVectorDialect.cpp.inc -gen-dialect-defs)

add_public_tablegen_target(MLIRXVectorOpsIncGen)