set(LLVM_TARGET_DEFINITIONS TopOps.td)

mlir_tablegen(TopOps.h.inc -gen-op-decls)
mlir_tablegen(TopOps.cpp.inc -gen-op-defs)
mlir_tablegen(TopDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TopDialect.cpp.inc -gen-dialect-defs)

add_public_tablegen_target(MLIRTopOpsIncGen)

