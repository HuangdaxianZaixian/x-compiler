set(LLVM_TARGET_DEFINITIONS TopDialectPasses.td)
mlir_tablegen(TopDialectPasses.h.inc -gen-pass-decls -name Top)
add_public_tablegen_target(TopDialectPassesIncGen)