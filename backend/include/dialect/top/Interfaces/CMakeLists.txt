set(LLVM_TARGET_DEFINITIONS OpInterfaces.td)
mlir_tablegen(TopInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(TopInterfaces.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(MLIRTopOpInterfacesIncGen)
