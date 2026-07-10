get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
)
set(LIBS
  MLIROptLib
  XCMLIRTOP
  XCMLIRTILING
  ${dialect_libs}
  ${conversion_libs}
)
add_llvm_executable(xc-mlir-opt
  xc-mlir-opt.cpp
  DEPENDS
  ${LIBS}
)
target_link_libraries(xc-mlir-opt PRIVATE ${LIBS})
llvm_update_compile_flags(xc-mlir-opt)
mlir_check_all_link_libraries(xc-mlir-opt)

