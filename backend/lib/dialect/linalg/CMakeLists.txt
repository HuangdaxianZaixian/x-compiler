add_mlir_library(XCMLIRTILING
  TilingConfigManager.cpp

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRPass
  MLIRTransforms
)