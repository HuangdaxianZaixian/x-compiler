file(GLOB_RECURSE _srcs 
    *.cpp)

add_mlir_library(XCUTILS
    ${_srcs}

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRParser
    MLIRFuncDialect
    MLIRRewrite
    MLIRTransforms
    MLIRSideEffectInterfaces
)