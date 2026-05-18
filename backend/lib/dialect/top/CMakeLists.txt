file(GLOB_RECURSE _srcs 
    Interfaces/*.cpp
    IR/*.cpp
    Transforms/*.cpp
    Transforms/Patterns/*.cpp
    Interfaces/*.cpp)

add_mlir_library(XCMLIRTOP
    ${_srcs}
    DEPENDS
    MLIRTopOpsIncGen
    MLIRTopDialectPassesIncGen
    MLIRTopOpInterfacesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRParser
    MLIRFuncDialect
    MLIRRewrite
    MLIRTransforms
    MLIRSideEffectInterfaces
    XCUTILS
)