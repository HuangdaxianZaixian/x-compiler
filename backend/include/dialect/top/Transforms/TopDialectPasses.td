#ifndef TOP_OP_CHECK_PASS__TD
#define TOP_OP_CHECK_PASS__TD

include "mlir/Pass/PassBase.td"

def TopOpCheckPass : Pass<"top-op-check", "ModuleOp"> {
    let summary = "check top op";
    let constructor = "createTopOpCheckPass()";
    let dependentDialects = ["xc::top::TopDialect"];
}

def TopOpTilingPass : Pass<"tiling", "ModuleOp"> {
    let summary = "op tiling";
    let constructor = "createTopOpTilingPass()";
    let dependentDialects = ["xc::top::TopDialect", "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect"];
}

#endif // TOP_OP_CHECK_PASS__TD