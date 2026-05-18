
#ifndef OP_INTERFACES__TD
#define OP_INTERFACES__TD

include "mlir/IR/OpBase.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];
  let cppNamespace = "::xc::top";

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}

#endif // OP_INTERFACES__TD
