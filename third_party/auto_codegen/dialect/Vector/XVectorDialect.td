#ifndef X_VECTOR_DIALECT__TD
#define X_VECTOR_DIALECT__TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def XVector_Dialect : Dialect {
  let name = "xvec";
  let cppNamespace = "::xvec";

  let summary = [{
    XVector dialect
  }];

  let description = [{
    the bridge dialect between xma and rvv
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // X_VECTOR_DIALECT__TD