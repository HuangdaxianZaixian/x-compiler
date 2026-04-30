#ifndef TOP_DIALECT__TD
#define TOP_DIALECT__TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Top_Dialect : Dialect {
  let name = "top";
  let cppNamespace = "::xc::top";

  let summary = [{
    top dialect
  }];

  let description = [{
    top dialect is a high-level dialect
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // TOP_DIALECT__TD