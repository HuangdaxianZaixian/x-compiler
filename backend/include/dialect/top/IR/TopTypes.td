// TopTypes.td
#ifndef TOP_TYPES__TD
#define TOP_TYPES__TD

include "mlir/IR/AttrTypeBase.td"
include "TopDialect.td"

class Top_BaseType<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Top_Dialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

def OpIndexType : Top_BaseType<"OpIndex", "OpIndex"> {
  let summary = "op index";
  let description = [{
    "op index"
  }];

  let parameters = (ins
      "int64_t":$id
  );

  let assemblyFormat = "$id";

  let genVerifyDecl = 1;
}

#endif // TOP_TYPES__TD