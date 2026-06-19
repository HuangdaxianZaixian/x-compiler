// XVectorTypes.td
#ifndef X_VECTOR_TYPES__TD
#define X_VECTOR_TYPES__TD

include "mlir/IR/AttrTypeBase.td"
include "XVectorDialect.td"
include "XVectorAttrs.td"

class XVector_BaseType<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<XVector_Dialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

def ScalarRegType : XVector_BaseType<"ScalarReg", "ScalarReg"> {
  let summary = "scalar register";
  let description = [{
    "vector core scalar register"
  }];

  let parameters = (ins
      "int32_t":$id,
      "RvvDtype":$dtype
  );

  let assemblyFormat = "`<` `xreg` `=` $id `,` `dtype` `=` $dtype `>`";
}

def VectorRegType : XVector_BaseType<"VectorReg", "VectorReg"> {
  let summary = "vector register";
  let description = [{
    "vector core vector register"
  }];

  let parameters = (ins
      "int32_t":$id,
      "RvvDtype":$dtype
  );

  let assemblyFormat = "`<` `vreg` `=` $id `,` `dtype` `=` $dtype `>`";
}

def RvvImmType : XVector_BaseType<"RvvImm", "RvvImm"> {
  let summary = "rvv imm";
  let description = [{
    "rvv imm"
  }];

  let parameters = (ins);
}

#endif // X_VECTOR_TYPES__TD