#ifndef X_VECTOR_ATTRS__TD
#define X_VECTOR_ATTRS__TD

include "XVectorDialect.td"

class XVector_BaseAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<XVector_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def OpIndexAttr : XVector_BaseAttr<"OpIndex", "OpIndex"> {
  let summary = "op index attribute";
  let description = [{
    op index description
  }];

  let parameters = (ins
      "int32_t":$op_index
  );

  let assemblyFormat = "`<` `id` `=` $op_index `>`";
}

def RvvDtype_I8  : I32EnumAttrCase<"I8", 8, "I8">;
def RvvDtype_I10 : I32EnumAttrCase<"I10", 10, "I10">;
def RvvDtype_I11 : I32EnumAttrCase<"I11", 11, "I11">;
def RvvDtype_I12 : I32EnumAttrCase<"I12", 12, "I12">;
def RvvDtype_I13 : I32EnumAttrCase<"I13", 13, "I13">;
def RvvDtype_I14 : I32EnumAttrCase<"I14", 14, "I14">;
def RvvDtype_I16 : I32EnumAttrCase<"I16", 16, "I16">;
def RvvDtype_I32 : I32EnumAttrCase<"I32", 32, "I32">;

def XVector_RvvDtypeAttr : I32EnumAttr<
    "RvvDtype",  // c++ enum class name
    "rvv input/output data type",
    [
      RvvDtype_I8,
      RvvDtype_I10,
      RvvDtype_I11,
      RvvDtype_I12,
      RvvDtype_I13,
      RvvDtype_I14,
      RvvDtype_I16,
      RvvDtype_I32
    ]> {
  let cppNamespace = "::xvec";
  let underlyingType = "int32_t";
}

#endif // X_VECTOR_ATTRS__TD