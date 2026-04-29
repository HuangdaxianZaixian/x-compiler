#ifndef TOP_OPS__TD
#define TOP_OPS__TD

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
}

class Top_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<Top_Dialect, mnemonic, traits>;

def Top_AddOp : Top_BaseOp<"add", [Pure]> {
  let summary = "add operation";
  let description = [{
    Add(a, b) = a + b
  }];

  let arguments = (ins
    AnyRankedTensor:$lft,
    AnyRankedTensor:$rht
  );

  let results = (outs AnyRankedTensor:$output);

  let hasVerifier = 1;
}

#endif // TOP_OPS__TD