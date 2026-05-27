#ifndef TOP_OPS__TD
#define TOP_OPS__TD

include "TopDialect.td"
include "TopAttrs.td"
include "TopTraits.td"
include "TopTypes.td"
include "dialect/top/Interfaces/OpInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"

class Top_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<Top_Dialect, mnemonic, traits>;

def Top_ConstantOp : Top_BaseOp<"constant"> {
  let summary = "constant operation";
  let description = [{
    constant op
  }];

  let arguments = (ins);

  let results = (outs AnyRankedTensor:$output);
}

def Top_AddOp : Top_BaseOp<"add", [NPUTrait, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>, DeclareOpInterfaceMethods<TilingInterface,
          ["getIterationDomain",
           "getLoopIteratorTypes",
           "getTiledImplementation",
           "getResultTilePosition"]>]> {
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

def Top_MatmulOp : Top_BaseOp<"matmul", [NPUTrait, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>, DeclareOpInterfaceMethods<TilingInterface,
        ["getIterationDomain",
         "getLoopIteratorTypes",
         "getTiledImplementation",
         "getResultTilePosition"]>]> {
  let summary = "matmul operation";
  let description = [{
    matmul(a, b) = a x b
  }];

  let arguments = (ins
    AnyRankedTensor:$ifm,
    AnyRankedTensor:$wgts
  );

  let results = (outs AnyRankedTensor:$output);

  let hasVerifier = 1;
}

#endif // TOP_OPS__TD