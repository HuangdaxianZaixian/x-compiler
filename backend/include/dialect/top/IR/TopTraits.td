#ifndef TOP_TRAITS__TD
#define TOP_TRAITS__TD

include "mlir/IR/OpBase.td"

def ShardingTrait : NativeAttrTrait<"ShardingTrait"> {
    let cppNamespace = "::xc::top";
}

def NPUTrait : NativeOpTrait<"NPUTrait"> {
    let cppNamespace = "::xc::top";
}

#endif // TOP_TRAITS__TD