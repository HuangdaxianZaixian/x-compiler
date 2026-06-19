#ifndef X_VECTOR_OPS__TD
#define X_VECTOR_OPS__TD

include "XVectorDialect.td"
include "XVectorTypes.td"
include "XVectorAttrs.td"


/*
 * 指令支持三种数据输入: imm, xreg, vreg
 * imm: ImmOp产生
 * xreg: InitXregOp, ScalarLoadOp产生
 * vreg: VectorLoadOp, 其它计算op产生
 */

class XVector_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<XVector_Dialect, mnemonic, traits>;

// 为支持imm的指令生成输入
def XVector_ImmOp : XVector_BaseOp<"imm", []> {
  let summary = "create imm";
  let description = [{
    create imm for rvv instruction
  }];

  let arguments = (ins
    I32Attr:$val
  );

  let results = (outs RvvImmType:$output);
}

// 用常量或变量初始值初始化scalar register
def XVector_InitXregOp : XVector_BaseOp<"initXreg", []> {
  let summary = "init x reg op";
  let description = [{
    init x reg with const or initial value of variable
  }];

  let arguments = (ins
    I32Attr:$val
  );

  let results = (outs ScalarRegType:$output);
}

// 输入数据类型编码在输出类型中
def XVector_ScalarLoadOp : XVector_BaseOp<"scalarLoad", []> {
  let summary = "load scalar from L1";
  let description = [{
    load(L1_addr)
  }];

  let arguments = (ins
    ScalarRegType:$addr
  );

  let results = (outs ScalarRegType:$output);
}

// 输入数据类型编码在输出类型中
def XVector_VectorLoadOp : XVector_BaseOp<"vectorLoad", []> {
  let summary = "load vector from L1";
  let description = [{
    load(L1_addr)
  }];

  let arguments = (ins
    ScalarRegType:$addr
  );

  let results = (outs VectorRegType:$output);
}

def XVector_StoreOp : XVector_BaseOp<"store", []> {
  let summary = "L1 store operation";
  let description = [{
    store(L1_addr, VREG)
  }];

  let arguments = (ins
    VectorRegType:$src,
    ScalarRegType:$addr
  );

  let results = (outs);
}

def XVector_ReciprocalOp : XVector_BaseOp<"reciprocal", []> {
  let summary = "reciprocal";
  let description = [{
    reciprocal(x, scalar) = scalar / x 
  }];

  let arguments = (ins
    VectorRegType:$input,
    ScalarRegType:$scalar
  );

  let results = (outs VectorRegType:$output);
}

def XVector_Xadd : XVector_BaseOp<"xadd", []> {
  let summary = "scalar add";
  let description = [{
    add(x, y) = x + y, y can be imm[0:11] or xreg 
  }];

  let arguments = (ins
    ScalarRegType:$x,
    AnyTypeOf<[ScalarRegType, RvvImmType]>:$y
  );

  let results = (outs ScalarRegType:$output);
}

def XVector_Xmul : XVector_BaseOp<"xmul", []> {
  let summary = "scalar mul";
  let description = [{
    mul(x, y) = x * y
  }];

  let arguments = (ins
    ScalarRegType:$x,
    ScalarRegType:$y
  );

  let results = (outs ScalarRegType:$output);
}

def XVector_Xdiv : XVector_BaseOp<"xdiv", []> {
  let summary = "scalar div";
  let description = [{
    div(x, scalar) = x / scalar 
  }];

  let arguments = (ins
    ScalarRegType:$x,
    ScalarRegType:$scalar
  );

  let results = (outs ScalarRegType:$output);
}

def XVector_Xmod : XVector_BaseOp<"xmod", []> {
  let summary = "scalar mod";
  let description = [{
    mod(x, scalar) = x % scalar 
  }];

  let arguments = (ins
    ScalarRegType:$x,
    ScalarRegType:$n,
    ScalarRegType:$scalar
  );

  let results = (outs ScalarRegType:$output);
}

def XVector_Xincrement : XVector_BaseOp<"xincrement", []> {
  let summary = "scalar increment op";
  let description = [{
    xincrement(x) = ++x 
  }];

  let arguments = (ins
    ScalarRegType:$x
  );

  let results = (outs ScalarRegType:$output);
}

def XVector_SelectScalarFromVectorAndBroadcast : XVector_BaseOp<"selectScalarFromVectorAndBroadcast", []> {
  let summary = "seletc scalar from vector and broadcast to vector";
  let description = [{
    selectScalarFromVectorAndBroadcast(vs0, index)
  }];

  let arguments = (ins
    VectorRegType:$input,
    ScalarRegType:$index
  );

  let results = (outs VectorRegType:$output);
}

def XVector_Extend : XVector_BaseOp<"extend", []> {
  let summary = "vector type extend op";
  let description = [{
    extend(vs0)
  }];

  let arguments = (ins
    VectorRegType:$input
  );

  let results = (outs VectorRegType:$output);
}

def XVector_Vadd : XVector_BaseOp<"vadd", []> {
  let summary = "vector add";
  let description = [{
    add(x, y) = x + y, y can be xreg or vreg 
  }];

  let arguments = (ins
    VectorRegType:$x,
    AnyTypeOf<[VectorRegType, ScalarRegType, RvvImmType]>:$y
  );

  let results = (outs VectorRegType:$output);
}

def XVector_Vsub : XVector_BaseOp<"vsub", []> {
  let summary = "vector sub";
  let description = [{
    sub(x, y) = x - y, y can be xreg or vreg 
  }];

  let arguments = (ins
    VectorRegType:$x,
    AnyTypeOf<[VectorRegType, ScalarRegType]>:$y
  );

  let results = (outs VectorRegType:$output);
}

def XVector_Vmul : XVector_BaseOp<"vmul", []> {
  let summary = "vector mul";
  let description = [{
    mul(x, y) = x * y, y can be xreg or vreg 
  }];

  let arguments = (ins
    VectorRegType:$x,
    AnyTypeOf<[VectorRegType, ScalarRegType]>:$y
  );

  let results = (outs VectorRegType:$output);
}

def XVector_Vdiv : XVector_BaseOp<"vdiv", []> {
  let summary = "vector div";
  let description = [{
    div(x, y) = x / y, y can be xreg or vreg 
  }];

  let arguments = (ins
    VectorRegType:$x,
    AnyTypeOf<[VectorRegType, ScalarRegType]>:$y
  );

  let results = (outs VectorRegType:$output);
}

def XVector_Vmax : XVector_BaseOp<"vmax", []> {
  let summary = "vector max";
  let description = [{
    max(x, scalar)
  }];

  let arguments = (ins
    VectorRegType:$x,
    ScalarRegType:$scalar
  );

  let results = (outs VectorRegType:$output);
}

def XVector_Vmin : XVector_BaseOp<"vmin", []> {
  let summary = "vector min";
  let description = [{
    min(x, scalar)
  }];

  let arguments = (ins
    VectorRegType:$x,
    ScalarRegType:$scalar
  );

  let results = (outs VectorRegType:$output);
}

def XVector_Vabs : XVector_BaseOp<"vabs", []> {
  let summary = "vector abs";
  let description = [{
    abs(x)
  }];

  let arguments = (ins
    VectorRegType:$x
  );

  let results = (outs VectorRegType:$output);
}

def XVector_Vquant : XVector_BaseOp<"vquant", []> {
  let summary = "vector quant";
  let description = [{
    quant(x, param), param can be xreg or vreg 
  }];

  let arguments = (ins
    VectorRegType:$x,
    AnyTypeOf<[VectorRegType, ScalarRegType]>:$param
  );

  let results = (outs VectorRegType:$output);
}

def XVector_Vqlut : XVector_BaseOp<"vqlut", []> {
  let summary = "vector quant-lut op";
  let description = [{
    qlut(x, param, table), param can be xreg or vreg
  }];

  let arguments = (ins
    VectorRegType:$x,
    AnyTypeOf<[VectorRegType, ScalarRegType]>:$param,
    ScalarRegType:$table,
    I32Attr:$table_size
  );

  let results = (outs VectorRegType:$output);
}

def XVector_RvvConfig : XVector_BaseOp<"rvvConfig", []> {
  let summary = "rvv config op";
  let description = [{
    rvvConfig(vl_return, vl_set)
  }];

  let arguments = (ins
    ScalarRegType:$vl_set
  );

  let results = (outs ScalarRegType:$output);
}

def XVector_LoopBegin : XVector_BaseOp<"loopBegin", []> {
  let summary = "rvv loop begin";
  let description = [{
    loopBegin(loop_times, loop_id)
  }];

  let arguments = (ins
    ScalarRegType:$loop_times,
    I32Attr:$loop_id
  );

  let results = (outs);
}

def XVector_LoopEnd : XVector_BaseOp<"loopEnd", []> {
  let summary = "rvv loop end";
  let description = [{
    loopEnd(loop_id)
  }];

  let arguments = (ins
    I32Attr:$loop_id
  );

  let results = (outs);
}

#endif // X_VECTOR_OPS__TD