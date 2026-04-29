#ifndef TOP_OP__HPP
#define TOP_OP__HPP

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;

#include "dialect/top/IR/TopDialect.h.inc"

#define GET_OP_CLASSES
#include "dialect/top/IR/TopOps.h.inc"

#endif // TOP_OP__HPP