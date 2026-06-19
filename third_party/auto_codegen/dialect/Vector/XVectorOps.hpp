#ifndef X_VECTOR_OP__HPP
#define X_VECTOR_OP__HPP

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;

#include "xp_mlir/Dialect/XMA/Vector/XVectorDialect.h.inc"
#include "xp_mlir/Dialect/XMA/Vector/XVectorEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "xp_mlir/Dialect/XMA/Vector/XVectorAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "xp_mlir/Dialect/XMA/Vector/XVectorTypes.h.inc"

#define GET_OP_CLASSES
#include "xp_mlir/Dialect/XMA/Vector/XVectorOps.h.inc"

#endif // X_VECTOR_OP__HPP