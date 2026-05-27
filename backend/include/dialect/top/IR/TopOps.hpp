#ifndef TOP_OP__HPP
#define TOP_OP__HPP

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;

#include "dialect/top/IR/TopTraits.hpp"

#include "dialect/top/IR/TopDialect.h.inc"

#include "dialect/top/Interfaces/TopInterfaces.h.inc"

#define GET_TYPEDEF_CLASSES
#include "dialect/top/IR/TopTypes.h.inc"

#define GET_OP_CLASSES
#include "dialect/top/IR/TopOps.h.inc"

#define GET_ATTRDEF_CLASSES
#include "dialect/top/IR/TopAttrs.h.inc"

#endif // TOP_OP__HPP