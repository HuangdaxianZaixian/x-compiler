#include "xp_mlir/Dialect/XMA/Vector/XVectorOps.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/OpImplementation.h"

#include "xp_mlir/Dialect/XMA/Vector/XVectorDialect.cpp.inc"
#include "xp_mlir/Dialect/XMA/Vector/XVectorEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "xp_mlir/Dialect/XMA/Vector/XVectorAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "xp_mlir/Dialect/XMA/Vector/XVectorTypes.cpp.inc"

#define GET_OP_CLASSES
#include "xp_mlir/Dialect/XMA/Vector/XVectorOps.cpp.inc"

namespace xvec {

void XVectorDialect::initialize() {
       addAttributes<
                #define GET_ATTRDEF_LIST
                #include "xp_mlir/Dialect/XMA/Vector/XVectorAttrs.cpp.inc"
               >();
       addTypes<
                #define GET_TYPEDEF_LIST
                #include "xp_mlir/Dialect/XMA/Vector/XVectorTypes.cpp.inc"
               >();
       addOperations<
                #define GET_OP_LIST
                #include "xp_mlir/Dialect/XMA/Vector/XVectorOps.cpp.inc"
               >();
}

} // namespace xvec