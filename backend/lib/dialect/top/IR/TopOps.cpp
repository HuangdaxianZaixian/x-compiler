#include "dialect/top/IR/TopOps.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/OpImplementation.h"

#include "dialect/top/IR/TopDialect.cpp.inc"

#define GET_OP_CLASSES
#include "dialect/top/IR/TopOps.cpp.inc"

namespace xc {
namespace top {

void TopDialect::initialize() {
  addOperations<
                #define GET_OP_LIST
                #include "dialect/top/IR/TopOps.cpp.inc"
               >();
}

mlir::LogicalResult AddOp::verify() {
  return mlir::success();
}

} // namespace top
} // namespace xc