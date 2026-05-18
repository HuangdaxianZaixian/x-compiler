#include "dialect/top/IR/TopOps.hpp"

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

#include "dialect/top/IR/TopDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "dialect/top/IR/TopTypes.cpp.inc"

#define GET_OP_CLASSES
#include "dialect/top/IR/TopOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "dialect/top/IR/TopAttrs.cpp.inc"

#define DEBUG_TYPE "op-verify"

namespace xc {
namespace top {

void TopDialect::initialize() {
       addTypes<
                #define GET_TYPEDEF_LIST
                #include "dialect/top/IR/TopTypes.cpp.inc"
               >();
  addAttributes<
                #define GET_ATTRDEF_LIST
                #include "dialect/top/IR/TopAttrs.cpp.inc"
               >();
  addOperations<
                #define GET_OP_LIST
                #include "dialect/top/IR/TopOps.cpp.inc"
               >();
}

mlir::LogicalResult AddOp::verify() {
  LLVM_DEBUG(llvm::dbgs() << "AddOp verify" << "\n");
  return mlir::success();
}

mlir::LogicalResult MatmulOp::verify() {
  llvm::outs() << "llvm::DebugFlag = " << llvm::DebugFlag << "\n";
  llvm::dbgs() << "MatmulOp verify" << "\n";
  return mlir::success();
}

::llvm::LogicalResult shardingAttr::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int64_t num, ::llvm::ArrayRef<int64_t> shards) {
  return mlir::success();
}

LogicalResult OpIndexType::verify(function_ref<InFlightDiagnostic()> emitError, int64_t id) {
  if (id < 0) {
    emitError() << "id must be non-negative";
    return mlir::failure();
  }
  return mlir::success();
}

} // namespace top
} // namespace xc