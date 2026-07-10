#include "ScheduleSplitOpRewriter.h"
#include "OpSplitUtils.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "xp_mlir/Dialect/XFR/IR/XFROps.h"

namespace xp_mlir {
namespace xrt {

LogicalResult ScheduleSplitOpRewriter::match(Operation *op) const {
  return success();
}

LogicalResult ScheduleSplitOpRewriter::rewrite(Operation *op,
                           PatternRewriter &rewriter) const {
  if (!op) return llvm::failure();
  if (!ifOpHasSplitIndex(op)) return llvm::failure();
  if (isOpInRegionOp(op)) return llvm::failure();
  if (isOpMoved(op)) return llvm::failure();
  
  auto dependent_values = getOpDependentValues(op);
  auto last_dependent_value = getLatestDefinedValueInModule(dependent_values);
  if (!isBlockArgument(last_dependent_value)) {
    auto val_def_op = last_dependent_value.getDefiningOp();
    assert(val_def_op && "the latest defined value should have defining op");
    op->moveAfter(val_def_op);
    setOpMoved(op);
    return llvm::success();
  }

  return llvm::failure();
}

} // namespace xrt
} // namespace xp_mlir