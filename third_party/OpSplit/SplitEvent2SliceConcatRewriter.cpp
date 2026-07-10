#include "SplitEvent2SliceConcatRewriter.h"
#include "OpSplitUtils.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/LogicalResult.h"
#include "xp_mlir/Dialect/XFR/IR/XFROps.h"

namespace xp_mlir {
namespace xrt {

LogicalResult SplitEvent2SliceConcatRewriter::match(Operation *op) const {
  return llvm::success();
}

void SplitEvent2SliceConcatRewriter::rewrite(Operation *op,
                           PatternRewriter &rewriter) const {
  if (!op) return;

  if (isOpSliced(op)) return;

  if (ifOpHasSplitIndex(op)) return;

  if (isInsertedSliceConcat(op)) return;
  
  // op ofm with valid split event
  if (op && op->getNumResults() == 1) {
      auto ofm = op->getResult(0);
      if (ofm && !isNone(ofm) && getTensorSplitEvent(ofm).isValid()) {
        createSplitValueSliceAndConcat(ofm, rewriter);
        setOpSliced(op);
      }
  }

  // funcOp argument with valid split event
  if (auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
      for (auto arg : funcOp.getArguments()) {
          if (arg && !isNone(arg)) {
              auto alias_op = getAliasOp(arg);
              if (getTensorSplitEvent(arg).isValid() || (alias_op && getTensorSplitEvent(alias_op->getResult(0)).isValid())) {
                createSplitValueSliceAndConcat(arg, rewriter);
                setOpSliced(op);
              }
          }
      }
  }                          
}

} // namespace xrt
} // namespace xp_mlir

