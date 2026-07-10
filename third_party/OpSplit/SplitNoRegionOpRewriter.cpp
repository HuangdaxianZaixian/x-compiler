#include "SplitNoRegionOpRewriter.h"
#include "OpSplitUtils.h"
#include "llvm/IR/Function.h"
#include "xp_mlir/Dialect/XFR/IR/XFROps.h"

/*
 * 1. 如果is_labeled op的ofm的split event是valid, 且op不是const op
 * 2. 将op拆分成splitNum个子op:
 *      a. 如果op的输入需要split, 则子op对应输入取concat的对应输入; 如果输入不需要拆分, 则子op对应输入保持不变
 *      b. 更新ofm对应的slice op的user的input(包括concat和已经split的子op)为对应的子op的ofm输出
 * 
 * 这样处理的结果是:
        a. 消除了输入的concat op
        b. 消除了输出的slice op
 * 最终实现将所有插入的slice op和concat op消除掉, 以达到真正的拆分输入和输出的目的
 */

namespace xp_mlir {
namespace xrt {

LogicalResult SplitNoRegionOpRewriter::match(Operation *op) const {
  return success();
}

void SplitNoRegionOpRewriter::rewrite(Operation *op,
                           PatternRewriter &rewriter) const {
  if (!op) return;
  if (isOpSplited(op)) return;
  if (isOpWithRegion(op)) return;
  if (llvm::isa<xfr::XFRConstantOp>(op)) return;
  if (ifOpHasSplitIndex(op)) return;

  // op ofm with valid split event
  if (op && op->getNumResults() == 1 && isOpSliced(op) && isOpLabeled(op)) {
      auto ofm = op->getResult(0);
      if (ofm && !isNone(ofm) && getTensorSplitEvent(ofm).isValid()) {
        splitNoRegionOp(op, rewriter);
      }
  }                       
}

} // namespace xrt
} // namespace xp_mlir

