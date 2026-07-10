#include "SplitRegionOpRewriter.h"
#include "Helper/OpUtils.h"
#include "OpSplitUtils.h"
#include "llvm/IR/Function.h"
#include "xp_mlir/Dialect/XFR/IR/XFROps.h"

/*
 * 1. 在LableOpSplitDim中已经保证对于ofm split event valid的TMUfuse op, 它region内的op,
 *    除了returnOp, concatOp, 其它op的ofm split event都是valid, 也就是都是需要拆分的
 * 2. 将region op clone 成splitNum份, 每份内只保留对于split_indexd的op, 其它op都删除掉
 * 3. 对return op进行特殊处理
 * 4. 对TMUfuse op的user(slice)进行处理
 */

namespace xp_mlir {
namespace xrt {

LogicalResult SplitRegionOpRewriter::match(Operation *op) const {
  return success();
}

void SplitRegionOpRewriter::rewrite(Operation *op,
                           PatternRewriter &rewriter) const {
  if (!op) return;
  if (!isOpWithRegion(op)) return;
  if (isOpSplited(op)) return;
  if (ifOpHasSplitIndex(op)) return;

  if (isOpSliced(op) && isOpLabeled(op) && getTensorSplitEvent(op->getResult(0)).isValid()) {
      splitRegionOp(op, rewriter);
  }                      
}

} // namespace xrt
} // namespace xp_mlir

