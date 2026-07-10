#include "FolderConstantSliceRewriter.h"
#include "Helper/OpUtils.h"
#include "OpSplitUtils.h"
#include "llvm/IR/Function.h"
#include "xp_mlir/Dialect/XFR/IR/XFROps.h"

namespace xp_mlir {
namespace xrt {

LogicalResult FolderConstantSliceRewriter::match(Operation *op) const {
  return success();
}

void FolderConstantSliceRewriter::rewrite(Operation *op,
                           PatternRewriter &rewriter) const {
  if (!op) return;

  if (llvm::isa<xfr::XFRSliceOp>(op) && ifOpHasSplitIndex(op)) {
    foldConstantSlice(op, rewriter);
  }                      
}

} // namespace xrt
} // namespace xp_mlir

