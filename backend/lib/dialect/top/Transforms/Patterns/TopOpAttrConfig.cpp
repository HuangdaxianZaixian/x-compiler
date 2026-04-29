#include "TopOpAttrConfig.hpp"

namespace xc {
namespace top {

mlir::LogicalResult TopOpAttrConfig::matchAndRewrite(mlir::Operation *op,
                                 mlir::PatternRewriter &rewriter) const {
  llvm::outs() << "sim op code = " << op->getName().getStringRef() << "\n";
  if (op) {
    op->setAttr("do_relu", rewriter.getBoolAttr(true));
    return mlir::success();
  }
  
  return mlir::failure();
}

}
}