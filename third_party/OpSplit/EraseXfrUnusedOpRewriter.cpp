#include "EraseXfrUnusedOpRewriter.h"
#include "OpSplitUtils.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "xp_mlir/Dialect/XFR/IR/XFROps.h"

namespace xp_mlir {
namespace xrt {

LogicalResult EraseXfrUnusedOpRewriter::match(Operation *op) const {
  return success();
}

LogicalResult EraseXfrUnusedOpRewriter::rewrite(Operation *op,
                           PatternRewriter &rewriter) const {
  if (!op) return failure();
  if (llvm::isa<ModuleOp, mlir::func::FuncOp, xfr::XFRReturnOp, mlir::func::ReturnOp>(op)) {
    return failure();
  }

  if (0 == frontendMlir::getUserNum(op)) {
    rewriter.eraseOp(op);
    return llvm::success();
  }  
  
  return failure();
}

} // namespace xrt
} // namespace xp_mlir

