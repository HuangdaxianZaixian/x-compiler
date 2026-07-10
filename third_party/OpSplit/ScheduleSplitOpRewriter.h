#ifndef SCHEDULE_SPLIT_OP__H
#define SCHEDULE_SPLIT_OP__H

#include "OpSplitUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "Dialect/XRT/Rewrites/RewriteUtils.h"
#include "Helper/OpUtils.h"

namespace xp_mlir {
namespace xrt {
struct ScheduleSplitOpRewriter : public mlir::RewritePattern {
  ScheduleSplitOpRewriter(mlir::MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context, {}) {}

public:
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (failed(match(op))) return failure();
    return rewrite(op, rewriter);
  }

  LogicalResult rewrite(Operation *op, PatternRewriter &rewriter) const;
  LogicalResult match(Operation *op) const;
};
    
}
}

#endif // SCHEDULE_SPLIT_OP__H