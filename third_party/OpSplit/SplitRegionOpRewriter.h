#ifndef DIALECT_XRT_SPLIT_REGION_OP_REWRITER
#define DIALECT_XRT_SPLIT_REGION_OP_REWRITER

#include "Dialect/XRT/Rewrites/RewriteUtils.h"
#include "Helper/OpUtils.h"

namespace xp_mlir {
namespace xrt {
struct SplitRegionOpRewriter : public mlir::RewritePattern {
  SplitRegionOpRewriter(mlir::MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context, {}) {}

public:
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (failed(match(op))) return failure();
    rewrite(op, rewriter);
    return success();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const;
  LogicalResult match(Operation *op) const;
};

} // namespace xrt
} // namespace xp_mlir

#endif