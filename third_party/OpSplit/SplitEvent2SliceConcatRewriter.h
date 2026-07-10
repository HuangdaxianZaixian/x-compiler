#ifndef DIALECT_XRT_SPLIT_EVENT_2_SLICE_CONCAT_REWRITER
#define DIALECT_XRT_SPLIT_EVENT_2_SLICE_CONCAT_REWRITER

#include "Dialect/XRT/Rewrites/RewriteUtils.h"
#include "Helper/OpUtils.h"

namespace xp_mlir {
namespace xrt {
struct SplitEvent2SliceConcatRewriter : public mlir::RewritePattern {
    SplitEvent2SliceConcatRewriter(mlir::MLIRContext *context)
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