#ifndef TOP_OP_TILING_REWRITTER__HPP
#define TOP_OP_TILING_REWRITTER__HPP

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace xc {
namespace top {
struct TopOpTilingRewritter : public mlir::RewritePattern {
    TopOpTilingRewritter(mlir::MLIRContext *context)
      : mlir::RewritePattern(MatchAnyOpTypeTag(), 1, context, {}) {}

public:
  mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
                                mlir::PatternRewriter &rewriter) const override;
};
} // namespace top
} // namespace xc

#endif