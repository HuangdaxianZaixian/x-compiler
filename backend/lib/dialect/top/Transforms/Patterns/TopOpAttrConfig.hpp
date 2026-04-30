#ifndef TOP_OP_ATTR_CONFIG__HPP
#define TOP_OP_ATTR_CONFIG__HPP

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace xc {
namespace top {
struct TopOpAttrConfig : public mlir::RewritePattern {
    TopOpAttrConfig(mlir::MLIRContext *context)
      : mlir::RewritePattern(MatchAnyOpTypeTag(), 1, context, {}) {}

public:
  mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
                                mlir::PatternRewriter &rewriter) const override;
};
} // namespace top
} // namespace xc

#endif