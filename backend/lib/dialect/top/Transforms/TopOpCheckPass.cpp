#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "dialect/top/Transforms/TopDialectPasses.hpp"
#include "dialect/top/Transforms/Patterns/TopOpAttrConfig.hpp"

namespace xc {
namespace top {

struct TopOpCheckPass : public top::TopOpCheckPassBase<TopOpCheckPass> {

  void runOnOperation() override {
        auto mOp = getOperation();
        auto ctx = mOp.getContext();

        GreedyRewriteConfig config;
        config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
        config.setMaxIterations(1);  // use no greedy mode for insert
        RewritePatternSet patterns(ctx);

        patterns.clear();
        patterns.add<TopOpAttrConfig>(ctx);
        (void)applyPatternsGreedily(mOp, std::move(patterns), config);
    }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createTopOpCheckPass() {
    return std::make_unique<TopOpCheckPass>();
}

} // namespace top
} // namespace xc

