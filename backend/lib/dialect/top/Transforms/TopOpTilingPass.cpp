#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "dialect/top/Transforms/TopDialectPasses.hpp"
#include "dialect/top/Transforms/Patterns/TopOpTilingRewritter.hpp"

#include "utils/com_utils.hpp"

namespace xc {
namespace top {

struct TopOpTilingPass : public top::TopOpTilingPassBase<TopOpTilingPass> {

  void runOnOperation() override {
        auto mOp = getOperation();
        auto ctx = mOp.getContext();

        GreedyRewriteConfig config;
        config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
        config.setMaxIterations(1);  // use no greedy mode for insert
        RewritePatternSet patterns(ctx);

        patterns.clear();
        patterns.add<TopOpTilingRewritter>(ctx);
        (void)applyPatternsGreedily(mOp, std::move(patterns), config);
        xc::utils::moduleDump(mOp, "top-op-tiling.mlir", true);
    }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createTopOpTilingPass() {
    return std::make_unique<TopOpTilingPass>();
}

} // namespace top
} // namespace xc

