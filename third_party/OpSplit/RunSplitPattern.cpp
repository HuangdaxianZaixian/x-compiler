#include "RunSplitPattern.h"
#include "Dialect/XRT/Rewrites/OpSplit/ScheduleSplitOpRewriter.h"
#include "SplitPatterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/XRT/Rewrites/OpSplit/LabelOpSplitDim.h"
#include "Dialect/XRT/Rewrites/OpSplit/SplitEvent2SliceConcatRewriter.h"
#include "Dialect/XRT/Rewrites/OpSplit/SplitNoRegionOpRewriter.h"
#include "Dialect/XRT/Rewrites/OpSplit/SplitRegionOpRewriter.h"
#include "Dialect/XRT/Rewrites/OpSplit/EraseXfrUnusedOpRewriter.h"
#include "Dialect/XRT/Rewrites/OpSplit/OpSplitUtils.h"
#include "Dialect/XRT/Rewrites/OpSplit/FolderConstantSliceRewriter.h"
#include <string>

namespace xp_mlir {
namespace xrt {

RunSplitPattern::RunSplitPattern() {
    registerPatterns();
}

void RunSplitPattern::run(mlir::ModuleOp module_op) {
    for (auto &pattern_func : pattern_funcs_) {
        runPattern(module_op, pattern_func.first, pattern_func.second);
    }
}

void RunSplitPattern::registerPatterns() {
    // pattern_funcs_["pattern_matmul_k6144_n1536"] = pattern_matmul_k6144_n1536;
    // pattern_funcs_["pattern_matmul_k1536_n6144"] = pattern_matmul_k1536_n6144;
    // pattern_funcs_["pattern_pertoken"] = pattern_pertoken;
    // pattern_funcs_["pattern_softmax"] = pattern_softmax;
    pattern_funcs_["pattern_moe_0"] = pattern_moe_0;
}

std::string mlirFileNameWithIndex(const std::string& base_name, int index) {
    return base_name + "_" + std::to_string(index) + ".mlir";
}

void RunSplitPattern::runPattern(mlir::ModuleOp module_op, const std::string& pattern_name, SplitPatternFunc pattern_func) {
    static int pattern_index = 0;
    std::cout << "running split pattern " + std::to_string(pattern_index) + ": " << pattern_name << std::endl;
    pattern_func(module_op);

    auto ctx = module_op.getContext();

    (void)frontendMlir::modulePrinter(module_op, mlirFileNameWithIndex("xrt_before_split_op", pattern_index), false);

    auto lable_split_op_dim = std::make_unique<LabelOpSplitDim>();
    lable_split_op_dim->run(module_op);
    (void)frontendMlir::modulePrinter(module_op, mlirFileNameWithIndex("xrt_label_op_split_dim", pattern_index),false);

    unsetModuleAllOpsAttr(module_op, "split_target");

    mlir::GreedyRewriteConfig config;
    config.setUseTopDownTraversal(false);
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    config.setMaxIterations(1);  // use no greedy mode for insert
    RewritePatternSet patterns(ctx);

    patterns.clear();
    patterns.add<SplitEvent2SliceConcatRewriter>(ctx);
    (void)applyPatternsGreedily(module_op, std::move(patterns), config);
    (void)frontendMlir::modulePrinter(module_op, mlirFileNameWithIndex("xrt_split_event_2_slice_concat", pattern_index), false);

    patterns.clear();
    patterns.add<SplitNoRegionOpRewriter>(ctx);
    (void)applyPatternsGreedily(module_op, std::move(patterns), config);
    (void)frontendMlir::modulePrinter(module_op, mlirFileNameWithIndex("xrt_split_no_region_op", pattern_index), false);

    config.setMaxIterations(10);
    patterns.clear();
    patterns.add<EraseXfrUnusedOpRewriter>(ctx);
    (void)applyPatternsGreedily(module_op, std::move(patterns), config);
    (void)frontendMlir::modulePrinter(module_op, mlirFileNameWithIndex("xrt_erase_xfr_unused_op_0", pattern_index), false);
    config.setMaxIterations(1);

    patterns.clear();
    // 存在多层region嵌套的情况, 需要优先处理子region
    config.setUseTopDownTraversal(false);
    patterns.add<SplitRegionOpRewriter>(ctx);
    (void)applyPatternsGreedily(module_op, std::move(patterns), config);
    (void)frontendMlir::modulePrinter(module_op, mlirFileNameWithIndex("xrt_split_region_op", pattern_index), false);

    patterns.clear();
    patterns.add<FolderConstantSliceRewriter>(ctx);
    (void)applyPatternsGreedily(module_op, std::move(patterns), config);
    (void)frontendMlir::modulePrinter(module_op, mlirFileNameWithIndex("xrt_fold_const_slice", pattern_index), false);

    config.setMaxIterations(10);
    patterns.clear();
    patterns.add<EraseXfrUnusedOpRewriter>(ctx);
    (void)applyPatternsGreedily(module_op, std::move(patterns), config);
    (void)frontendMlir::modulePrinter(module_op, mlirFileNameWithIndex("xrt_erase_xfr_unused_op_1", pattern_index),  false);
    config.setMaxIterations(1);

    patterns.clear();
    // 需要按照从上到下的顺序遍历op, 这样可以保证op一次移动到位
    config.setUseTopDownTraversal(true);
    patterns.add<ScheduleSplitOpRewriter>(ctx);
    (void)applyPatternsGreedily(module_op, std::move(patterns), config);
    (void)frontendMlir::modulePrinter(module_op, mlirFileNameWithIndex("xrt_schedule_split_op", pattern_index), false);

    unsetModuleAllOpsAttr(module_op, "is_labeled");
    unsetModuleAllOpsAttr(module_op, "is_sliced");
    unsetModuleAllOpsAttr(module_op, "is_splited");
    unsetModuleAllOpsAttr(module_op, "split_index");
    unsetModuleAllOpsAttr(module_op, "is_moved");
    eraseModuleAllValuesSplitInfo(module_op);

    (void)frontendMlir::modulePrinter(module_op, mlirFileNameWithIndex("xrt_after_split_op", pattern_index), false);

    ++pattern_index;
}

}
}