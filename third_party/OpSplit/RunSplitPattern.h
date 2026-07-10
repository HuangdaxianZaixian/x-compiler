#ifndef RUN_SPLIT_PATTERN__H
#define RUN_SPLIT_PATTERN__H

#include "LabelOpSplitDim.h"
#include "mlir/IR/BuiltinOps.h"

/// TODO
/// 1. 如果regionOp的输出被多个user使用, 有些user需要切分, 有些不需要切分

namespace xp_mlir {
namespace xrt {

using SplitPatternFunc = std::function<void(mlir::ModuleOp)>;

class RunSplitPattern {
public:
    RunSplitPattern();
    void run(mlir::ModuleOp module_op);

private:
    void registerPatterns();
    void runPattern(mlir::ModuleOp module_op, const std::string& pattern_name, SplitPatternFunc pattern_func);

private:
    std::map<std::string, SplitPatternFunc> pattern_funcs_;
};
    
}
}

#endif // RUN_SPLIT_PATTERN__H