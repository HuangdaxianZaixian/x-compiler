#ifndef LABLE_OP_SPLIT_DIM__H
#define LABLE_OP_SPLIT_DIM__H

#include "OpSplitUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"


namespace xp_mlir {
namespace xrt {

using SplitEventForwardFn = std::function<SplitEvent(mlir::Operation*, const std::vector<SplitEvent>&)>;
using SplitEventBackwardFn = std::function<std::vector<SplitEvent>(mlir::Operation*, const SplitEvent&)>;

enum class InferReturnType {
    DO_NOTHING = 0,
    INFER_FAILED = 1,
    INFER_SUCCESS = 2
};

class LabelOpSplitDim {
public:
    LabelOpSplitDim();
    void run(mlir::ModuleOp moduleOp);

private:
    void registerOp();
    SplitEvent opForward(mlir::Operation* op, InferReturnType& is_inferred);
    std::vector<SplitEvent> opBackward(mlir::Operation* op, InferReturnType& is_inferred);
    bool labelOp(mlir::Operation* op);
    void labelOps(mlir::ModuleOp moduleOp);
    void checkRegionOp(mlir::ModuleOp moduleOp);
    void moveConstOpFromRegionOp(mlir::ModuleOp moduleOp);
    void canonicalizeQuantOpParamShape(mlir::ModuleOp moduleOp);
    void canonicalizeEletwOpOperandShape(mlir::ModuleOp moduleOp);

private:
    std::map<std::string, SplitEventForwardFn> forwardFns_;
    std::map<std::string, SplitEventBackwardFn> backwardFns_;
};

} // namespace xrt
} // namespace xp_mlir

#endif // LABLE_OP_SPLIT_DIM__H
