#include "LabelOpSplitDim.h"
#include "Dialect/XRT/Rewrites/OpSplit/OpSplitUtils.h"
#include "Dialect/XRT/Rewrites/RewriteUtils.h"
#include "Helper/OpUtils.h"
#include "Helper/TensorUtils.h"
#include "Helper/XPMLIRUtils.h"
#include "Utils/mlirUtils.h"
#include "mlir/IR/Operation.h"
#include "xp_mlir/Dialect/XFR/IR/XFROps.h"
#include <algorithm>
#include <cassert>
#include <fcntl.h>

/*
 * 1. 算子是否需要split, 是看它的ofm是否需要拆分; ofm需要split的条件: is_labeled = 1, do_split = 1
 * 2. 如果一个算子is_labeled = 1, 那么说明它的输入输出的split信息是推导获取的, 是自洽的, 因为
      如果其它算子的推导需要更新它的输入输出的split信息, 且存在conflict, 那么会使用XFRAliasOp来解除conflict
 * 3. 由于算子的split信息最终是由反向推导确定的, 所以一切以反向推导为准
 * 3. 当正向推导, ofm的old/new split信息存在conflict, 由于old信息是下游user算子反向推导得到, 所以以old信息为准, 不做更新
 * 4. 当反向推导, ifm的old/new split信息存在conflict, 由于old信息要么是其它下游算子反向推导得到, 要么是上游算子已经完成正/反
      向推导得到, 要么是用户指定的, 所以old, new都需要满足, 所以需要插入XFRAliasOp来承载新的split信息, 避免冲突
 * 5. 总而言之, 对于is_label = 1的算子, 它的输入输出split信息不能再被串改, 否则会信息混乱

 * 6. 如果推导过程中, 输入的event对于算子是不支持的, 那么输出event是noSplit. 表示这种情况下, 切分到该算子终止.
 */
namespace xp_mlir {
namespace xrt {
LabelOpSplitDim::LabelOpSplitDim() {
    registerOp();
}

#define REGISTER_OP_SPLIT_FORWARD_BACKWARD(op_name, event_op_class) \
    forwardFns_[op_name] = event_op_class::SplitEventForward; \
    backwardFns_[op_name] = event_op_class::SplitEventBackward

void LabelOpSplitDim::registerOp() {
    REGISTER_OP_SPLIT_FORWARD_BACKWARD("xfr.MatMul", MatmulEventOp);
    REGISTER_OP_SPLIT_FORWARD_BACKWARD("xfr.MatMulTC", MatmulEventOp);
    REGISTER_OP_SPLIT_FORWARD_BACKWARD("xfr.Add", EletwEventOp);
    REGISTER_OP_SPLIT_FORWARD_BACKWARD("xfr.Mul", EletwEventOp);
    REGISTER_OP_SPLIT_FORWARD_BACKWARD("xfr.Sub", EletwEventOp);
    REGISTER_OP_SPLIT_FORWARD_BACKWARD("xfr.QuantFixed", QuantEventOp);
    REGISTER_OP_SPLIT_FORWARD_BACKWARD("xfr.Extend", DefaultEventOp);
    REGISTER_OP_SPLIT_FORWARD_BACKWARD("xfr.AddConst", DefaultEventOp);
    REGISTER_OP_SPLIT_FORWARD_BACKWARD("xfr.Lut", DefaultEventOp);
    REGISTER_OP_SPLIT_FORWARD_BACKWARD("xfr.Relu", DefaultEventOp);
    REGISTER_OP_SPLIT_FORWARD_BACKWARD("xfr.Reciprocal", DefaultEventOp);
    REGISTER_OP_SPLIT_FORWARD_BACKWARD("xfr.FullConnection", FullConnectEventOp);
    REGISTER_OP_SPLIT_FORWARD_BACKWARD("xfr.Transpose", TransposeEventOp);
    REGISTER_OP_SPLIT_FORWARD_BACKWARD("xfr.ReduceMax", ReduceEventOp);
    REGISTER_OP_SPLIT_FORWARD_BACKWARD("xfr.Reshape", ReshapeEventOp);
}

SplitEvent LabelOpSplitDim::opForward(mlir::Operation* op, InferReturnType& is_inferred) {
    std::vector<SplitEvent> input_events;
    for (size_t i = 0; i < op->getNumOperands(); ++i) {
        auto input_event = getTensorSplitEvent(op->getOperand(i));
        input_events.push_back(input_event);
    }

    SplitEvent output_event;
    if (std::any_of(input_events.begin(), input_events.end(), [](const SplitEvent& event) { return event.isValid(); })) {
        output_event = forwardFns_.at(getOpSignature(op))(op, input_events);
        is_inferred = InferReturnType::INFER_SUCCESS;
    } 

    return output_event;
}

std::vector<SplitEvent> LabelOpSplitDim::opBackward(mlir::Operation* op, InferReturnType& is_inferred) {
    auto ofm = op->getResult(0);
    auto ofm_event = getTensorSplitEvent(ofm);

    std::vector<SplitEvent> input_events(op->getNumOperands());
    if (ofm_event.isValid()) {
        input_events = backwardFns_.at(getOpSignature(op))(op, ofm_event);
        // if input events are invalid, there is no need to split them
        for (auto& event : input_events) {
            if (!event.isValid()) {
                event.do_split = 0; 
                event.splitDim = -1;
                event.splitNum = -1;
            }
        }
        is_inferred = InferReturnType::INFER_SUCCESS;
    }

    // 如果ofm_event明确表示不拆分, 可以省略推导过程, 设置is_inferred为true, 避免死循环
    // input events保持原样
    if (ofm_event.isNosplit()) {
        is_inferred = InferReturnType::INFER_FAILED;
    }

    return input_events;
}

void LabelOpSplitDim::run(mlir::ModuleOp moduleOp) {
    unittest_reshapeInputOutputAxisRelations();
    moveConstOpFromRegionOp(moduleOp);
    // canonicalizeQuantOpParamShape(moduleOp); // 对于非const的量化参数, 该规范不合理
    canonicalizeEletwOpOperandShape(moduleOp);
    labelOps(moduleOp);
    checkRegionOp(moduleOp);
}

void LabelOpSplitDim::moveConstOpFromRegionOp(mlir::ModuleOp moduleOp) {
    // 保证region op内没有const op
    std::vector<mlir::Operation*> const_ops_to_move;
    moduleOp.walk([&](mlir::Operation* op) -> mlir::WalkResult {
        if (isOpWithRegion(op)) {
            for (auto& op_in_region : op->getRegion(0).front().getOperations()) {
                if (llvm::isa<xfr::XFRConstantOp>(op_in_region)) {
                   const_ops_to_move.push_back(&op_in_region); 
                }
            }
        }
        return mlir::WalkResult::advance();
    });

    for (auto const_op : const_ops_to_move) {
        mlir::Operation* the_outter_parent_op = nullptr;
        auto parent_op = const_op->getParentOp();
        while (isOpWithRegion(parent_op)) {
            the_outter_parent_op = parent_op;
            parent_op = parent_op->getParentOp();
        }
        assert(the_outter_parent_op && "const op should be in region op");
        const_op->moveBefore(the_outter_parent_op);
    }
}

void LabelOpSplitDim::canonicalizeQuantOpParamShape(mlir::ModuleOp moduleOp) {
    moduleOp.walk([&](mlir::Operation* op) -> mlir::WalkResult {
        if (op && llvm::isa<xfr::XFRQuantFixedOp>(op)) {
            auto multipler_val = op->getOperand(1);
            auto shift_val = op->getOperand(2);

            auto multipler_shape = getTensorShape(multipler_val);
            auto shift_shape = getTensorShape(shift_val);
            assert(multipler_shape.size() < 3 && shift_shape.size() < 3 && "multipler and shift must be 1D/2D");

            if (multipler_shape.size() == 2) {
                assert(multipler_shape[0] == 1 && "batch dim of multipler must be 1");
                setTensorShape(multipler_val, {multipler_shape[1]});
            }

            if (shift_shape.size() == 2) {
                assert(shift_shape[0] == 1 && "batch dim of shift must be 1");
                setTensorShape(shift_val, {shift_shape[1]});
            }
        }
        return mlir::WalkResult::advance();
    });
}


void LabelOpSplitDim::canonicalizeEletwOpOperandShape(mlir::ModuleOp moduleOp) {
    moduleOp.walk([&](mlir::Operation* op) -> mlir::WalkResult {
        if (op && llvm::isa<xfr::XFRMulOp, xfr::XFRSubOp, xfr::XFRAddOp>(op)) {
            auto ofm = op->getResult(0);
            auto ofm_shape = getTensorShape(ofm);

            for (auto operand : op->getOperands()) {
                if (isNone(operand)) continue;

                auto operand_def_op = operand.getDefiningOp();
                if (operand_def_op && llvm::isa<xfr::XFRConstantOp>(operand_def_op)) {
                    auto operand_shape = getTensorShape(operand);
                    if (operand_shape.size() == 1) {
                        assert(ofm_shape.back() == operand_shape[0] && "the dim of 1D operand should be the same as ofm channel dim");
                        setTensorShape(operand, ofm_shape);
                    }
                }
            }
        }
        return mlir::WalkResult::advance();
    });
}


void LabelOpSplitDim::checkRegionOp(mlir::ModuleOp moduleOp) {
    moduleOp.walk([](mlir::Operation* op) -> mlir::WalkResult {
        // 如果region内op需要切分, 那么region op必需要切分
        if (isOpInRegionOp(op)) {
            auto region_op = op->getParentOp();
            if (isOpLabeled(op)) {
                if (!isOpLabeled(region_op)) {
                    std::cout << "region op: " << getOpName(region_op) << " should be labeled since op: " << getOpName(op) << " in region is labeled" << std::endl;
                    dumpDebugMlirFile(op);
                    assert(false && "if op in region is labeled, region op should be labeled");
                }
            }
        }

        // 如果region op需要切分, 那么region内所有op都必需要切分, 特殊op除外
        if (isOpWithRegion(op) && isOpLabeled(op) && getTensorSplitEvent(op->getResult(0)).isValid()) {
            assert(0 == op->getNumOperands() && "region op should not have input operand");
            for (auto& op_in_region : op->getRegion(0).front().getOperations()) {
                assert(!llvm::isa<xfr::XFRConstantOp>(op_in_region) && "const op should be moved out of tmu-fuse region");
                if (llvm::isa<xfr::XFRReturnOp>(op_in_region)) {
                    continue;
                }

                if (!(isOpLabeled(&op_in_region) && getTensorSplitEvent(op_in_region.getResult(0)).isValid())) {
                    std::cout << "region op: " << getOpName(op) << " should be labeled since op: " << getOpName(&op_in_region) << " in region is labeled" << std::endl;
                    dumpDebugMlirFile(op);
                    assert(false &&  "all ops in region should be labeled");
                }
            }
        }
        return mlir::WalkResult::advance();
    });
}

void LabelOpSplitDim::labelOps(mlir::ModuleOp moduleOp) {
    bool is_updated = true;
    while (is_updated) {
        is_updated = false;
        moduleOp.walk([this, &is_updated](mlir::Operation* op) -> mlir::WalkResult {
            if(labelOp(op)) {
                is_updated = true;
            }
            return mlir::WalkResult::advance();
        });
    }
}
    
bool LabelOpSplitDim::labelOp(mlir::Operation* op) {
    if (!op) return false;
    // if (!isSplitTarget(op)) return false;

    if (op && forwardFns_.count(getOpSignature(op)) > 0) {
        if(isOpInfered(op)) return false;

        if(llvm::isa<xfr::XFRQuantFixedOp>(op) && frontendMlir::getValueName(op->getResult(0)) == "n_output_quant_fixed") {
            std::cout << "find quant op: " << op->getName().getStringRef().str() << std::endl;
        }

        // 存在如下情况:
        // matmul的输入split k, 导致ofm splitEvent是空的
        // 此时无法反向推导

        // 先尝试正向推导, 如果成功, 则可以设置ofm的split属性,
        // 后面则可以再反向推导, 更新其它input的split属性, 
        // 这样可以一次迭代完成ofm和ifm/wgts的split属性设置, 避免多次迭代才能完成属性设置
        InferReturnType is_forward_inferred = InferReturnType::DO_NOTHING;
        auto output_event = opForward(op, is_forward_inferred);
        if (is_forward_inferred == InferReturnType::INFER_SUCCESS) {
            assert((output_event.isNosplit() || output_event.isValid()) && "output event should be valid or noSplit if forward inference has update");
            auto ofm = op->getResult(0);
            setTensorSplitAttribute(ofm, output_event, op);
        }

        // 反向推导
        InferReturnType is_backward_inferred = InferReturnType::DO_NOTHING;
        auto input_events = opBackward(op, is_backward_inferred);

        // caution:
        // 比如matmul, ifm拆分了k, wgts/bias没有split信息, 正向推导出ofm不需要拆分
        // 此时反向推导无法推导出ifm/wgts/bias的split信息
        if (is_backward_inferred == InferReturnType::INFER_SUCCESS) {
            for (size_t i = 0; i < input_events.size(); ++i) {
                auto operand = op->getOperand(i); 
                setTensorSplitAttribute(operand, input_events[i], op);

                // 如果推导出了region op的输出split信息, 还需要设置region op内return op的输入split信息, 保证信息传递
                if (operand && !isNone(operand)) {
                    auto operand_def_op = operand.getDefiningOp();
                    if (isOpWithRegion(operand_def_op)) {
                        // 因为region op的user可以有些需要split, 有些不需要, 
                        // 所以, 如果是不需要拆分的, 不需要传递给region内部的op
                        // 如果是需要拆分的, 才传递给region内部的op 
                        // 避免region内op存在split event存在两种情况, 因为对于不拆分的情况, 只需要保留原region op就行
                        if (input_events[i].isValid()) {
                            auto return_op = llvm::dyn_cast<xfr::XFRReturnOp>(operand_def_op->getRegion(0).front().getOperations().back());
                            assert(return_op && "the terminator op of region should be xfr.return op");
                            assert(return_op->getNumOperands() == 1 && "xfr.return op should have only one operand");
                            setTensorSplitAttribute(return_op->getOperand(0), input_events[i], return_op);
                        }
                    }
                }
            }
        }
        if (is_backward_inferred != InferReturnType::DO_NOTHING) {
            int64_t label_type = (is_backward_inferred == InferReturnType::INFER_SUCCESS) ? 1 : 0;
            setOpLabeled(op, label_type);
        }

        // 处理region op
        if (isOpLabeled(op)) {
            auto parent_op = op->getParentOp();
            auto user = *(op->getResult(0).getUsers().begin());
            if(isOpWithRegion(parent_op) && llvm::isa<xfr::XFRReturnOp>(user)) {
                setTensorSplitAttribute(parent_op->getResult(0), getTensorSplitEvent(op->getResult(0)), parent_op);
                setOpLabeled(parent_op, 1);
            }
        }

        return (is_forward_inferred != InferReturnType::DO_NOTHING) || (is_backward_inferred != InferReturnType::DO_NOTHING);
    }

    return false;
}



} // namespace xrt
} // namespace xp_mlir