#include "OpSplitUtils.h"
#include "Helper/TensorUtils.h"
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include "llvm/Support/Casting.h"
#include "Helper/XPMLIRUtils.h"
#include "Utils/mlirUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "xp_mlir/Dialect/XFR/IR/XFROps.h"
#include "xp_mlir/Dialect/XFR/Interfaces/interfaceUtils.h"

namespace xp_mlir {
namespace xrt {

SplitEvent DefaultEventOp::SplitEventForward(mlir::Operation* op, const std::vector<SplitEvent>& input_events) {
    assert(input_events.size() == 1 && "DefaultOp should have exactly 1 input event");
    return input_events[0];
}

std::vector<SplitEvent> DefaultEventOp::SplitEventBackward(mlir::Operation* op, const SplitEvent& output_event) {
    assert(output_event.isValid() && "Output event must be valid for backward split");
    return std::vector<SplitEvent>{output_event};
}

SplitEvent MatmulEventOp::SplitEventForward(mlir::Operation* op, const std::vector<SplitEvent>& input_events) {
    const auto operand_num = op->getNumOperands();
    assert(input_events.size() == operand_num && "error input event num");
    SplitEvent output_event;

    auto ifm_shape = getTensorShape(op->getOperand(0));
    auto wgts_shape = getTensorShape(op->getOperand(1));
    auto ofm_shape = getTensorShape(op->getResult(0));

    assert(ifm_shape.size() == 3 && "ifm must be 3D for matmul");
    assert((wgts_shape.size() == 2 || wgts_shape.size() == 3) && "wts must be 2D/3D for matmul");
    assert(ofm_shape.size() == 3 && "ofm must be 3D for matmul");

    auto ifm_event = input_events[0];
    auto wgts_event = input_events[1];
    

    bool is_split_batch = false;
    bool is_split_m = false;
    bool is_split_n = false;

    if (ifm_event.isValid()) {
        // assert(ifm_event.splitDim != (ifm_shape.size() - 1) && "k cannot be split for matmul");
        if (ifm_event.splitDim == (ifm_shape.size() - 1)) {
            std::cout << "[warning] value name = " << frontendMlir::getValueName(op->getOperand(0)) << ", split k" << std::endl;
            // 如果拆分k的话, 则ofm不需要拆分
            output_event.setNoSplitInfo();
        } else {
            if (ifm_event.splitDim == 0) is_split_batch = true;
            if (ifm_event.splitDim == 1) is_split_m = true;
            output_event.setSplitOpInfo(ifm_event.splitDim, ifm_event.splitNum);
        }
    }

    // in xfr, transpose_b must be false, so the dim -2 of wgts is k, which cannot be split
    if (wgts_event.isValid()) {
        // assert(wgts_event.splitDim != (wgts_shape.size() - 2) && "k cannot be split for matmul");
        if (wgts_event.splitDim == (wgts_shape.size() - 2)) {
            std::cout << "[warning] value name = " << frontendMlir::getValueName(op->getOperand(1)) << ", split k" << std::endl;
            output_event.setNoSplitInfo();
        } else {
            if (wgts_shape.size() == 2) {
                is_split_n = true;
                output_event.setSplitOpInfo(wgts_event.splitDim + 1, wgts_event.splitNum); // output shape is 3D
            }
    
            if (wgts_shape.size() == 3) {
                if (wgts_event.splitDim == 2) {
                    is_split_n = true;
                    output_event.setSplitOpInfo(wgts_event.splitDim, wgts_event.splitNum);
                }
            }
        }
    }

    if (operand_num == 3) {
        auto bias_shape = getTensorShape(op->getOperand(2));
        if (!isNone(op->getOperand(2))) {
            assert((bias_shape.size() == 3 || bias_shape.size() == 1) && "bias must be 1D/3D for matmul");
        }

        auto bias_event = input_events[2];
        if (is_split_n) {
            assert(bias_event.isValid() && bias_event.splitDim == (bias_shape.size() - 1) && "bias must be split along n if n is split");
        }
    }
    
    assert(is_split_batch + is_split_m + is_split_n <= 1 && "Only one dimension can be split for matmul");

    return output_event;
}

std::vector<SplitEvent> MatmulEventOp::SplitEventBackward(mlir::Operation* op, const SplitEvent& output_event) {
    assert(output_event.isValid() && "Output event must be valid for backward split");
    const auto operand_num = op->getNumOperands();

    auto ifm_shape = getTensorShape(op->getOperand(0));
    auto wgts_shape = getTensorShape(op->getOperand(1));
    auto ofm_shape = getTensorShape(op->getResult(0));

    assert(ifm_shape.size() == 3 && "ifm must be 3D for matmul");
    assert((wgts_shape.size() == 2 || wgts_shape.size() == 3) && "wts must be 2D/3D for matmul");
    assert(ofm_shape.size() == 3 && "ofm must be 3D for matmul");

    std::vector<SplitEvent> input_events(((operand_num == 3) ? 3 : 2));
    auto& ifm_event = input_events[0];
    auto& wgts_event = input_events[1];

    bool is_split_batch = (output_event.splitDim == 0);
    bool is_split_m = (output_event.splitDim == 1);
    bool is_split_n = (output_event.splitDim == 2);

    if (operand_num == 3) {
        auto bias_shape = getTensorShape(op->getOperand(2));
        if (!isNone(op->getOperand(2))) {
            assert((bias_shape.size() == 3 || bias_shape.size() == 1) && "bias must be 1D/3D for matmul");
        }
    }

    if (is_split_batch) {
        ifm_event.setSplitOpInfo(0, output_event.splitNum);

        if (wgts_shape.size() == 3 && wgts_shape[2] > 1) {
            wgts_event.setSplitOpInfo(0, output_event.splitNum);
        }

        if (operand_num == 3) {
            auto bias_shape = getTensorShape(op->getOperand(2));
            auto& bias_event = input_events[2];
            if (bias_shape.size() == 3 && bias_shape[2] > 1) {
                bias_event.setSplitOpInfo(0, output_event.splitNum);
            }
        }
    }

    if (is_split_m) {
        ifm_event.setSplitOpInfo(1, output_event.splitNum);
    }

    if (is_split_n) {
        wgts_event.setSplitOpInfo(wgts_shape.size() - 1, output_event.splitNum);
        
        if (operand_num == 3) {
            auto bias_shape = getTensorShape(op->getOperand(2));
            auto& bias_event = input_events[2];
            if (bias_shape.size() > 0) {
                bias_event.setSplitOpInfo(bias_shape.size() - 1, output_event.splitNum);
            }
        }
    }

    return input_events;
}

SplitEvent EletwEventOp::SplitEventForward(mlir::Operation* op, const std::vector<SplitEvent>& input_events) {
    UNUSED(op);
    assert(input_events.size() == 2 && "EletwOp should have exactly 2 input events");
    SplitEvent output_event;

    for (const auto& input_event : input_events) {
        if (input_event.isValid()) {
            if (output_event.splitDim == -1) {
                output_event.setSplitOpInfo(input_event.splitDim, input_event.splitNum);
            } else {
                assert(output_event.splitDim == input_event.splitDim && "Split dimensions must match for both inputs");
                // maybe broadcast
                assert((output_event.splitNum == input_event.splitNum || input_event.splitNum == 1) && "Split numbers must be compatible for both inputs");
                output_event.splitNum = std::max(output_event.splitNum, input_event.splitNum);
            }
        }
    }

    return output_event;
}

std::vector<SplitEvent> EletwEventOp::SplitEventBackward(mlir::Operation* op, const SplitEvent& output_event) {
    assert(output_event.isValid() && "Output event must be valid for backward split");

    std::vector<SplitEvent> input_events(2);
    for(int32_t idx_operand = 0; idx_operand < 2; ++idx_operand) {
        auto operand = op->getOperand(idx_operand);
        if (!isNone(operand)) {
            auto input_shape = getTensorShape(operand);
            auto split_dim = output_event.splitDim;
            if (input_shape.at(split_dim) > 1) {
                input_events[idx_operand].setSplitOpInfo(split_dim, output_event.splitNum);
            }
        }
    }

    return input_events;
}

SplitEvent QuantEventOp::SplitEventForward(mlir::Operation* op, const std::vector<SplitEvent>& input_events) {
    assert(input_events.size() == 3 && "QuantOp should have exactly 3 input events");
    SplitEvent output_event;

    auto ifm_shape = getTensorShape(op->getOperand(0));
    auto multipler_shape = getTensorShape(op->getOperand(1));
    auto shift_shape = getTensorShape(op->getOperand(2));

    assert((multipler_shape.size() < 3) && "multipler must be 1D/2D");
    assert(shift_shape.size() < 3 && "shift must be 1D/2D");
    if (multipler_shape.size() == 2) {
        assert(multipler_shape[0] == 1 && "if multipler is 2D, the dim 0 must be 1");
    }
    if (shift_shape.size() == 2) {
        assert(shift_shape[0] == 1 && "if shift is 2D, the dim 0 must be 1");
    }

    auto ifm_event = input_events[0];
    auto multipler_event = input_events[1];
    auto shift_event = input_events[2];

    int64_t channel_dim = ifm_shape.size() - 1;
    if (op->hasAttr("channelIdx")) {
        channel_dim = llvm::dyn_cast<mlir::IntegerAttr>(op->getAttr("channelIdx")).getInt();
        if (channel_dim < 0) {
            channel_dim += ifm_shape.size();
        }
    }
    bool is_split_channel = false;

    if (ifm_event.isValid()) {
        if (ifm_event.splitDim == channel_dim) is_split_channel = true;
        output_event.setSplitOpInfo(ifm_event.splitDim, ifm_event.splitNum);
    }

    if (ifm_event.isValid() && !is_split_channel) {
        if ((multipler_event.isValid() || shift_event.isValid())) {
            std::cout << "[warning] value name = " << frontendMlir::getValueName(op->getOperand(0)) << ", ifm is not split along channel, multipler and shift cannot be split, ignore multipler and shift split event" << std::endl;
            dumpDebugMlirFile(op);
            assert(false && "If ifm is not split along channel, multipler and shift cannot be split");
        }
    }

    if (multipler_event.isValid()) {
        if (!(multipler_event.splitDim == (multipler_shape.size() - 1))) {
            dumpDebugMlirFile(op);
            assert(false && "multipler must be split along last dim");
        }
        output_event.setSplitOpInfo(channel_dim, multipler_event.splitNum);
    }

    if (shift_event.isValid()) {
        assert(shift_event.splitDim == (shift_shape.size() - 1) && "shift must be split along last dim");
        output_event.setSplitOpInfo(channel_dim, shift_event.splitNum);
    }

    return output_event;
}

std::vector<SplitEvent> QuantEventOp::SplitEventBackward(mlir::Operation* op, const SplitEvent& output_event) {
    assert(output_event.isValid() && "Output event must be valid for backward split");

    auto ifm_shape = getTensorShape(op->getOperand(0));
    auto multipler_shape = getTensorShape(op->getOperand(1));
    auto shift_shape = getTensorShape(op->getOperand(2));

    assert((multipler_shape.size() < 3) && "multipler must be 1D/2D");
    assert(shift_shape.size() < 3 && "shift must be 1D/2D");
    if (multipler_shape.size() == 2) {
        assert(multipler_shape[0] == 1 && "if multipler is 2D, the dim 0 must be 1");
    }
    if (shift_shape.size() == 2) {
        assert(shift_shape[0] == 1 && "if shift is 2D, the dim 0 must be 1");
    }

    std::vector<SplitEvent> input_events(3);
    auto& ifm_event = input_events[0];
    auto& multipler_event = input_events[1];
    auto& shift_event = input_events[2];

    int64_t channel_dim = ifm_shape.size() - 1;
    if (op->hasAttr("channelIdx")) {
        channel_dim = llvm::dyn_cast<mlir::IntegerAttr>(op->getAttr("channelIdx")).getInt();
        if (channel_dim < 0) {
            channel_dim += ifm_shape.size();
        }
    }
    bool is_split_channel = (output_event.splitDim == channel_dim);

    ifm_event.setSplitOpInfo(output_event.splitDim, output_event.splitNum);

    // 对于per-tensor quant param, 不需要split
    if (is_split_channel && multipler_shape.back() > 1) {
        multipler_event.setSplitOpInfo(multipler_shape.size() - 1, output_event.splitNum);
        shift_event.setSplitOpInfo(shift_shape.size() - 1, output_event.splitNum);
    }

    return input_events;
}

SplitEvent TransposeEventOp::SplitEventForward(mlir::Operation* op, const std::vector<SplitEvent>& input_events) {
    assert(input_events.size() == 1 && "TransposeOp should have exactly 1 input event");
    SplitEvent output_event;

    auto perm_attr = llvm::dyn_cast<mlir::ArrayAttr>(op->getAttr("perm"));
    auto perms = xfr::getDataFromI64ArrayAttr(perm_attr);

    auto ifm_event = input_events[0];
    if (ifm_event.isValid()) {
        auto input_split_dim = input_events[0].splitDim;
        auto input_split_num = input_events[0].splitNum;
        auto output_split_dim = transposeInputDim2OutputDim(perms, input_split_dim);
        output_event.setSplitOpInfo(output_split_dim, input_split_num);
    }

    return output_event;
}

std::vector<SplitEvent> TransposeEventOp::SplitEventBackward(mlir::Operation* op, const SplitEvent& output_event) {
    assert(output_event.isValid() && "Output event must be valid for backward split");
    SplitEvent input_event;

    auto perm_attr = llvm::dyn_cast<mlir::ArrayAttr>(op->getAttr("perm"));
    auto perms = xfr::getDataFromI64ArrayAttr(perm_attr);

    if (output_event.isValid()) {
        auto output_split_dim = output_event.splitDim;
        auto output_split_num = output_event.splitNum;
        auto input_split_dim = transposeOutputDim2InputDim(perms, output_split_dim);
        input_event.setSplitOpInfo(input_split_dim, output_split_num);
    }

    return {input_event};
}

SplitEvent ReduceEventOp::SplitEventForward(mlir::Operation* op, const std::vector<SplitEvent>& input_events) {
    assert(input_events.size() == 1 && "ReduceOp should have exactly 1 input event");
    SplitEvent input_event = input_events[0];
    auto ifm_shape = getTensorShape(op->getOperand(0));

    SplitEvent output_event;
    if (input_event.isValid()) {
        if (input_event.splitDim == ifm_shape.size() - 1) {
            std::cout << "[warning] reduce op's ifm name = " << frontendMlir::getValueName(op->getOperand(0)) << ", split last dim" << std::endl;
            output_event.setNoSplitInfo();
        } else {
            output_event.setSplitOpInfo(input_event.splitDim, input_event.splitNum);
        }
    }

    return output_event;
}

std::vector<SplitEvent> ReduceEventOp::SplitEventBackward(mlir::Operation* op, const SplitEvent& output_event) {
    assert(output_event.isValid() && "Output event must be valid for backward split");
    return std::vector<SplitEvent>{output_event};
}

SplitEvent FullConnectEventOp::SplitEventForward(mlir::Operation* op, const std::vector<SplitEvent>& input_events) {
    const auto operand_num = op->getNumOperands();
    assert(input_events.size() == operand_num && "error input event num");
    SplitEvent output_event;

    auto ifm_shape = getTensorShape(op->getOperand(0));
    auto wgts_shape = getTensorShape(op->getOperand(1));
    auto ofm_shape = getTensorShape(op->getResult(0));

    assert(ifm_shape.size() == 2 && "ifm must be 2D");
    assert(ifm_shape.at(0) == 1 && "ifm must be a vector");
    assert((wgts_shape.size() == 3) && "wts must be 3D for matmul");
    assert(wgts_shape.at(0) == 1 && "batch of wgts must be 1");
    assert(ofm_shape.size() == 2 && "ofm must be 3D for matmul");

    auto ifm_event = input_events[0];
    auto wgts_event = input_events[1];
    

    if (ifm_event.isValid()) {
        assert(ifm_event.splitDim != 0 && "ifm axis 0 must be 1, cannot be split for fullconnect");
        std::cout << "[warning] fullconnect ifm name = " << frontendMlir::getValueName(op->getOperand(0)) << ", to be splited" << std::endl;
        // 如果拆分ifm的话, 则ofm不拆分
        output_event.setNoSplitInfo(); 
    }

    // in xfr, the dim -2 of wgts is k, which cannot be split
    if (wgts_event.isValid()) {
        if (wgts_event.splitDim == (wgts_shape.size() - 2)) {
            std::cout << "[warning] fullconnect wgts name = " << frontendMlir::getValueName(op->getOperand(1)) << ", split k" << std::endl;
            output_event.setNoSplitInfo();
        } else {
            if (wgts_event.splitDim == (wgts_shape.size() - 1)) {
                output_event.setSplitOpInfo(ofm_shape.size() - 1, wgts_event.splitNum);
            } else {
                output_event.setSplitOpInfo(wgts_event.splitDim, wgts_event.splitNum);
            }
        }
    }

    return output_event;
}

std::vector<SplitEvent> FullConnectEventOp::SplitEventBackward(mlir::Operation* op, const SplitEvent& output_event) {
    assert(output_event.isValid() && "Output event must be valid for backward split");

    auto ifm_shape = getTensorShape(op->getOperand(0));
    auto wgts_shape = getTensorShape(op->getOperand(1));
    auto ofm_shape = getTensorShape(op->getResult(0));

    assert(ifm_shape.size() == 2 && "ifm must be 2D");
    assert(ifm_shape.at(0) == 1 && "ifm must be a vector");
    assert((wgts_shape.size() == 3) && "wts must be 3D for matmul");
    assert(wgts_shape.at(0) == 1 && "batch of wgts must be 1");
    assert(ofm_shape.size() == 2 && "ofm must be 3D for matmul");

    std::vector<SplitEvent> input_events(2);
    auto& wgts_event = input_events[1];

    if (output_event.isValid()) {
        if (output_event.splitDim == (ofm_shape.size() - 1)) {
            wgts_event.setSplitOpInfo(wgts_shape.size() - 1, output_event.splitNum);
        } else {
            wgts_event.setSplitOpInfo(output_event.splitDim, output_event.splitNum);
        }
    }
    
    return input_events;
}

SplitEvent ReshapeEventOp::SplitEventForward(mlir::Operation* op, const std::vector<SplitEvent>& input_events) {
    assert(input_events.size() == 1 && "ReshapeOp should have exactly 1 input event");
    SplitEvent output_event;

    auto input_shape = getTensorShape(op->getOperand(0));
    auto output_shape = getTensorShape(op->getResult(0));

    auto input_event = input_events[0];
    if (input_event.isValid()) {
        auto infer_axis = getReshapeInferAxis(input_shape, output_shape, input_event);
        if (infer_axis == -1) {
            output_event.setNoSplitInfo();
        } else {
            output_event.setSplitOpInfo(infer_axis, input_event.splitNum);
        }
    }

    return output_event;
}

std::vector<SplitEvent> ReshapeEventOp::SplitEventBackward(mlir::Operation* op, const SplitEvent& output_event) {
    assert(output_event.isValid() && "Output event must be valid for backward split");
    SplitEvent input_event;

    auto input_shape = getTensorShape(op->getOperand(0));
    auto output_shape = getTensorShape(op->getResult(0));

    // 注意此处是output推input, 所以output_shape在前
    auto infer_axis = getReshapeInferAxis(output_shape, input_shape, output_event);
    if (infer_axis == -1) {
        input_event.setNoSplitInfo();
    } else {
        input_event.setSplitOpInfo(infer_axis, output_event.splitNum);
    }
    
    return {input_event};
}

mlir::DictionaryAttr event2DictionaryAttr(mlir::MLIRContext* ctx, const SplitEvent& event) {
    mlir::NamedAttrList attrs;
    attrs.append("doSplit", mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), event.do_split));
    attrs.append("splitDim", mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), event.splitDim));
    attrs.append("splitNum", mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), event.splitNum));

    return mlir::DictionaryAttr::get(ctx, attrs);
}

SplitEvent dictionaryAttr2Event(mlir::DictionaryAttr splitInfo) {
    SplitEvent event;
    if (splitInfo) {
        if (auto doSplitAttr = splitInfo.getAs<mlir::IntegerAttr>("doSplit")) {
            event.do_split = doSplitAttr.getInt();
        }
        if (auto splitDimAttr = splitInfo.getAs<mlir::IntegerAttr>("splitDim")) {
            event.splitDim = splitDimAttr.getInt();
        }
        if (auto splitNumAttr = splitInfo.getAs<mlir::IntegerAttr>("splitNum")) {
            event.splitNum = splitNumAttr.getInt();
        }
    }
    
    return event;
}

void setTensorSplitAttribute(mlir::Value val, const SplitEvent& new_event, mlir::Operation* owner) {
    if (val == nullptr || isNone(val)) return;
    auto ctx = val.getContext();

    /*
     * 1. {valid, valid}
     * 2. {valid, noSplit}
     * 3. {valid, unknown} 
     * 4. {noSplit, noSplit}
     * 5. {noSplit, valid}
     * 6. {noSplit, unknown}
     * 7. {unknown, valid}
     * 8. {unknown, noSplit}
     * 9. {unknown, unknown}
     */

    auto old_event = getTensorSplitEvent(val);

    // case 3, case 6
    if (new_event.isUnknown()) {
        if (!old_event.isUnknown()) {
            std::cout << "value name = " << frontendMlir::getValueName(val) << ", conflict split event on tensor, existing: " << old_event.to_string() << ", new: " << new_event.to_string() << std::endl;
            dumpDebugMlirFile(val);
            assert(false && "cannot set unknown split event on tensor with existing known split event");
        }
        return;
    }

    // case 9
    if (new_event.isUnknown() && old_event.isUnknown()) {
        return;
    }

    // case 1
    // 如果old_event和new_event都有效, 则必须完全相同, 否则说明存在冲突, 且认为这种conflict不可解
    if (old_event.isValid() && new_event.isValid()) {
        if (old_event != new_event) {
            std::cout << "value name = " << frontendMlir::getValueName(val) << ", conflict split event on tensor, existing: " << old_event.to_string() << ", new: " << new_event.to_string() << std::endl;
            dumpDebugMlirFile(val);
            assert(false && "Existing split event on tensor conflicts with new split event");
        }
        return;
    }

    // case 4
    if (old_event.isNosplit() && new_event.isNosplit()) {
        return;
    }

    // case 2, case 5
    // 如果old_event有效但new_event不需要split,
    // 说明存在多个user对同一tensor有不同的split需求
    // 则添加一个XfrAliasOp(以val为输入)来承载新的split属性
    if (old_event.isValid() && new_event.isNosplit() ||
        old_event.isNosplit() && new_event.isValid()) { 
        auto operand_index = getOperandIndex(owner, val);
        assert(operand_index != -1 && "owner operation must have the value as its operand or result");

        if (operand_index == owner->getNumOperands()) {
            // val is a result, 不做任何处理， 维持old event的split属性
        } else {
            // val is an operand
            auto alias_op = getAliasOp(val);
            if (alias_op) {
                auto alias_op_ofm = alias_op->getResult(0);
                auto alias_op_output_event = getTensorSplitEvent(alias_op_ofm);
                assert(alias_op_output_event == new_event && "error alias op with different split event, this should not happen");
            } else {
                // 创建aliasOp, 承载new_event的split属性
                alias_op = createAliasOpAfterValue(val, new_event);
            }

            // 将user挂载到alias_op的输出上
            owner->setOperand(operand_index, alias_op->getResult(0));
        }

        return;
    }

    // case 7, case 8
    auto dic_attr = event2DictionaryAttr(ctx, new_event);
    for (auto& attr : dic_attr) {
        frontendMlir::setValueEncodingAttr(val, attr);
    }

    return;
}

mlir::Operation* createAliasOpAfterValue(mlir::Value val, const SplitEvent& event) {
    auto ctx = val.getContext();
    OpBuilder builder(ctx);

    builder.setInsertionPointAfterValue(val);
    mlir::Operation* alias_op = builder.create<xfr::XFRAliasOp>(val.getLoc(), val.getType(), val);

    // 设置split属性
    auto alias_op_ofm = alias_op->getResult(0);
    auto dic_attr = event2DictionaryAttr(ctx, event);
    for (auto& attr : dic_attr) {
        frontendMlir::setValueEncodingAttr(alias_op_ofm, attr);
    }

    // // 设置已推导 
    // setOpLabeled(alias_op);

    return alias_op;
}

SplitEvent getTensorSplitEvent(mlir::Value val) {
    SplitEvent event;
    if (val == nullptr || isNone(val)) return event;

    auto do_split_attr_opt = frontendMlir::getValueEncodingAttr(val, "doSplit");
    auto split_dim_attr_opt = frontendMlir::getValueEncodingAttr(val, "splitDim");
    auto split_num_attr_opt = frontendMlir::getValueEncodingAttr(val, "splitNum");

    if (do_split_attr_opt.has_value() && split_dim_attr_opt.has_value() && split_num_attr_opt.has_value()) {
        event.do_split = mlir::dyn_cast<mlir::IntegerAttr>(do_split_attr_opt->getValue()).getInt();
        event.splitDim = mlir::dyn_cast<mlir::IntegerAttr>(split_dim_attr_opt->getValue()).getInt();
        event.splitNum = mlir::dyn_cast<mlir::IntegerAttr>(split_num_attr_opt->getValue()).getInt();
        return event;
    }

    assert(!do_split_attr_opt.has_value() && !split_dim_attr_opt.has_value() && !split_num_attr_opt.has_value() &&
            "doSplit, splitDim and splitNum attributes should either all exist or all not exist");
    return event;
}

std::string getOpSignature(mlir::Operation* op) {
    assert(op && "operation should not be null");
    return op->getName().getStringRef().str();
}

int64_t transposeInputDim2OutputDim(const std::vector<int64_t>& perm, int64_t input_dim) {
    assert(input_dim >= 0 && input_dim < static_cast<int64_t>(perm.size()) && "input_dim is out of bounds for perm");
    auto index_in_perm = indexInContainer(input_dim, perm);
    return index_in_perm;
}

int64_t transposeOutputDim2InputDim(const std::vector<int64_t>& perm, int64_t output_dim) {
    assert(output_dim >= 0 && output_dim < static_cast<int64_t>(perm.size()) && "output_dim is out of bounds for perm");
    return perm.at(output_dim);
}

mlir::ModuleOp getModuleFromValue(mlir::Value val) {
    if (!val) return nullptr;
    
    // 获取包含该 Value 的 Operation
    mlir::Operation* op = nullptr;
    if (auto defOp = val.getDefiningOp()) {
        op = defOp;
    } else if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(val)) {
        op = blockArg.getOwner()->getParentOp();
    }
    
    while (op && !llvm::isa<mlir::ModuleOp>(op)) {
        op = op->getParentOp();
    }
  
    if (op) {
        auto moduleOp = llvm::cast<mlir::ModuleOp>(op);
        return moduleOp;
    }
  
    return nullptr;
}

int64_t indexInContainer(int64_t value, const std::vector<int64_t>& container) {
    auto it = std::find(container.begin(), container.end(), value);
    if (it != container.end()) {
        return std::distance(container.begin(), it);
    } else {
        return -1; // Not found
    }
}

int64_t getOperandIndex(mlir::Operation* op, mlir::Value operand) {
    for (size_t i = 0; i < op->getNumOperands(); ++i) {
        if (op->getOperand(i) == operand) {
            return i;
        }
    }

    if (op->getNumResults() > 0 && op->getResult(0) == operand) {
        return op->getNumOperands();
    }

    return -1;
}

mlir::Operation* getAliasOp(mlir::Value val) {
    assert(val && (!isNone(val)) && "value should not be null");
    for (auto user : val.getUsers()) {
        if (llvm::isa<xfr::XFRAliasOp>(user)) {
            return user;
        }
    }

    return nullptr;
}

void setOpLabeled(mlir::Operation* op, int64_t type) {
    // 0: infered but not labeled, 1: labeled for split
    assert(op && "operation should not be null");
    assert(type >= 0 && type <= 1 && "invalid label type, should be 0 or 1");
    auto ctx = op->getContext();
    op->setAttr("is_labeled", mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), type));
}

void unsetOpLabeled(mlir::Operation* op) {
    assert(op && "operation should not be null");
    if (op->hasAttr("is_labeled")) op->removeAttr("is_labeled");
}

void setOpSliced(mlir::Operation* op) {
    // 因为slice/concat算子的插入, 会导致算子重复迭代, 引入is_sliced属性来标记op ofm引进拆分, 避免重复处理
    assert(op && "operation should not be null");
    auto ctx = op->getContext();
    op->setAttr("is_sliced", mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), 1));
}

void unsetOpSliced(mlir::Operation* op) {
    assert(op && "operation should not be null");
    if (op->hasAttr("is_sliced")) op->removeAttr("is_sliced");
}

bool isOpLabeled(mlir::Operation* op) {
    assert(op && "operation should not be null");
    auto attr = op->getAttrOfType<mlir::IntegerAttr>("is_labeled");
    return attr && attr.getInt() == 1;
}

bool isOpInfered(mlir::Operation* op) {
    assert(op && "operation should not be null");
    auto attr = op->getAttrOfType<mlir::IntegerAttr>("is_labeled");
    return attr && attr.getInt() >= 0;
}

bool isOpSliced(mlir::Operation* op) {
    assert(op && "operation should not be null");
    auto attr = op->getAttrOfType<mlir::IntegerAttr>("is_sliced");
    return attr && attr.getInt() == 1;
}

void setOpSplited(mlir::Operation* op) {
    assert(op && "operation should not be null");
    auto ctx = op->getContext();
    op->setAttr("is_splited", mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), 1));
}

void unsetOpSplited(mlir::Operation* op) {
    assert(op && "operation should not be null");
    if (op->hasAttr("is_splited")) op->removeAttr("is_splited");
}

bool isOpSplited(mlir::Operation* op) {
    assert(op && "operation should not be null");
    auto attr = op->getAttrOfType<mlir::IntegerAttr>("is_splited");
    return attr && attr.getInt() == 1;
}

void setOpSplitIndex(mlir::Operation* op, int64_t split_index) {
    assert(op && "operation should not be null");
    auto ctx = op->getContext();
    op->setAttr("split_index", mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), split_index));
}

void unsetOpSplitIndex(mlir::Operation* op) {
    assert(op && "operation should not be null");
    if (op->hasAttr("split_index")) op->removeAttr("split_index");
}

bool ifOpHasSplitIndex(mlir::Operation* op) {
    assert(op && "operation should not be null");
    auto attr = op->getAttrOfType<mlir::IntegerAttr>("split_index");
    return attr && attr.getInt() >= 0;
}

int64_t getOpSplitIndex(mlir::Operation* op) {
    assert(op && "operation should not be null");
    auto attr = op->getAttrOfType<mlir::IntegerAttr>("split_index");
    return attr.getInt();
}

void setSplitTarget(mlir::Operation* op) {
    assert(op && "operation should not be null");

    auto ctx = op->getContext();
    if (isOpInRegionOp(op)) {
        op = op->getParentOp();
    }

    op->setAttr("split_target", mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), 1));
    if (isOpWithRegion(op)) {
        for (auto& op_in_region : op->getRegion(0).front().getOperations()) {
            op_in_region.setAttr("split_target", mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), 1));
        }
    }
}

bool isSplitTarget(mlir::Operation* op) {
    assert(op && "operation should not be null");
    auto attr = op->getAttrOfType<mlir::IntegerAttr>("split_target");
    return attr && attr.getInt() == 1;
}

void setInsertedSliceConcat(mlir::Operation* op) {
    assert(op && "operation should not be null");
    auto ctx = op->getContext();
    op->setAttr("inserted_slice_concat", mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), 1));
}

bool isInsertedSliceConcat(mlir::Operation* op) {
    assert(op && "operation should not be null");
    auto attr = op->getAttrOfType<mlir::IntegerAttr>("inserted_slice_concat");
    return attr && attr.getInt() == 1;
}

void setOpMoved(mlir::Operation* op) {
    assert(op && "operation should not be null");
    auto ctx = op->getContext();
    op->setAttr("is_moved", mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), 1));
}

bool isOpMoved(mlir::Operation* op) {
    assert(op && "operation should not be null");
    auto attr = op->getAttrOfType<mlir::IntegerAttr>("is_moved");
    return attr && attr.getInt() == 1;
}

std::vector<SplitEvent> getOpInputEvents(mlir::Operation* op) {
    std::vector<SplitEvent> input_events;
    for (size_t i = 0; i < op->getNumOperands(); ++i) {
        auto input_event = getTensorSplitEvent(op->getOperand(i));
        input_events.push_back(input_event);
    }
    return input_events;
}

std::vector<int64_t> splitDim(const SplitEvent& event, const std::vector<int64_t>& original_shape, int64_t align_value, std::vector<std::vector<int64_t>>& split_shapes) {
    assert(event.isValid() && "SplitEvent must be valid to get split dimensions");
    
    int64_t dim_size = original_shape.at(event.splitDim);
    int64_t split_num = event.splitNum;

    const int64_t align_value_num = dim_size / align_value;
    const int64_t remain_unalign_value = dim_size % align_value;

    const int64_t align_value_num_per_split = align_value_num / split_num;
    const int64_t remain_align_value_num = align_value_num % split_num;

    std::vector<int64_t> split_dims(split_num, align_value_num_per_split * align_value);
    size_t idx_split = 0;
    for (; idx_split < remain_align_value_num; ++idx_split) {
        split_dims.at(idx_split) += align_value;
    }
    split_dims.at(idx_split) += remain_unalign_value;

    split_shapes = std::vector<std::vector<int64_t>>(split_num, original_shape);
    for (size_t i = 0; i < split_num; ++i) {
        split_shapes.at(i).at(event.splitDim) = split_dims.at(i);
    }
    
    return split_dims;
}

mlir::Operation* createSliceOp(const mlir::Value &val, mlir::PatternRewriter &rewriter,
                    const std::vector<int64_t> &outShape, std::vector<int64_t> axes, 
                    std::vector<int64_t> starts, std::vector<int64_t> ends, std::vector<int64_t> steps, int64_t idx_slice) {
  assert(val && "input value for slice op should not be null");
  RankedTensorType sliceType = llvm::dyn_cast<RankedTensorType>(val.getType());
  if (sliceType == nullptr) {
    LOGE("Invalid input tensor for slice op input\n");
    return nullptr;
  }
  RankedTensorType sliceOutType =
      RankedTensorType::get(outShape, sliceType.getElementType());
  std::vector<mlir::NamedAttribute> attrs = {
    {rewriter.getStringAttr("axes"), rewriter.getI64ArrayAttr(axes)},
    {rewriter.getStringAttr("starts"), rewriter.getI64ArrayAttr(starts)},
    {rewriter.getStringAttr("ends"), rewriter.getI64ArrayAttr(ends)},
    {rewriter.getStringAttr("steps"), rewriter.getI64ArrayAttr(steps)},
  };

  xfr::XFRSliceOp sliceOp = rewriter.create<xfr::XFRSliceOp>(rewriter.getUnknownLoc(), sliceOutType, val, attrs);
  const std::string sliceOutputName = frontendMlir::getValueName(val) + "_slice_" + std::to_string(idx_slice) + "_output";
  setValueName(sliceOp.getResult(), sliceOutputName);
  
  return sliceOp;
}

Operation* createConcatOp(const std::vector<Value> &val, mlir::PatternRewriter &rewriter, int64_t axis) {
  RankedTensorType inType = llvm::cast<RankedTensorType>(val.back().getType());
  if (inType == nullptr) {
    LOGE("Invalid input tensor for concat op input\n");
    return nullptr;
  }
  // get concat out shape
  int64_t concatSize = 0;
  for (size_t i = 0; i < val.size(); i++) {
    std::vector<int64_t> inShape = llvm::cast<RankedTensorType>(val[i].getType()).getShape();
    concatSize += getTensorShape(val[i])[axis];
  }

  std::vector<int64_t> outShape = llvm::cast<RankedTensorType>(val.back().getType()).getShape();
  outShape[axis] = concatSize;

  RankedTensorType concatOutType = RankedTensorType::get(outShape, inType.getElementType());
  std::vector<mlir::NamedAttribute> attrs = {{rewriter.getStringAttr("axis"), rewriter.getI64IntegerAttr(axis)}};
  // 这里的loc需要设置成非重复的, 不能从其它op/val获取, 否则会导致文件重名, 因为后面会将ofm name设置成op loc
  auto concatOp = rewriter.create<xfr::XFRConcatOp>(rewriter.getUnknownLoc(), concatOutType, val, attrs);

  return concatOp;
}

void createSplitValueSliceAndConcat(mlir::Value val, mlir::PatternRewriter &rewriter) {
    bool has_alias_op = false;
    bool if_alias_op_has_valid_split_event = false;
    std::vector<Operation*> origin_users;
    for (auto user : val.getUsers()) {
        if (!llvm::isa<xfr::XFRAliasOp>(user)) {
            origin_users.push_back(user);
        }
    }

    auto split_event = getTensorSplitEvent(val);
    auto alias_op = getAliasOp(val);
    if (alias_op) {
        has_alias_op = true;
        auto alias_op_event = getTensorSplitEvent(alias_op->getResult(0));
        assert((split_event.isValid() && alias_op_event.isNosplit()) || (split_event.isNosplit() && alias_op_event.isValid()) && 
                 "split event on value and its alias op output cannot be both valid or both noSplit");

        if (split_event.isNosplit() && alias_op_event.isValid()) {
            split_event = alias_op_event;
            if_alias_op_has_valid_split_event = true;
        }
    }
    assert(split_event.isValid() && "SplitEvent must be valid to create split value slice and concat");

    rewriter.setInsertionPointAfterValue(val);

    std::vector<std::vector<int64_t>> split_shapes;
    auto align_value = getSplitAlignValue();
    auto split_dims = splitDim(split_event, getTensorShape(val), align_value, split_shapes);

    std::vector<mlir::Value> slice_op_outputs;
    for (int32_t idx_split = 0; idx_split < split_event.splitNum; ++idx_split) {
        int64_t start = std::accumulate(split_dims.begin(), split_dims.begin() + idx_split, 0);
        int64_t end = start + split_dims.at(idx_split);
        int64_t axis = split_event.splitDim;
        int64_t step = 1;

        auto slice_op = createSliceOp(val, rewriter, split_shapes.at(idx_split), {axis}, {start}, {end}, {step}, idx_split);
        setOpSplitIndex(slice_op, idx_split);
        setInsertedSliceConcat(slice_op);
        const std::string sliceOutputName = frontendMlir::getValueName(val) + "_slice_" + std::to_string(idx_split) + "_output";
        frontendMlir::setValueName(slice_op->getResult(0), sliceOutputName);
        slice_op_outputs.push_back(slice_op->getResult(0));
    }

    auto concat_op = createConcatOp(slice_op_outputs, rewriter, split_event.splitDim);
    setTensorSplitAttribute(concat_op->getResult(0), split_event, concat_op);
    const std::string concatOutputName = frontendMlir::getValueName(val) + "_concat_output";
    frontendMlir::setValueName(concat_op->getResult(0), concatOutputName);
    setInsertedSliceConcat(concat_op);

    /*
     * 如果is_labled op的ofm存在有效的split event:
     *   1. 如果op不存在alias op, 则它的所有user变换成以concat为输入
     *   2. 如果op存在alias op:
     *      a. 如果alias op的ofm存在有效的split event, 则将alias op的user变换成以concat为输入, op的user保持不变
     *      b. 如果alias op的ofm不存在有效的split event, 则将op的user变换成以concat为输入, alias op的user变换成以op为输入 
     *   原因:
     *     因为需要op的operand保留实际的拆分信息, 这部分信息在split op时被使用
     */
     if (!has_alias_op) {
        for (auto user : origin_users) {
            user->replaceUsesOfWith(val, concat_op->getResult(0));
        }
    } else {
        if (if_alias_op_has_valid_split_event) {
            for (auto user : alias_op->getResult(0).getUsers()) {
                user->replaceUsesOfWith(alias_op->getResult(0), concat_op->getResult(0));
            }
        } else {
            for (auto user : origin_users) {
                user->replaceUsesOfWith(val, concat_op->getResult(0));
            }
            for (auto user : alias_op->getResult(0).getUsers()) {
                user->replaceUsesOfWith(alias_op->getResult(0), val);
            }
        }
    }

    if (alias_op) rewriter.eraseOp(alias_op);
}

mlir::Value getSplitOperand(mlir::Value val, int64_t split_index) {
   assert(val && "value should not be null");
   if (isNone(val)) return val;

   auto split_event = getTensorSplitEvent(val);
   if (!split_event.isValid()) {
       return val;
   }

   auto val_def_op = val.getDefiningOp();
   assert(llvm::isa<xfr::XFRConcatOp>(val_def_op) && "the defining op of a value with valid split event should be concat op");
   return val_def_op->getOperand(split_index);
}

mlir::Operation* getOfmConcatOp(mlir::Value val) {
    assert(val && "value should not be null");

    mlir::Operation* concat_op = nullptr;
    for (auto user : val.getUsers()) {
        if (auto slice_op = llvm::dyn_cast<xfr::XFRSliceOp>(user)) {
            for (auto slice_user : slice_op->getResult(0).getUsers()) {
                if (llvm::isa<xfr::XFRConcatOp>(slice_user)) {
                    concat_op = slice_user;
                }
            }
        }
    }

    return concat_op;
}

void splitNoRegionOp(mlir::Operation* op, mlir::PatternRewriter &rewriter) {
    auto ofm = op->getResult(0);
    auto split_event = getTensorSplitEvent(ofm);
    
    Operation* last_clone_op = op;
    for (int32_t split_index = 0; split_index < split_event.splitNum; ++split_index) {
        auto split_op = op->clone();
        rewriter.insert(split_op);
        split_op->moveAfter(last_clone_op);
        last_clone_op = split_op;
        unsetOpSliced(split_op);
        eraseValueSplitInfo(split_op->getResult(0));
        setOpSplitIndex(split_op, split_index);

        // input operand
        for (int32_t idx_operand = 0; idx_operand < op->getNumOperands(); ++idx_operand) {
            auto operand = op->getOperand(idx_operand);
            auto split_operand = getSplitOperand(operand, split_index);
            split_op->setOperand(idx_operand, split_operand);
        }

        // ofm
        auto ofm_concat_op = getOfmConcatOp(ofm);
        if (!ofm_concat_op) {
            std::cout << "value name = " << frontendMlir::getValueName(ofm) << ", cannot find concat op consuming ofm after split, this should not happen" << std::endl;
            dumpDebugMlirFile(ofm);
        }
        assert(ofm_concat_op && "ofm should be consumed by a concat op after split");
        auto ofm_concat_operand = ofm_concat_op->getOperand(split_index);
        setTensorShape(split_op->getResult(0), getTensorShape(ofm_concat_operand));
        auto new_ofm_name = frontendMlir::getValueName(ofm) + "_split_" + std::to_string(split_index);
        frontendMlir::setValueName(split_op->getResult(0), new_ofm_name);

        // 重置reshape op的shape attribute
        if (auto reshape_op = llvm::dyn_cast<xfr::XFRReshapeOp>(split_op)) {
            mlir::ArrayAttr shape_attr = rewriter.getI64ArrayAttr(getTensorShape(split_op->getResult(0)));
            reshape_op.setShapeAttr(shape_attr);
        }

        ofm_concat_operand.replaceAllUsesWith(split_op->getResult(0));
    }
    setOpSplited(op);
}

bool isOpWithRegion(Operation *op) {
    if (!op) return false;
    if (0 == op->getNumRegions()) return false;
    if (llvm::isa<mlir::func::FuncOp, mlir::ModuleOp>(op)) return false;
    
    assert((llvm::isa<xfr::XFRTmuFuseOp, xfr::XFRSoftmaxFuseOp>(op)) && "is there any other region op?");
    return true;
}

bool isOpInRegionOp(Operation *op) {
    if (!(op && op->getParentOp())) return false;
    Operation *parentOp = op->getParentOp();
    if (llvm::isa<mlir::func::FuncOp, mlir::ModuleOp>(parentOp)) {
        return false;
    }
    return true;
}

void splitRegionOp(mlir::Operation* op, mlir::PatternRewriter &rewriter) {
    assert(isOpWithRegion(op) && "must be region op");
    assert(isOpSliced(op) && "only sliced op can be split");
    auto ofm = op->getResult(0);
    auto split_event = getTensorSplitEvent(ofm);
    assert(split_event.isValid() && "split event must be valid for splitting region op");

    auto last_clone_op = op;
    std::vector<mlir::Operation*> split_region_ops;
    for (int32_t split_index = 0; split_index < split_event.splitNum; ++split_index) {
        auto split_region_op = op->clone();
        rewriter.insert(split_region_op);
        split_region_op->moveAfter(last_clone_op);
        last_clone_op = split_region_op;
        unsetOpSliced(split_region_op);
        eraseValueSplitInfo(split_region_op->getResult(0));
        setOpSplitIndex(split_region_op, split_index);

        auto return_op = split_region_op->getRegion(0).getBlocks().back().getTerminator();
        assert(llvm::isa<xfr::XFRReturnOp>(return_op) && "the terminator op of tmu-fuse region should be xfr.return op");

        auto ofm_concat_op = llvm::dyn_cast<xfr::XFRConcatOp>(return_op->getOperand(0).getDefiningOp());
        custom_assert(ofm_concat_op, "the operand of xfr.return op should be defined by a concat op", split_region_op);

        auto ofm_concat_operand = ofm_concat_op->getOperand(split_index);
        return_op->setOperand(0, ofm_concat_operand);
        setTensorShape(split_region_op->getResult(0), getTensorShape(ofm_concat_operand));
        auto new_ofm_name = frontendMlir::getValueName(ofm) + "_split_" + std::to_string(split_index);
        frontendMlir::setValueName(split_region_op->getResult(0), new_ofm_name);

        // 反向遍历, 使删除操作合法
        auto& block = split_region_op->getRegion(0).getBlocks().back();
        std::vector<mlir::Operation*> ops_to_erase;
        for (auto it = block.rbegin(); it != block.rend(); ++it) {
            auto op_in_region = &*it;
            if (llvm::isa<xfr::XFRReturnOp>(op_in_region)) {
                continue;
            }

            if (!(ifOpHasSplitIndex(op_in_region) || llvm::isa<xfr::XFRConcatOp>(op_in_region))) {
                dumpDebugMlirFile(op_in_region);
                assert(false && "all ops in the region should be marked with split_index");
            }
            if (ifOpHasSplitIndex(op_in_region) && getOpSplitIndex(op_in_region) == split_index) {
                continue;
            } 
            ops_to_erase.push_back(op_in_region);
        }
        for (auto op_to_erase : ops_to_erase) {
            rewriter.eraseOp(op_to_erase);
        }
        split_region_ops.push_back(split_region_op);
    }

    // 消除region op ofm的slice op, 将slice op的用户替换成对应split region op的输出
    for (auto user : ofm.getUsers()) {
        if (llvm::isa<xfr::XFRSliceOp>(user) && ifOpHasSplitIndex(user)) {
            auto slice_op_split_index = getOpSplitIndex(user);
            auto split_region_op = split_region_ops.at(slice_op_split_index);
            assert(getOpSplitIndex(split_region_op) == slice_op_split_index && "split region op should have the same split index as slice op");
            user->getResult(0).replaceAllUsesWith(split_region_op->getResult(0));

            // 为了处理嵌套region, 这里提前将slice op删除
            assert(user->use_empty() && "slice op should have no use after replaceAllUsesWith");
            rewriter.eraseOp(user);
        }
    }

    // 为了处理嵌套region, 这里提前将region op删除
    assert(op->use_empty() && "region op should have no use after split");
    rewriter.eraseOp(op);
}

void eraseValueSplitInfo(mlir::Value val) {
    assert(val && "value should not be null");
    if (isNone(val)) return;

    frontendMlir::eraseValueEncodingAttr(val, "doSplit");
    frontendMlir::eraseValueEncodingAttr(val, "splitDim");
    frontendMlir::eraseValueEncodingAttr(val, "splitNum");
}

void unsetModuleAllOpsAttr(mlir::ModuleOp module, const std::string& attr_name) {
    assert((attr_name == "is_labeled" || 
            attr_name == "is_sliced" ||
            attr_name == "is_splited" ||
            attr_name == "is_moved" ||
            attr_name == "split_target" ||
            attr_name == "split_index") && "attr_name should be is_labeled or is_sliced");

    module.walk([&](mlir::Operation* op) {
        if (op && op->hasAttr(attr_name)) {
            op->removeAttr(attr_name);
        }

        return mlir::WalkResult::advance();
    });
}

void eraseModuleAllValuesSplitInfo(mlir::ModuleOp module) {
    module.walk([&](mlir::Operation* op) {
        if (auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
            for (auto arg : funcOp.getArguments()) {
                if (arg && !isNone(arg)) {
                    eraseValueSplitInfo(arg);
                }
            }
        }
        
        for (auto result : op->getResults()) {
            if (result && !isNone(result)) {
                eraseValueSplitInfo(result);
            }
        }
        
        return mlir::WalkResult::advance();
    });
}

std::string AxesRelation::to_string() const {
    std::map<AxisDep, std::string> dep_type_to_str = {
        {AxisDep::IS_ONE, "IS_ONE"},
        {AxisDep::EQUAL, "EQUAL"},
        {AxisDep::A_FULL_B_PARTIAL, "A_FULL_B_PARTIAL"},
        {AxisDep::A_PARTIAL_B_FULL, "A_PARTIAL_B_FULL"},
        {AxisDep::A_PARTIAL_B_PARTIAL, "A_PARTIAL_B_PARTIAL"}
    };

    std::string result = "AxesRelation: \n";
    for (size_t i = 0; i < dep_axes.size(); ++i) {
        result += "Input Axis " + std::to_string(i) + ": ";
        for (size_t j = 0; j < dep_axes[i].size(); ++j) {
            result += "(Output Axis " + std::to_string(dep_axes[i][j]) + ", DepType: " + dep_type_to_str.at(dep_types[i][j]) + ") ";
        }
        result += "\n";
    }
    return result;
}

AxesRelation reshapeInputOutputAxisRelations(const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape) {
    AxesRelation ret;
    ret.dep_axes = std::vector<std::vector<int64_t>>(in_shape.size());
    ret.dep_types = std::vector<std::vector<AxisDep>>(in_shape.size());

    int32_t idx_out_axis = 0;
    auto tmp_out_shape = out_shape;
    for (int32_t idx_in_axis = 0; idx_in_axis < in_shape.size(); ++idx_in_axis) {
        int64_t in_axis_size = in_shape.at(idx_in_axis);
        if (in_axis_size == 1) {
            ret.dep_axes.at(idx_in_axis).push_back(-1);
            continue;
        }

        for (; idx_out_axis < tmp_out_shape.size();) {
            auto out_axis_size = tmp_out_shape.at(idx_out_axis);
            if (out_axis_size == 1) {
                ++idx_out_axis;
                continue;
            }

            if (in_axis_size == out_axis_size) {
                ret.dep_axes.at(idx_in_axis).push_back(idx_out_axis);
                ++idx_out_axis;
                break;
            }

            if (in_axis_size > out_axis_size) {
                ret.dep_axes.at(idx_in_axis).push_back(idx_out_axis);
                in_axis_size /= out_axis_size;
                ++idx_out_axis;
                continue;
            } 

            if (in_axis_size < out_axis_size) {
                ret.dep_axes.at(idx_in_axis).push_back(idx_out_axis);
                tmp_out_shape.at(idx_out_axis) = out_axis_size / in_axis_size;
                break;
            }
        }
    }

    std::map<int64_t, int64_t> axis_count_map;
    for (auto &dep_axes : ret.dep_axes) {
        assert(!dep_axes.empty() && "each input axis should have at least one dependent output axis");
        for (auto out_axis : dep_axes) {
            axis_count_map[out_axis]++;
        }
    }

    for (int32_t idx_in_axis = 0; idx_in_axis < ret.dep_axes.size(); ++idx_in_axis) {
        auto &dep_axes = ret.dep_axes.at(idx_in_axis);
        auto& dep_types = ret.dep_types.at(idx_in_axis);

        if (in_shape.at(idx_in_axis) == 1) {
            dep_types.push_back(AxisDep::IS_ONE);
            continue;
        }

        for (auto out_axis : dep_axes) {
            if (axis_count_map[out_axis] == 1) {
                if (dep_axes.size() == 1) {
                    dep_types.push_back(AxisDep::EQUAL);
                } else {
                    dep_types.push_back(AxisDep::A_PARTIAL_B_FULL);
                }
            } else {
                if (dep_axes.size() == 1) {
                    dep_types.push_back(AxisDep::A_FULL_B_PARTIAL);
                } else {
                    dep_types.push_back(AxisDep::A_PARTIAL_B_PARTIAL);
                }
            }
        }
    }

    return ret;
}

int64_t getReshapeInferAxis(const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape, const SplitEvent& input_event) {
    assert(input_event.isValid() && "input split event must be valid for checking if reshape can infer");
    auto split_dim = input_event.splitDim;
    auto split_num = input_event.splitNum;
    assert(in_shape.at(split_dim) > 1 && "the split dimension size should be greater than 1");

    auto axis_relations = reshapeInputOutputAxisRelations(in_shape, out_shape);
    auto dep_axes = axis_relations.dep_axes.at(split_dim);
    auto dep_types = axis_relations.dep_types.at(split_dim);

    if (dep_types.at(0) == AxisDep::EQUAL) {
        assert(dep_axes.size() == 1 && "if the split axis has EQUAL dependency, it should only have one dependent output axis");
        return dep_axes.at(0);
    }

    auto align_value = getSplitAlignValue();
    auto algin_block_size = split_num * align_value;

    // A是B的第一个拆分项
    if (dep_types.at(0) == AxisDep::A_FULL_B_PARTIAL) {
        auto dep_first_out_axis = dep_axes.at(0); 
        if (split_dim == 0) {
            return dep_first_out_axis;
        }

        auto last_in_axis = split_dim - 1;
        if (!is_in_container(axis_relations.dep_axes.at(last_in_axis), dep_first_out_axis)) {
            auto dep_first_out_axis_size = out_shape.at(dep_first_out_axis);
            if ((0 == dep_first_out_axis_size % algin_block_size) && (0 == in_shape.at(split_dim) % algin_block_size)) {
                return dep_first_out_axis;
            }
        }
    }

    // A和A的第一个拆分项都能被algin_block_size整除
    if (dep_types.at(0) == AxisDep::A_PARTIAL_B_FULL) {
        auto dep_first_out_axis = dep_axes.at(0);
        auto dep_first_out_axis_size = out_shape.at(dep_first_out_axis);
        if ((0 == dep_first_out_axis_size % algin_block_size) && (0 == in_shape.at(split_dim) % algin_block_size)) {
            return dep_first_out_axis;
        }
    }

    return -1;
}

void unittest_reshapeInputOutputAxisRelations() {
    {
        std::vector<int64_t> in_shape = {1, 40, 20};
        std::vector<int64_t> out_shape = {2, 10, 2, 20};

        AxesRelation result = reshapeInputOutputAxisRelations(in_shape, out_shape);
        std::cout << "Test Case 1:\n" << result.to_string() << std::endl;
    }

    {
        std::vector<int64_t> in_shape = {1, 40, 20};
        std::vector<int64_t> out_shape = {1, 40, 20};

        AxesRelation result = reshapeInputOutputAxisRelations(in_shape, out_shape);
        std::cout << "Test Case 2:\n" << result.to_string() << std::endl;
    }

    {
        std::vector<int64_t> in_shape = {3, 40, 20};
        std::vector<int64_t> out_shape = {120, 2, 10};

        AxesRelation result = reshapeInputOutputAxisRelations(in_shape, out_shape);
        std::cout << "Test Case 3:\n" << result.to_string() << std::endl;
    }

    {
        std::vector<int64_t> in_shape = {1, 3, 40, 1, 20};
        std::vector<int64_t> out_shape = {120, 2, 10};

        AxesRelation result = reshapeInputOutputAxisRelations(in_shape, out_shape);
        std::cout << "Test Case 4:\n" << result.to_string() << std::endl;
    }
    
}

std::vector<char> getSlicedData(mlir::Operation *slice_op, std::vector<char> const_data, int64_t data_elem_byte_size) {
    // axes = [0], ends = [64], starts = [0], steps = [1]
    auto axes = getIntArrayValue<int64_t>(slice_op->getAttr("axes"));
    auto starts = getIntArrayValue<int64_t>(slice_op->getAttr("starts"));
    auto ends = getIntArrayValue<int64_t>(slice_op->getAttr("ends"));
    auto steps = getIntArrayValue<int64_t>(slice_op->getAttr("steps"));
    assert(axes.size() == 1 && starts.size() == 1 && ends.size() == 1 && steps.size() == 1 && "only support slice with one axis for now");
    assert (steps[0] == 1 && "only support slice with step 1 for now");

    auto ifm_shape = getTensorShape(slice_op->getOperand(0));

    auto axis = axes[0];
    auto start = starts[0];
    auto end = ends[0];
    assert(axis >= 0 && axis < ifm_shape.size() && "slice axis should be valid");

    int64_t sub_block_size = data_elem_byte_size;
    for (int32_t i = axis + 1; i < ifm_shape.size(); i++) {
        sub_block_size *= ifm_shape[i];
    }
    
    int64_t block_num = 1;
    for (int32_t i = 0; i < axis; i++) {
        block_num *= ifm_shape[i];
    }

    std::vector<char> sliced_data;
    for (int32_t k = 0; k < block_num; k++) {
        for (int32_t i = start; i < end; ++i) {
            int64_t block_base = k * ifm_shape.at(axis) * sub_block_size + i * sub_block_size;
            std::copy(const_data.begin() + block_base, const_data.begin() + block_base + sub_block_size, std::back_inserter(sliced_data));
        }
    }

    return sliced_data;
}


std::vector<char> getConstOpRawData(mlir::Operation *const_op) {
    assert(llvm::isa<xfr::XFRConstantOp>(const_op) && "operation should be XFRConstantOp");
    mlir::DenseElementsAttr data = llvm::dyn_cast<mlir::DenseElementsAttr>(const_op->getAttr("data"));

    void *outData = nullptr;
    int sizeBytes = 0;

    if (llvm::isa<mlir::DenseStringElementsAttr>(data)) {
        mlir::DenseStringElementsAttr stringData = llvm::dyn_cast<mlir::DenseStringElementsAttr>(const_op->getAttr("data"));
        auto serial = stringData.getValues<char>();
        std::string serialData(serial.begin(), serial.end());
        sizeBytes = serialData.size();
        outData = malloc(serialData.size());
        memcpy(outData, serialData.data(), serialData.size());
    } else if (llvm::isa<mlir::DenseIntOrFPElementsAttr>(data)) {
        mlir::DenseIntOrFPElementsAttr arrayData = llvm::dyn_cast<mlir::DenseIntOrFPElementsAttr>(const_op->getAttr("data"));
        auto typeData = arrayData.getElementType();
        if (typeData.isUnsignedInteger(8)) {
            auto serial = arrayData.getValues<uint8_t>();
            std::vector<uint8_t> serialData(serial.begin(), serial.end());
            sizeBytes = sizeof(uint8_t) * serial.size();
            outData = malloc(sizeBytes);
            memcpy(outData, serialData.data(), sizeof(char) * serial.size());
        } else if (typeData.isInteger(8)) {
            auto serial = arrayData.getValues<char>();
            std::vector<char> serialData(serial.begin(), serial.end());
            sizeBytes = sizeof(char) * serial.size();
            outData = malloc(sizeBytes);
            memcpy(outData, serialData.data(), sizeof(char) * serial.size());
        } else if (typeData.isInteger(16) || typeData.isInteger(10)) {
            auto serial = arrayData.getValues<int16_t>();
            std::vector<int16_t> serialData(serial.begin(), serial.end());
            sizeBytes = sizeof(int16_t) * serial.size();
            outData = malloc(sizeBytes);
            memcpy(outData, serialData.data(), sizeof(int16_t) * serial.size());
        } else if (typeData.isInteger(32)) {
            auto serial = arrayData.getValues<int32_t>();
            std::vector<int32_t> serialData(serial.begin(), serial.end());
            sizeBytes = sizeof(int32_t) * serial.size();
            outData = malloc(sizeBytes);
            memcpy(outData, serialData.data(), sizeof(int32_t) * serial.size());
        } else if (typeData.isInteger(64)) {
            auto serial = arrayData.getValues<int64_t>();
            std::vector<int64_t> serialData(serial.begin(), serial.end());
            sizeBytes = sizeof(int64_t) * serial.size();
            outData = malloc(sizeBytes);
            memcpy(outData, serialData.data(), sizeof(int64_t) * serial.size());
        } else if (typeData.isF16()) {
            mlir::ArrayRef<char> rawData = arrayData.getRawData();
            auto inputAttr = llvm::dyn_cast<mlir::DenseElementsAttr>(const_op->getAttr("data"));
            auto serial = arrayData.getValues<mlir::APFloat>();
            if (inputAttr.isSplat()) {
                std::vector<uint16_t> serialData;
                serialData.reserve(serial.size());
                for (const auto& apf : serial) {
                    serialData.push_back(static_cast<uint16_t>(apf.bitcastToAPInt().getZExtValue()));
                }
                sizeBytes = sizeof(uint16_t) * serial.size();
                outData = malloc(sizeBytes);
                memcpy(outData, serialData.data(), sizeof(uint16_t) * serial.size());
            } else {
                const uint16_t* fp16Data = reinterpret_cast<const uint16_t*>(rawData.data());
                std::vector<uint16_t> serialData;
                sizeBytes = sizeof(uint16_t) * serial.size();
                outData = malloc(sizeBytes);
                memcpy(outData, fp16Data, sizeof(uint16_t) * serial.size());
            }
        }
    } else {
        assert("data type not realized!\n" && 0);
    }

    std::vector<char> raw_data(sizeBytes, 0);
    assert(sizeBytes > 0 && outData != nullptr && "data should not be empty");
    memcpy(raw_data.data(), outData, sizeBytes);

    free(outData);
    return raw_data;
}

xfr::XFRConstantOp getRawDataConstantOp(const std::vector<char>& raw_data, mlir::RankedTensorType type, const std::string& name, mlir::PatternRewriter &rewriter) {
    auto newType = mlir::RankedTensorType::get(
    type.getShape(), 
    type.getElementType());

    llvm::ArrayRef<char> rawBuffer(raw_data.data(), raw_data.size());
    auto denseAttr = mlir::DenseElementsAttr::getFromRawBuffer(newType, rawBuffer);

    auto newOp = rewriter.create<xfr::XFRConstantOp>(
    rewriter.getUnknownLoc(), denseAttr);
    frontendMlir::setValueName(newOp->getResult(0), name);

    return newOp;
}

void foldConstantSlice(mlir::Operation *slice_op, mlir::PatternRewriter &rewriter) {
    auto const_op = slice_op->getOperand(0).getDefiningOp();
    if(!(const_op && llvm::isa<xfr::XFRConstantOp>(const_op))) return;
    if (frontendMlir::getUserNum(slice_op) == 0) return;

    auto raw_data = getConstOpRawData(const_op);
    auto element_byte_size = (getTensorEltTypeWidth(const_op->getResult(0)) + 7) / 8;
    auto sliced_data = getSlicedData(slice_op, raw_data, element_byte_size);
    auto slice_out_type = llvm::cast<RankedTensorType>(slice_op->getResult(0).getType());
    auto new_const_op = getRawDataConstantOp(sliced_data, slice_out_type, frontendMlir::getValueName(slice_op->getResult(0)), rewriter);
    new_const_op->moveAfter(const_op);

    slice_op->getResult(0).replaceAllUsesWith(new_const_op->getResult(0));
}

void dumpDebugMlirFile(Operation* op) {
    assert(op && "operation should not be null");
    auto mOp = getModuleFromValue(op->getResult(0));
    assert(mOp && "module should not be null");
    (void)frontendMlir::modulePrinter(mOp, "xrt_debug_dump.mlir", false);
}

void dumpDebugMlirFile(mlir::Value val) {
    assert(val && "value should not be null");
    auto mOp = getModuleFromValue(val);
    assert(mOp && "module should not be null");
    (void)frontendMlir::modulePrinter(mOp, "xrt_debug_dump.mlir", false);
}

void custom_assert(bool flag, const std::string& message, mlir::Operation* op) {
    if (!flag) {
        if (op->getNumResults() > 0) {
            std::cout << "op ofm name = " << frontendMlir::getValueName(op->getResult(0)) << ", " << message << std::endl;
        }
        dumpDebugMlirFile(op);
        assert(flag && message.c_str());
    }
}

std::vector<mlir::Value> getOpDependentValues(mlir::Operation* op) {
    std::vector<mlir::Value> dependent_values;
    if (isOpWithRegion(op)) {
        for (auto& op_in_region : op->getRegion(0).front().getOperations()) {
            for (size_t i = 0; i < op_in_region.getNumOperands(); ++i) {
                auto operand = op_in_region.getOperand(i);
                if (isInsideOp(operand, op)) continue;
                dependent_values.push_back(op_in_region.getOperand(i));
            }
        }
    } else {
        return std::vector<mlir::Value>(op->operand_begin(), op->operand_end());
    }
    assert(dependent_values.size() > 0 && "dependent values should not be empty");
    return dependent_values;
}

bool isBlockArgument(mlir::Value val) {
    if (!val) return false;
    return llvm::isa<mlir::BlockArgument>(val);
}

Value getLatestDefinedValueInModule(const std::vector<Value> &values) {
  assert(values.size() > 0 && "values should not be empty");
  
  if (values.size() == 1) {
    return values[0];
  }

  std::vector<mlir::Value> no_block_arg_values;
  for (auto val_ : values) {
    if (!isBlockArgument(val_)) {
        no_block_arg_values.push_back(val_);
    }
  }

  // all values are block arguments, return any of them
  if (no_block_arg_values.size() == 0) return values[0]; 

  auto isValueABeforeValueB = [&](const Value &A, const Value &B) -> bool {
    auto *ADefOp = A.getDefiningOp();
    auto *BDefOp = B.getDefiningOp();
    assert(ADefOp && BDefOp && "values should have defining operations");
    assert(ADefOp->getParentOp() == BDefOp->getParentOp() && "values's defining operations should be in the same block");

    return ADefOp->isBeforeInBlock(BDefOp);
  };

  Value latest = no_block_arg_values[0];
  for (size_t i = 1; i < no_block_arg_values.size(); ++i) {
    if (isValueABeforeValueB(latest, no_block_arg_values[i])) {
      latest = no_block_arg_values[i];
    }
  }
  return latest;
}

int32_t getSplitUserNum(Operation *op) {
  int64_t total_user_num = 0;
  for (auto result : op->getResults()) {
    total_user_num += getValueUserNums(result);
  }
  return total_user_num;
}


} // namespace xrt
} // namespace xp_mlir

