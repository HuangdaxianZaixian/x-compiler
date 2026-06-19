#include "tmu_fuse/TmuFusePattern.hpp"
#include "Helper/TensorUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "tmu_fuse/RegManager.hpp"
#include "xp_mlir/Dialect/XPCommon.h"
#include "xp_mlir/Dialect/XMA/IR/XMAOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include <ostream>

namespace xp_mlir {
namespace xp_tmu {

TmuFusePattern::TmuFusePattern(mlir::Operation* fuse_region_op) : fuse_region_op_(fuse_region_op) {
    assert(isTmuFuse(fuse_region_op_) && "The operation is expected to be a tmu fuse region op");
    fuse_region_op_origin_output_value_ = getOpOutputValue(fuse_region_op_);  
}

bool TmuFusePattern::isTmuFuse(mlir::Operation *op) {
    if (op->getNumRegions() == 1 && 
        llvm::isa<xma::MatMulOp, xma::MatMulTCOp, 
                  xma::ConvOp, xma::WinogradOp, xma::DepthwiseOp, 
                  xma::FCOp, xma::ColWiseOp, xma::AvgPoolingOp, 
                  xma::MaxPoolingOp, xma::GlobalAvgPoolingOp, xma::GlobalMaxPoolingOp, 
                  xma::ResizeOp, xma::AffineOp, xma::GridSampleOp>(op)) {
        return true;
    }
    return false;
}

bool TmuFusePattern::isNoRegionTmu(mlir::Operation *op) {
    bool ret = (op->getNumRegions() == 0);
    if (ret) {
        assert((llvm::isa<xma::MatrixMaxPoolingOp, xma::ConvOp, 
                          xma::WinogradOp, xma::DepthwiseOp, 
                          xma::FCOp, xma::ColWiseOp, xma::AvgPoolingOp, 
                          xma::MaxPoolingOp, xma::GlobalAvgPoolingOp, xma::GlobalMaxPoolingOp, 
                          xma::ResizeOp, xma::AffineOp, xma::GridSampleOp>(op)) && "illegal no-region tmu op");
    }
    return ret;
}

void TmuFusePattern::getAllOpsInFuseRegion() {
    fuse_region_op_->walk([&](mlir::Operation* op) {
        if (op == fuse_region_op_) return;
        if (llvm::isa<xma::MaxOp, xma::MinOp>(op)) {
            assert(isNone(op->getOperand(2)) && "The second operand of MaxOp/MinOp in tmu fuse region should be None");
        }
        all_ops_in_fuse_region_.push_back(op);
    });
}

void TmuFusePattern::genValue2DefTable() {
    std::set<mlir::Value, mlir_value_compare> virtual_output_values;
    value_2_def_table_.clear();
    fuse_region_op_->walk([this, &virtual_output_values](mlir::Operation* op) {
        if (op == fuse_region_op_) return;

        auto result = getOpOutputValue(op);
        assert(value_2_def_table_.count(result) == 0 && "The output value already exist in value_2_def_table_");
        value_2_def_table_[result] = op;

        // 收集virtual op的输出信息
        auto isTensorNameWithVirtualSuffix = [](mlir::Value value) {
            if (isNone(value)) return false;
            auto name = getValueName(value);
            return name.find("_Virtual") != std::string::npos;
        };

        auto input_values = getOpInputValues(op);
        for (auto input_value : input_values) {
            if (isTensorNameWithVirtualSuffix(input_value)) {
                virtual_output_values.insert(input_value);
            }
        }
    });
    assert(virtual_output_values.size() == 1 && "The number of virtual output values should be 1");
    auto virtual_output_value = *virtual_output_values.begin();
    auto tmu_online_output_index = getOutputOperandIndex(fuse_region_op_);
    value_2_def_table_[virtual_output_value] = fuse_region_op_;
    fuse_region_op_->setOperand(tmu_online_output_index, virtual_output_value);

    // auto tmu_online_output_index = getOutputOperandIndex(fuse_region_op_);
    // auto tmu_online_output_value = fuse_region_op_->getOperand(tmu_online_output_index);
    // if (isNone(tmu_online_output_value)) {
    //     // 如果fuse_region_op_ output operand是None, 则默认region内第一个op的第一个operand作为tmu fuse的输出
    //     // 并reset fuse_region_op_的output operand
    //     auto first_op_in_region = all_ops_in_fuse_region_.front();
    //     auto first_operand = getOpInputValues(first_op_in_region).front();
    //     assert(!isNone(first_operand) && "The first operand of the first op in region can't be None when tmu fuse region output is None");
    //     value_2_def_table_[first_operand] = fuse_region_op_;
    //     fuse_region_op_->setOperand(tmu_online_output_index, first_operand);
    // } else {
    //     value_2_def_table_[tmu_online_output_value] = fuse_region_op_;
    // }
}

mlir::Operation* TmuFusePattern::isValueGeneratedInTmuFuse(mlir::Value value) {
    if (value_2_def_table_.count(value) == 0) return nullptr;
    return value_2_def_table_[value];
}

void TmuFusePattern::genOp2InputDefsTable() {
    // 只有当op的operand是defining op在region op内部, 才记录为def op, 否则记录为nullptr
    op_2_input_defs_table_.clear();
    fuse_region_op_->walk([this](mlir::Operation* op) {
        if (op == fuse_region_op_) return;
        auto input_values = getOpInputValues(op);
        for (auto input_value : input_values) {
            auto def_op = isNone(input_value) ? nullptr : isValueGeneratedInTmuFuse(input_value);
            op_2_input_defs_table_[op].push_back(def_op);
        }
    });
}

void TmuFusePattern::genOp2OutputUsesTable() {
    // 只有当user在region op内部, 才记录为user op, 否则记录为nullptr
    op_2_output_uses_table_.clear();
    std::vector<mlir::Operation*> all_target_ops = all_ops_in_fuse_region_;
    all_target_ops.push_back(fuse_region_op_);

    for (auto op_ : all_target_ops) {
        op_2_output_uses_table_[op_] = {};
        for (auto op_defs_pair : op_2_input_defs_table_) {
            auto user_op = op_defs_pair.first;
            auto def_ops = op_defs_pair.second;
            if (is_in_container(def_ops, op_)) {
                op_2_output_uses_table_[op_].push_back(user_op);
            }
        }
    }
}

void TmuFusePattern::parseHwSlotOps() {
    hw_slot_op_partition_.tmu_op_ = fuse_region_op_;
    for (auto op_ : all_ops_in_fuse_region_) {
        if (llvm::isa<xma::QuantOp, xma::DeQuantOp>(op_)) {
            if (isPerTensorQuant(op_) || !isLastDimQuant(op_)) {
                hw_slot_op_partition_.veu_core_ops_.push_back(op_);
            } else {
                // output quant
                if (op_2_output_uses_table_.at(op_).empty()) {
                    assert(hw_slot_op_partition_.quant_op_ == nullptr && "only one veu hw quant op");
                    hw_slot_op_partition_.quant_op_ = op_;
                } else {
                    auto users = op_2_output_uses_table_.at(op_);
                    // quant-lut mode
                    if (std::any_of(users.begin(), users.end(), [](mlir::Operation* user) { return llvm::isa<xma::LutOp>(user); })) {
                        hw_slot_op_partition_.veu_core_ops_.push_back(op_);
                        continue;
                    }

                    assert(hw_slot_op_partition_.dequant_op_ == nullptr && "only one veu hw dequant op");
                    hw_slot_op_partition_.dequant_op_ = op_;
                }
            }
        } else {
            hw_slot_op_partition_.veu_core_ops_.push_back(op_);
        }
    }

    if (hw_slot_op_partition_.dequant_op_ != nullptr) {
        auto input_values = getOpInputValues(hw_slot_op_partition_.dequant_op_);
        assert(input_values.size() == 2 && "dequant op should have 2 input values");
        auto input_def = isValueGeneratedInTmuFuse(input_values.at(0));
        if (input_def == nullptr) {
            hw_slot_op_partition_.is_dequant_input_from_tmu_online_ = false;
        } else {
            assert(input_def == fuse_region_op_ && "the input defining op of dequant op should be the fuse region op");
            hw_slot_op_partition_.is_dequant_input_from_tmu_online_ = true;
        } 
    }
}

void TmuFusePattern::parseVeuCoreOps() {
    for (auto op_ : hw_slot_op_partition_.veu_core_ops_) {
        if (llvm::isa<xma::LutOp>(op_)) {
            assert(veu_core_op_partition_.quant_lut_partion_.lut_op_ == nullptr && "only one quant of quant-lut op");
            veu_core_op_partition_.quant_lut_partion_.lut_op_ = op_;
            continue;
        }

        if (llvm::isa<xma::QuantOp, xma::DeQuantOp>(op_)) {
            if (!isLastDimQuant(op_)) {
                assert(veu_core_op_partition_.matmultc_partition_.not_last_dim_quant_op_ == nullptr && "only one not last dim quant op");
                veu_core_op_partition_.matmultc_partition_.not_last_dim_quant_op_ = op_;
                continue;
            }

            auto users = op_2_output_uses_table_.at(op_);
            // quant-lut mode
            if (std::any_of(users.begin(), users.end(), [](mlir::Operation* user) { return llvm::isa<xma::LutOp>(user); })) {
                assert(veu_core_op_partition_.quant_lut_partion_.quant_op_ == nullptr && "only one quant of quant-lut op");
                veu_core_op_partition_.quant_lut_partion_.quant_op_ = op_;
                continue;
            } else {
                assert(isPerTensorQuant(op_) && "veu core quant/dequant op must be per-tensor quant");
            }
        }

        if (llvm::isa<xma::AddOp, xma::SubOp, xma::MulOp, xma::DivOp>(op_)) {
            auto input_values = getOpInputValues(op_);
            // 如果某个参数为None, 表示是scalar
            if ((isValueGeneratedInTmuFuse(input_values.at(0)) && !isNone(input_values.at(1)) && !isValueGeneratedInTmuFuse(input_values.at(1))) ||
                (!isNone(input_values.at(0)) && !isValueGeneratedInTmuFuse(input_values.at(0)) && isValueGeneratedInTmuFuse(input_values.at(1)))) {
                assert(veu_core_op_partition_.input_from_l1_elem_wise_op_ == nullptr && "only one input from l1 element-wise op");
                veu_core_op_partition_.input_from_l1_elem_wise_op_ = op_;
                continue;
            }

            if (!isNone(input_values.at(0)) && !isValueGeneratedInTmuFuse(input_values.at(0)) && 
                !isNone(input_values.at(1)) && !isValueGeneratedInTmuFuse(input_values.at(1))) {
                assert(false && "unsupported veu core op with two inputs from l1");
            }
        }

        if (llvm::isa<xma::ExtI32Op>(op_)) {
            auto input_values = getOpInputValues(op_);
            auto matmultc_quant_op = veu_core_op_partition_.matmultc_partition_.not_last_dim_quant_op_;
            if (matmultc_quant_op) {
                auto output_uses = op_2_output_uses_table_.at(matmultc_quant_op);
                if (output_uses.size() == 1 && output_uses.at(0) == op_) {
                    assert(veu_core_op_partition_.matmultc_partition_.extend_op_ == nullptr && "only one extend op for matmultc not last dim quant op");
                    veu_core_op_partition_.matmultc_partition_.extend_op_ = op_;
                    continue;
                }
            }
        }
        
        veu_core_op_partition_.other_veu_core_ops_.push_back(op_);
    }

    if (veu_core_op_partition_.input_from_l1_elem_wise_op_ && hw_slot_op_partition_.dequant_op_ && !hw_slot_op_partition_.is_dequant_input_from_tmu_online_) {
        assert(false && "unsupported pattern where dequant op's input is from l1 and there is an element-wise op with one input from l1 in the veu core ops");
    }
}

void TmuFusePattern::genFuseOpInfo() {
    op_2_fuse_info_table_.clear();

    // 对于fuse_region_op_, 它的输出认为是online输出, 使用的是寄存器, 而不是整个op的L1输出
    std::vector<mlir::Operation*> all_target_ops = all_ops_in_fuse_region_;
    all_target_ops.push_back(fuse_region_op_);
    for (auto op_ : all_target_ops) {
        auto input_values = getOpInputValues(op_);
        for (int32_t i = 0; i < input_values.size(); ++i) {
            auto input_value = input_values.at(i);
            if (isNone(input_value)) {
                OpOperandInfo input_info;

                int32_t scalar = -1;
                if (llvm::isa<xma::AddOp, xma::SubOp, xma::MulOp, xma::DivOp, xma::MaxOp, xma::MinOp>(op_)) {
                    // 统一采用将scalar加载到xreg的方式, 因为部分指令不支持imm
                    scalar = getScalarAttr(op_, "scalar");
                    input_info.mem_type = OpOperandMemType::X_REG;
                    input_info.mem_value.x_reg_scalar = static_cast<uint32_t>(scalar); 
                } else if (llvm::isa<xma::QuantOp, xma::DeQuantOp>(op_)) {
                    // 采用imm, codegen是使用vmv.v.i指令将立即数加载到向量寄存器
                    scalar = getScalarAttr(op_, "scalar_param");
                    input_info.mem_type = OpOperandMemType::IMM;
                    input_info.mem_value.imm = static_cast<uint32_t>(scalar);
                } else if (op_ == fuse_region_op_) {
                    input_info.is_none = true;
                } else {
                    assert(false && "unsupported op with scalar input");
                }

                input_info.bit_width = 32;
                op_2_fuse_info_table_[op_].input_infos.push_back(input_info);
                continue;
            }

            auto input_def_op = input_value.getDefiningOp();
            xp_mlir::XDataType data_type = getValueDataType(input_value);

            if (llvm::isa<xma::LutOp>(op_) && i == 1) {
                OpOperandInfo input_info;

                // 对于lut op, 将它的table operand的buffer地址作为scalar参数, 后续codegen时会将这个地址加载到标量寄存器
                auto bit_width_ = getXDataTypeBitWidth(data_type);
                assert((bit_width_ == 8 || bit_width_ == 10) && "the data type of the lut table should be i8 or i10");
                auto buffer_size_ = getScalarAttr(input_def_op, "buffer_size");
                auto elem_num_ = buffer_size_ / ((bit_width_ + 7) / 8);
                assert((elem_num_ == 256 || elem_num_ == 1024) && "the number of elements in the lut table should be 256 or 1024");

                auto scalar = getScalarAttr(input_def_op, "buffer_addr");
                input_info.mem_type = OpOperandMemType::X_REG;
                input_info.mem_value.x_reg_scalar = static_cast<uint32_t>(scalar);
                input_info.bit_width = bit_width_;
                input_info.elem_num = elem_num_;
                op_2_fuse_info_table_[op_].input_infos.push_back(input_info);
                continue;
            }

            if (isValueGeneratedInTmuFuse(input_value)) {
                OpOperandInfo input_info;
                input_info.mem_type = OpOperandMemType::V_REG;
                input_info.mem_value.v_reg_id = -1; // placeholder
                input_info.bit_width = getXDataTypeBitWidth(data_type);
                op_2_fuse_info_table_[op_].input_infos.push_back(input_info);
            } else {
                auto buffer_addr_attr = input_def_op->getAttr("buffer_addr");
                assert(buffer_addr_attr && "buffer_addr attribute is expected in the defining op of the input value");
                auto buffer_addr = llvm::dyn_cast<mlir::IntegerAttr>(buffer_addr_attr).getInt();

                OpOperandInfo input_info;
                input_info.mem_type = OpOperandMemType::L1;
                input_info.mem_value.l1_addr = buffer_addr;
                input_info.bit_width = getXDataTypeBitWidth(data_type);
                op_2_fuse_info_table_[op_].input_infos.push_back(input_info);
            }
        }

        {
            auto output_value = getOpOutputValue(op_);
            assert(output_value && "output value is expected for the op");
            assert(isValueGeneratedInTmuFuse(output_value) && "the output value must be generated in the tmu fuse region");
            auto output_def_op = output_value.getDefiningOp();
            auto data_type_attr = output_def_op->getAttr("data_type");
            assert(data_type_attr && "data_type attribute is expected in the defining op of the input value");
            auto data_type_val = llvm::dyn_cast<mlir::IntegerAttr>(data_type_attr).getInt();
            xp_mlir::XDataType data_type = xp_mlir::symbolizeXDataType(data_type_val).value_or(xp_mlir::XDataType::None);
            assert(data_type != xp_mlir::XDataType::None && "unsupported data type for the output value");

            OpOperandInfo output_info;
            output_info.mem_type = OpOperandMemType::V_REG;
            output_info.mem_value.v_reg_id = -1; // placeholder
            output_info.bit_width = getXDataTypeBitWidth(data_type);
            op_2_fuse_info_table_[op_].output_info = output_info;
        }
    }
}

void TmuFusePattern::setOpUsersInputReg(mlir::Operation* op, int32_t reg_id) {
    auto op_users = op_2_output_uses_table_.at(op);
    for (auto user_op : op_users) {
        auto& user_op_info = op_2_fuse_info_table_.at(user_op);
        auto input_values = getOpInputValues(user_op);
        for (size_t i = 0; i < input_values.size(); ++i) {
            auto& input_info = user_op_info.input_infos.at(i);
            auto& input_value = input_values.at(i);
            auto def_op = isValueGeneratedInTmuFuse(input_value);
            if (def_op == op) {
                assert(input_info.mem_type == OpOperandMemType::V_REG && "the input from fuse_region_op_ must be in register");
                assert(((input_info.mem_value.v_reg_id == -1) || (input_info.mem_value.v_reg_id == reg_id)) && "the register id of the input must be consistent");
                input_info.mem_value.v_reg_id = reg_id;
            }
        }
    }
}

void TmuFusePattern::allocVirtualVectorReg() {
    // tmu online output
    assert(fuse_region_op_ && "fuse_region_op_ is nullptr");
    auto& fuse_region_op_info = op_2_fuse_info_table_.at(fuse_region_op_);
    fuse_region_op_info.output_info.mem_value.v_reg_id = 31; // TIB
    setOpUsersInputReg(fuse_region_op_, 31); // TIB

    // eletw op
    auto eletw_op = veu_core_op_partition_.input_from_l1_elem_wise_op_;
    if (eletw_op) {
        auto& eletw_op_info = op_2_fuse_info_table_.at(eletw_op);
        auto from_L1_operand_index = (eletw_op_info.input_infos.at(0).mem_type == OpOperandMemType::L1) ? 0 : 1;
        auto other_operand_index = 1 - from_L1_operand_index;
        assert(eletw_op_info.input_infos.at(from_L1_operand_index).mem_type == OpOperandMemType::L1 && "the input from l1 of the element-wise op must be in L1");
        assert(eletw_op_info.input_infos.at(other_operand_index).mem_type == OpOperandMemType::V_REG && "the other input of the element-wise op must be in register");

        // 修正为寄存器输入
        eletw_op_info.input_infos.at(from_L1_operand_index).mem_type = OpOperandMemType::V_REG;
        eletw_op_info.input_infos.at(from_L1_operand_index).mem_value.v_reg_id = 30; // ALB
    }

    // dequant op的输出vreg
    auto dequant_op = hw_slot_op_partition_.dequant_op_;
    if (dequant_op) {
        auto& dequant_op_info = op_2_fuse_info_table_.at(dequant_op);
        auto input_values = getOpInputValues(dequant_op);
        auto input_0_value = input_values.at(0);
        auto input_0_def_op = isValueGeneratedInTmuFuse(input_0_value);
        assert((input_0_def_op == nullptr || input_0_def_op == fuse_region_op_) && "one input of the dequant op should be from the fuse region op or None");
        
        int32_t dequant_output_vreg_id = -1;
        if ((input_0_def_op == nullptr)) {
            dequant_output_vreg_id = 30; // ALB
        } else {
            dequant_output_vreg_id = 31; // TIB
        }
        dequant_op_info.output_info.mem_value.v_reg_id = dequant_output_vreg_id;
        setOpUsersInputReg(dequant_op, dequant_output_vreg_id);
    }

    // quant param of lut-quant
    auto quant_of_quant_lut_op = veu_core_op_partition_.quant_lut_partion_.quant_op_;
    if (quant_of_quant_lut_op) {
        auto& quant_of_quant_lut_op_info = op_2_fuse_info_table_.at(quant_of_quant_lut_op);
        assert(quant_of_quant_lut_op_info.input_infos.at(1).mem_type == OpOperandMemType::L1 && "the param input of quant_of_quant_lut_op must be in L1");

        // 如果是swish模式的时候, 情况比较特殊
        // tmu online会先输出到v30, 然后quant从v30读取数据
        // mul的另一路是从v31(dequant输出)读取数据, 这样v31就不会被读取两次
        if (isSwishMode()) {
            quant_of_quant_lut_op_info.input_infos.at(0).mem_type = OpOperandMemType::V_REG;
            quant_of_quant_lut_op_info.input_infos.at(0).mem_value.v_reg_id = 30; // ALB
        }

        // 修正为寄存器输入
        quant_of_quant_lut_op_info.input_infos.at(1).mem_type = OpOperandMemType::V_REG;
        quant_of_quant_lut_op_info.input_infos.at(1).mem_value.v_reg_id = 29; // FB
    }

    // the last output op
    auto last_veu_op = (hw_slot_op_partition_.veu_core_ops_.empty()) ? nullptr : hw_slot_op_partition_.veu_core_ops_.back();
    if (last_veu_op) {
        auto& last_veu_op_info = op_2_fuse_info_table_.at(last_veu_op);
        assert(last_veu_op_info.output_info.mem_type == OpOperandMemType::V_REG && "the output of the last veu core op must be in register");
        auto output_bitwidth = last_veu_op_info.output_info.bit_width;
        assert(output_bitwidth <= 32 && "the output bitwidth of the last veu core op must be less than or equal to 32");
        if (output_bitwidth == 32) {
            last_veu_op_info.output_info.mem_value.v_reg_id = 25;
        } else {
            last_veu_op_info.output_info.mem_value.v_reg_id = 24;
        }
        setOpUsersInputReg(last_veu_op, last_veu_op_info.output_info.mem_value.v_reg_id);
    }
}

void TmuFusePattern::allocVectorReg() {
    // 需要按照IR的顺序进行遍历, 否则producer output reg和consumer input reg不一致的情况
    for (int32_t idx_veu_op = 0; idx_veu_op < all_ops_in_fuse_region_.size(); ++idx_veu_op) {
        auto veu_core_op = all_ops_in_fuse_region_.at(idx_veu_op);
        auto& op_info = op_2_fuse_info_table_.at(veu_core_op);
        assert(op_info.output_info.mem_type == OpOperandMemType::V_REG && "the output of veu core op must be in register");

        if (hw_slot_op_partition_.quant_op_ && (veu_core_op == hw_slot_op_partition_.quant_op_)) {
            auto input_0_info = op_info.input_infos.at(0);
            if (isEmptyVecCoreMode()) {
                assert(input_0_info.mem_type == OpOperandMemType::V_REG && 
                       input_0_info.mem_value.v_reg_id == 31 && 
                       "the input of quant op in single-quant mode must be the output of the fuse region op");
                       op_info.output_info.mem_value.v_reg_id = (op_info.output_info.bit_width == 32) ? 25 : 24;
            } else {
                assert(input_0_info.mem_type == OpOperandMemType::V_REG && 
                    (input_0_info.mem_value.v_reg_id == 24 || input_0_info.mem_value.v_reg_id == 25) &&
                    "the input of quant op in non-single-quant mode must be the output of the last veu core op") ;
                op_info.output_info.mem_value.v_reg_id = input_0_info.mem_value.v_reg_id;
            }
            continue;
        }

        // matmultc not last dim quant
        auto matmultc_quant_op = veu_core_op_partition_.matmultc_partition_.not_last_dim_quant_op_;
        if (matmultc_quant_op && veu_core_op == matmultc_quant_op) {
            auto& input_1_info = op_info.input_infos.at(1);
            assert(input_1_info.mem_type == OpOperandMemType::L1 && "the param input of not_last_dim_quant_op must be in L1");

            // 修正为寄存器输入
            input_1_info.mem_type = OpOperandMemType::V_REG;
            input_1_info.mem_value.v_reg_id = RegManager::get().allocSpecificVectorReg(15).id; // from MatMulTCLD

            // 如果是quant-extend模式, 则输出使用v23寄存器
            auto matmultc_extend_op = veu_core_op_partition_.matmultc_partition_.extend_op_;
            if (matmultc_extend_op) {
                auto& output_info = op_info.output_info;
                output_info.mem_type = OpOperandMemType::V_REG;
                output_info.mem_value.v_reg_id = 23;
                setOpUsersInputReg(veu_core_op, output_info.mem_value.v_reg_id);
            }
        }
        
        for (auto& input_info : op_info.input_infos) {
            // lut of quant-lut不需要分配输入寄存器
            if (veu_core_op == veu_core_op_partition_.quant_lut_partion_.lut_op_) {
                continue;
            }
            if (input_info.mem_type == OpOperandMemType::V_REG && input_info.mem_value.v_reg_id == -1) {
                input_info.mem_value.v_reg_id = RegManager::get().allocVectorReg().id;
            }
        }

        auto& output_info = op_info.output_info;
        // quant of quant-lut不需要分配输出寄存器
        if (veu_core_op == veu_core_op_partition_.quant_lut_partion_.quant_op_) {
            continue;
        }
        if (output_info.mem_value.v_reg_id == -1) {
            output_info.mem_value.v_reg_id = RegManager::get().allocVectorReg().id;
            setOpUsersInputReg(veu_core_op, output_info.mem_value.v_reg_id);
        }
    }
}

bool TmuFusePattern::isEmptyVecCoreMode() const {
    bool ret = hw_slot_op_partition_.veu_core_ops_.empty();
    if (ret) {
        assert(hw_slot_op_partition_.dequant_op_ == nullptr && "the empty veu core mode should not have dequant op");
    }
    return ret;
}

bool TmuFusePattern::isSwishMode() const {
    if (hw_slot_op_partition_.dequant_op_ && veu_core_op_partition_.quant_lut_partion_.quant_op_ && veu_core_op_partition_.quant_lut_partion_.lut_op_) {
        auto quant_of_quant_lut_input_defs = op_2_input_defs_table_.at(veu_core_op_partition_.quant_lut_partion_.quant_op_);
        // quant-lut的输入是matmul
        if (quant_of_quant_lut_input_defs.at(0) == fuse_region_op_) {
            // mul的输入分别是lut和dequant
            if (veu_core_op_partition_.other_veu_core_ops_.size() == 1) {
                auto mul_op = veu_core_op_partition_.other_veu_core_ops_.front();
                if (llvm::isa<xma::MulOp>(mul_op)) {
                    auto mul_input_defs = op_2_input_defs_table_.at(mul_op);
                    if ((mul_input_defs.at(0) == veu_core_op_partition_.quant_lut_partion_.lut_op_ && mul_input_defs.at(1) == hw_slot_op_partition_.dequant_op_) ||
                        (mul_input_defs.at(0) == hw_slot_op_partition_.dequant_op_ && mul_input_defs.at(1) == veu_core_op_partition_.quant_lut_partion_.lut_op_)) {
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

void TmuFusePattern::checkOpInfo() const {
    if (hw_slot_op_partition_.quant_op_) {
        auto op_info = op_2_fuse_info_table_.at(hw_slot_op_partition_.quant_op_);
        assert(op_info.input_infos.at(0).bit_width == 32 && "the input bitwidth of quant op should be 32");
    }

    if (veu_core_op_partition_.matmultc_partition_.not_last_dim_quant_op_) {
        assert(llvm::isa<xma::MatMulTCOp>(fuse_region_op_) && "the not last dim quant op only supports matmultc");

        if (veu_core_op_partition_.matmultc_partition_.extend_op_) {
            auto extend_op_info = op_2_fuse_info_table_.at(veu_core_op_partition_.matmultc_partition_.extend_op_);
            assert(extend_op_info.output_info.mem_type == OpOperandMemType::V_REG &&
                   extend_op_info.output_info.mem_value.v_reg_id == 25 && "the output of the extend op must be in v25");
        }
    }
}

void TmuFusePattern::printFuseInfo() const {
    std::vector<mlir::Operation*> all_target_ops = all_ops_in_fuse_region_;
    all_target_ops.insert(all_target_ops.begin(), fuse_region_op_);
    for (auto op : all_target_ops) {
        auto fuse_info = op_2_fuse_info_table_.at(op);
        std::cout << "Op: " << op->getName().getStringRef().str() << std::endl;
        std::cout << fuse_info.to_string() << std::endl;
    }
    std::cout << std::endl;
}

void TmuFusePattern::parseTmuFuse() {
    getAllOpsInFuseRegion();
    genValue2DefTable();
    genOp2InputDefsTable();
    genOp2OutputUsesTable();

    parseHwSlotOps();
    parseVeuCoreOps();

    genFuseOpInfo();
    allocVirtualVectorReg();
    allocVectorReg();
    checkOpInfo();

    // printFuseInfo();

    // 恢复operand配置
    fuse_region_op_->setOperand(getOutputOperandIndex(fuse_region_op_), fuse_region_op_origin_output_value_);
}

} // namespace xp_tmu
} // namespace xp_mlir

