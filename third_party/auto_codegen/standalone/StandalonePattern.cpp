#include "standalone/StandalonePattern.hpp"
#include "Helper/TensorUtils.h"
#include "Helper/XPMLIRUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/APSInt.h"
#include "standalone/RegManager.hpp"
#include "standalone/com_utils.hpp"
#include "xp_mlir/Analysis/Ops.h"
#include "xp_mlir/Dialect/XPCommon.h"
#include "xp_mlir/Dialect/XMA/IR/XMAOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include <numeric>
#include <ostream>

namespace xp_mlir {
namespace xp_std {

StandalonePattern::StandalonePattern(mlir::Operation* standalone_region_op) : 
    standalone_region_op_(standalone_region_op),
    origin_matmulTcLd_input_value_(nullptr) {
    assert(isStandalone(standalone_region_op_) && "The operation is expected to be a standalone op"); 
}

bool StandalonePattern::isStandalone(mlir::Operation *op) {
    if (op && op->getNumRegions() == 1 && 
        llvm::isa<xma::CallOp>(op)) {
        return true;
    }
    return false;
}

std::vector<int64_t> StandalonePattern::getExtendShape(const std::vector<int64_t>& shape) {
    // 统一将形状扩展到4维, 只能在左边扩展维度, 不能在右边扩展维度(因为-1轴涉及对齐)
    auto new_shape = shape;
    assert(new_shape.size() <= 4 && "the shape size is expected to be no more than 4");
    while (new_shape.size() < 4) {
        new_shape.insert(new_shape.begin(), 1);
    }
    return new_shape;
}

bool StandalonePattern::isMatmulTcLd() {
    return all_ops_in_standalone_region_.size() == 1 && llvm::isa<xma::MatMulTCLDOp>(all_ops_in_standalone_region_.front());
}

void StandalonePattern::getAllOpsInStandaloneRegion() {
    standalone_region_op_->walk([&](mlir::Operation* op) {
        if (op == standalone_region_op_) return;
        if (llvm::isa<xma::MaxOp, xma::MinOp>(op)) {
            assert(isNone(op->getOperand(2)) && "The second operand of MaxOp/MinOp in standalone region should be None");
        }

        if (llvm::isa<xma::QuantOp, xma::DeQuantOp>(op)) {
            // 统一将quant/dequant channel_dim设置成负数, 保证形状扩展后, channel_dim仍是正确的
            assert(op->hasAttr("channel_dim") && "quant/dequant op in standalone region should have channel_dim attribute");
            auto channel_dim = getScalarAttr(op, "channel_dim");
            if (channel_dim >= 0) {
                auto output_value = getOpOutputValue(op);
                auto output_shape = getXmaBufferShape(output_value);
                op->setAttr("channel_dim", mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 64), channel_dim - output_shape.size()));
            }
        }

        // 重置matmulTcLd的输入buffer
        if (llvm::isa<xma::MatMulTCLDOp>(op)) {
            auto input_indices = getInputOperandIndices(op);
            auto input_values = getOpInputValues(op);
            assert(input_values.size() == 1 && "matmulTcLd op is expected to have 1 input operands");
            assert(origin_matmulTcLd_input_value_ == nullptr && "only one matmulTdLd op is allowed in the standalone region");
            origin_matmulTcLd_input_value_ = input_values.at(0);
            auto real_input = standalone_region_op_->getOperand(MATMULTCLD_INPUT_INDEX_IN_CALL_OP);
            op->setOperand(input_indices.at(0), real_input);
        }

        all_ops_in_standalone_region_.push_back(op);        
    });

    // 剔除第一个重复op, 除非是MatMulTCLD
    auto iter_first_op = all_ops_in_standalone_region_.begin();
    if (!llvm::isa<xma::MatMulTCLDOp>(*iter_first_op)) {
     all_ops_in_standalone_region_.erase(iter_first_op);
    }
}

void StandalonePattern::genQuantLutPartition() {
    for (auto op : all_ops_in_standalone_region_) {
        if (llvm::isa<xma::LutOp>(op)) {
            assert(quant_lut_partition_.lut_op_ == nullptr && "only one quant of quant-lut op");
            quant_lut_partition_.lut_op_ = op;
            continue;
        }

        if (llvm::isa<xma::QuantOp, xma::DeQuantOp>(op)) {
            auto users = op_2_output_uses_table_.at(op);
            // quant-lut mode
            if (std::any_of(users.begin(), users.end(), [](mlir::Operation* user) { return llvm::isa<xma::LutOp>(user); })) {
                assert(quant_lut_partition_.quant_op_ == nullptr && "only one quant of quant-lut op");
                quant_lut_partition_.quant_op_ = op;
                continue;
            }
        }
    }
}

void StandalonePattern::genValue2DefTable() {
    value_2_def_table_.clear();
    for (auto op : all_ops_in_standalone_region_) {
        if (op == standalone_region_op_) return;

        auto result = getOpOutputValue(op);
        assert(value_2_def_table_.count(result) == 0 && "The output value already exist in value_2_def_table_");
        value_2_def_table_[result] = op;
    }
}

mlir::Operation* StandalonePattern::isValueGeneratedInStandaloneRegion(mlir::Value value) {
    if (value_2_def_table_.count(value) == 0) return nullptr;
    return value_2_def_table_[value];
}

void StandalonePattern::genOp2InputDefsTable() {
    // 只有当op的operand是defining op在region op内部, 才记录为def op, 否则记录为nullptr
    op_2_input_defs_table_.clear();
    for (auto op : all_ops_in_standalone_region_) {
        if (op == standalone_region_op_) return;
        auto input_values = getOpInputValues(op);
        for (auto input_value : input_values) {
            auto def_op = isNone(input_value) ? nullptr : isValueGeneratedInStandaloneRegion(input_value);
            op_2_input_defs_table_[op].push_back(def_op);
        }
    }
}

void StandalonePattern::genOp2OutputUsesTable() {
    // 只有当user在region op内部, 才记录为user op
    op_2_output_uses_table_.clear();
    std::vector<mlir::Operation*> all_target_ops = all_ops_in_standalone_region_;

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

void StandalonePattern::genStandaloneOpInfo() {
    op_2_standalone_info_table_.clear();

    std::vector<mlir::Operation*> all_target_ops = all_ops_in_standalone_region_;
    for (auto op_ : all_target_ops) {
        auto input_values = getOpInputValues(op_);
        for (int32_t i = 0; i < input_values.size(); ++i) {
            auto input_value = input_values.at(i);
            if (isNone(input_value)) {
                OpOperandInfo input_info;

                int32_t scalar = -1;
                if (llvm::isa<xma::AddOp, xma::SubOp, xma::MulOp, xma::DivOp, xma::MaxOp, xma::MinOp, xma::ReciprocalOp>(op_)) {
                    // 统一采用将scalar加载到xreg的方式, 因为部分指令不支持imm
                    scalar = getScalarAttr(op_, "scalar");
                    if (llvm::isa<xma::MulOp>(op_)) {
                        assert(is16bitImm(scalar) && "the scalar value for mul op must be a 16 bit immediate");
                    }
                    input_info.mem_type = OpOperandMemType::X_REG;
                    input_info.mem_value.x_reg_scalar = scalar; 
                } else if (llvm::isa<xma::QuantOp, xma::DeQuantOp>(op_)) {
                    // 采用imm, codegen是使用vmv.v.i指令将立即数加载到向量寄存器
                    scalar = getScalarAttr(op_, "scalar_param");
                    input_info.mem_type = OpOperandMemType::IMM;
                    input_info.mem_value.imm = scalar;
                } else {
                    assert(false && "unsupported op with scalar input");
                    input_info.is_none = true;
                }

                input_info.bit_width = 32;
                input_info.shape = {};
                op_2_standalone_info_table_[op_].input_infos.push_back(input_info);
                continue;
            }

            auto input_def_op = input_value.getDefiningOp();
            xp_mlir::XDataType data_type = getValueDataType(input_value);
            auto input_shape = getXmaBufferShape(input_value);

            if (llvm::isa<xma::LutOp>(op_) && i == 1) {
                OpOperandInfo input_info;

                // 对于lut op, 将它的table operand的buffer地址作为scalar参数, 将这个地址加载到标量寄存器
                // 不能是L1, 因为L1会被迭代, lut table不用迭代
                auto bit_width_ = getXDataTypeBitWidth(data_type);
                assert((bit_width_ == 8 || bit_width_ == 10) && "the data type of the lut table should be i8 or i10");
                auto buffer_size_ = getScalarAttr(input_def_op, "buffer_size");
                auto elem_num_ = buffer_size_ / ((bit_width_ + 7) / 8);
                assert((elem_num_ == 256 || elem_num_ == 1024) && "the number of elements in the lut table should be 256 or 1024");

                auto scalar = getScalarAttr(input_def_op, "buffer_addr");
                input_info.operand_type = OpOperandType::LUT_TABLE;
                input_info.mem_type = OpOperandMemType::X_REG;
                input_info.mem_value.x_reg_scalar = scalar;
                input_info.bit_width = bit_width_;
                input_info.shape = getExtendShape(input_shape);
                op_2_standalone_info_table_[op_].input_infos.push_back(input_info);
                continue;
            }

            if (isValueGeneratedInStandaloneRegion(input_value)) {
                OpOperandInfo input_info;
                input_info.mem_type = OpOperandMemType::V_REG;
                input_info.mem_value.v_reg_id = -1; // placeholder
                input_info.bit_width = getXDataTypeBitWidth(data_type);
                input_info.shape = getExtendShape(input_shape);
                op_2_standalone_info_table_[op_].input_infos.push_back(input_info);
            } else {
                auto buffer_addr_attr = input_def_op->getAttr("buffer_addr");
                assert(buffer_addr_attr && "buffer_addr attribute is expected in the defining op of the input value");
                auto buffer_addr = llvm::dyn_cast<mlir::IntegerAttr>(buffer_addr_attr).getInt();

                OpOperandInfo input_info;
                input_info.mem_type = OpOperandMemType::L1;
                input_info.mem_value.l1_addr = buffer_addr;
                input_info.bit_width = getXDataTypeBitWidth(data_type);
                input_info.shape = getExtendShape(input_shape);
                op_2_standalone_info_table_[op_].input_infos.push_back(input_info);
            }
        }

        // 修正quant/dequant第二个输入的operand类型和channel_dim属性
        if (llvm::isa<xma::QuantOp, xma::DeQuantOp>(op_)) {
            auto& input_1_info = op_2_standalone_info_table_[op_].input_infos.at(1);
            input_1_info.operand_type = OpOperandType::QUANT_PARAM;
            input_1_info.quant_channel_dim = getScalarAttr(op_, "channel_dim");
        }

        {
            auto output_value = getOpOutputValue(op_);
            assert(output_value && "output value is expected for the op");
            assert(isValueGeneratedInStandaloneRegion(output_value) && "the output value must be generated in the standalone region");
            xp_mlir::XDataType data_type = getValueDataType(output_value);
            assert(data_type != xp_mlir::XDataType::None && "unsupported data type for the output value");
            auto output_shape = getXmaBufferShape(output_value);

            OpOperandInfo output_info;
            output_info.mem_type = OpOperandMemType::V_REG;
            output_info.mem_value.v_reg_id = -1; // placeholder
            output_info.bit_width = getXDataTypeBitWidth(data_type);
            output_info.shape = getExtendShape(output_shape);
            op_2_standalone_info_table_[op_].output_info = output_info;
        }
    }

    // 处理call op
    // 对于call op, 认为其没有输出, 最后一个operand是输出
    {
        op_2_standalone_info_table_[standalone_region_op_].input_infos = {};

        auto output_value = getOpOutputValue(standalone_region_op_);
        assert(output_value && "output value is expected for the op");
        auto output_def_op = output_value.getDefiningOp();
        auto buffer_addr_attr = output_def_op->getAttr("buffer_addr");
        assert(buffer_addr_attr && "buffer_addr attribute is expected in the defining op of the output value");
        auto buffer_addr = llvm::dyn_cast<mlir::IntegerAttr>(buffer_addr_attr).getInt();
        auto output_shape = getXmaBufferShape(output_value);

        OpOperandInfo output_info;
        output_info.mem_type = OpOperandMemType::L1;
        output_info.mem_value.l1_addr = buffer_addr;
        xp_mlir::XDataType data_type = getValueDataType(output_value);
        output_info.bit_width = getXDataTypeBitWidth(data_type);
        output_info.shape = getExtendShape(output_shape);
        op_2_standalone_info_table_[standalone_region_op_].output_info = output_info;
    }
    
}

void StandalonePattern::setOpUsersInputReg(mlir::Operation* op, int32_t reg_id) {
    auto op_users = op_2_output_uses_table_.at(op);
    for (auto user_op : op_users) {
        auto& user_op_info = op_2_standalone_info_table_.at(user_op);
        auto input_values = getOpInputValues(user_op);
        for (size_t i = 0; i < input_values.size(); ++i) {
            auto& input_info = user_op_info.input_infos.at(i);
            auto& input_value = input_values.at(i);
            auto def_op = isValueGeneratedInStandaloneRegion(input_value);
            if (def_op == op) {
                assert(input_info.mem_type == OpOperandMemType::V_REG && "the input from standalone region must be in register");
                assert(((input_info.mem_value.v_reg_id == -1) || (input_info.mem_value.v_reg_id == reg_id)) && "the register id of the input must be consistent");
                input_info.mem_value.v_reg_id = reg_id;
            }
        }
    }
}

void StandalonePattern::allocVectorReg() {
    // 需要按照IR的顺序进行遍历, 否则producer output reg和consumer input reg不一致的情况
    for (auto veu_core_op : all_ops_in_standalone_region_) {
        auto& op_info = op_2_standalone_info_table_.at(veu_core_op);
        assert(op_info.output_info.mem_type == OpOperandMemType::V_REG && "the output of veu core op must be in register");
        
        for (auto& input_info : op_info.input_infos) {
            // lut of quant-lut不需要分配输入寄存器
            if (veu_core_op == quant_lut_partition_.lut_op_) {
                continue;
            }
            if (input_info.mem_type == OpOperandMemType::V_REG && input_info.mem_value.v_reg_id == -1) {
                input_info.mem_value.v_reg_id = RegManager::get().allocVectorReg().id;
            }
        }

        auto& output_info = op_info.output_info;
        // quant of quant-lut不需要分配输出寄存器
        if (veu_core_op == quant_lut_partition_.quant_op_) {
            continue;
        }
        if (output_info.mem_value.v_reg_id == -1) {
            output_info.mem_value.v_reg_id = RegManager::get().allocVectorReg().id;
            setOpUsersInputReg(veu_core_op, output_info.mem_value.v_reg_id);
        }
    }
}

void StandalonePattern::checkOpInfo() const {}

void StandalonePattern::printStandaloneInfo() const {
    std::vector<mlir::Operation*> all_target_ops = all_ops_in_standalone_region_;
    all_target_ops.insert(all_target_ops.begin(), standalone_region_op_);
    for (auto op : all_target_ops) {
        auto fuse_info = op_2_standalone_info_table_.at(op);
        std::cout << "Op: " << op->getName().getStringRef().str() << std::endl;
        std::cout << fuse_info.to_string() << std::endl;
    }
    std::cout << std::endl;
}

void StandalonePattern::restoreMatmulTcLdInput() {
    if (isMatmulTcLd()) {
        // matmulTcLd的输入输出寄存器必须一致
        auto matmulTcLd_op = all_ops_in_standalone_region_.front();
        auto input_indices = getInputOperandIndices(matmulTcLd_op);
        assert(input_indices.size() == 1 && "matmulTcLd op is expected to have 1 input operand");
        matmulTcLd_op->setOperand(input_indices.at(0), origin_matmulTcLd_input_value_);
    }
}

void StandalonePattern::parseStandalone() {
    RegManager::get().reset();

    getAllOpsInStandaloneRegion();
    genValue2DefTable();
    genOp2InputDefsTable();
    genOp2OutputUsesTable();

    genQuantLutPartition();

    genStandaloneOpInfo();
    allocVectorReg();
    checkOpInfo();

    // 恢复matmulTcLd的输入, 避免对后续的pattern匹配造成影响
    restoreMatmulTcLdInput();
    
    printStandaloneInfo();
}

} // namespace xp_std
} // namespace xp_mlir

