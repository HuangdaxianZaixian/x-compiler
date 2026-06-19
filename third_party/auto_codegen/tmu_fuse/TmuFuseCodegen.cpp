#include "tmu_fuse/TmuFuseCodegen.hpp"
#include "tmu_fuse/RvvInstGen.hpp"
#include "xp_mlir/Dialect/XMA/IR/XMAOps.h"
#include <iostream>

namespace xp_mlir {
namespace xp_tmu {
TmuFuseCodegen::TmuFuseCodegen(mlir::Operation* fuse_region_op, uint32_t loop_time) : 
    fuse_region_op_(fuse_region_op),
    loop_time_(loop_time) {
    prelogue_code_.clear();
    loop_code_.clear();
}

void TmuFuseCodegen::codegen() {
    RegManager::get().reset();
    
    // 顺序不能错乱
    genVeuOpsCode();
    genVsetvliCode();
    genLoopConfigCode();

    scheduleInsts();

    // std::cout << toAssembly() << std::endl;
}

void TmuFuseCodegen::genVeuOpsCode() {
    // no region tmu
    if (TmuFusePattern::isNoRegionTmu(fuse_region_op_)) {
        // std::cout << "no region op: " << fuse_region_op_->getName().getStringRef().str() << std::endl;
        auto tmu_online_bitwidth = getTmuOnlineOutputBitwidth(fuse_region_op_);
        auto inst_= OpCodegen::getEmptyVecCoreModeCode(tmu_online_bitwidth);
        loop_code_.push_back(inst_);
        return;
    }

    // tmu with region
    TmuFusePattern tmu_fuse_pattern_(fuse_region_op_);
    tmu_fuse_pattern_.parseTmuFuse();

    if (tmu_fuse_pattern_.isEmptyVecCoreMode()) {
        auto tmu_online_bitwidth = getTmuOnlineOutputBitwidth(fuse_region_op_);
        auto inst_= OpCodegen::getEmptyVecCoreModeCode(tmu_online_bitwidth);
        loop_code_.push_back(inst_);
        return;
    }

    auto& hw_slot_op_partition = tmu_fuse_pattern_.getHwSlotOpPartition();
    auto& veu_core_op_partition = tmu_fuse_pattern_.getVeuCoreOpPartition();
    auto& op_2_fuse_info_table = tmu_fuse_pattern_.getOp2FuseInfoTable();

    for (int32_t i = 0; i < hw_slot_op_partition.veu_core_ops_.size(); ++i) {
        auto op = hw_slot_op_partition.veu_core_ops_.at(i);

        if (veu_core_op_partition.quant_lut_partion_.quant_op_ && op == veu_core_op_partition.quant_lut_partion_.quant_op_) {
            continue;
        }

        if (veu_core_op_partition.matmultc_partition_.extend_op_ && op == veu_core_op_partition.matmultc_partition_.extend_op_) {
            continue;
        }

        std::vector<mlir::Operation*> ops_to_codegen;
        std::vector<FuseOpInfo> fuse_op_infos;
        CodegenOpType codegen_type = CodegenOpType::GENERAL;
        if (veu_core_op_partition.quant_lut_partion_.lut_op_ && op == veu_core_op_partition.quant_lut_partion_.lut_op_) {
            // quant-lut
            assert(i > 0 && "the quant op should be before lut op in the ops sequence");
            auto pre_quant_op = hw_slot_op_partition.veu_core_ops_.at(i - 1);
            assert(veu_core_op_partition.quant_lut_partion_.quant_op_ && 
                   pre_quant_op == veu_core_op_partition.quant_lut_partion_.quant_op_ && "the quant op should be right before lut op in the ops sequence");
            ops_to_codegen.push_back(pre_quant_op);
            ops_to_codegen.push_back(op);
            fuse_op_infos.push_back(op_2_fuse_info_table.at(pre_quant_op));
            fuse_op_infos.push_back(op_2_fuse_info_table.at(op));
            codegen_type = CodegenOpType::QUANT_LUT;
        } else if (veu_core_op_partition.matmultc_partition_.not_last_dim_quant_op_ && 
                   op == veu_core_op_partition.matmultc_partition_.not_last_dim_quant_op_) {
            // matmultc not last dim quantization pattern
            ops_to_codegen.push_back(op);
            fuse_op_infos.push_back(op_2_fuse_info_table.at(op));
            codegen_type = CodegenOpType::MATMULTC_NOT_LAST_DIM_QUANT;
            
            auto matmultc_extend_op = veu_core_op_partition.matmultc_partition_.extend_op_;
            if (matmultc_extend_op) {
                ops_to_codegen.push_back(matmultc_extend_op);
                fuse_op_infos.push_back(op_2_fuse_info_table.at(matmultc_extend_op));
                codegen_type = CodegenOpType::MATMULTC_QUANT_EXTEND;
            }
        } else {
            ops_to_codegen.push_back(op);
            fuse_op_infos.push_back(op_2_fuse_info_table.at(op));
        }

        OpCodegen codegen(ops_to_codegen, fuse_op_infos, codegen_type);
        codegen.codegen();
        prelogue_code_.insert(prelogue_code_.end(), codegen.getPrelogueCode().begin(), codegen.getPrelogueCode().end());
        loop_code_.insert(loop_code_.end(), codegen.getLoopCode().begin(), codegen.getLoopCode().end());
    }
}

void TmuFuseCodegen::genVsetvliCode() {
    // "addi x15, x0, 128",
    // "vsetvli x30, x15, e32, m1, ta, ma"
    {
        RvvInstGenerator addi_gen("addi");
        CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = {.id = 0}}};
        CodegenOpOperand arg1 = {.type = RvvOperandType::IMM, .value = {.imm = 128}};
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::XREG, .value = {.x_reg = {.id = 15}}};

        auto code_ = addi_gen.codegen(inputs, output);
        prelogue_code_.insert(prelogue_code_.end(), code_);
    }

    {
        RvvInstGenerator vsetvli_gen("vsetvli");
        CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = {.id = 15}}};
        std::vector<CodegenOpOperand> inputs = {arg0};
        CodegenOpOperand output = {.type = RvvOperandType::XREG, .value = {.x_reg = {.id = 30}}};

        auto code_ = vsetvli_gen.codegen(inputs, output);
        prelogue_code_.insert(prelogue_code_.end(), code_);
    }
}

void TmuFuseCodegen::genLoopConfigCode() {
    // "lui x17, #(loop_times)"
    // "addiu x17, x17, #(loop_times)"
    // "loop_cfgx x17, 0"
    // "loop_end 0"
    auto rd = OpCodegen::genScalarLoadCode(loop_time_, prelogue_code_);

    {
        RvvInstGenerator loop_cfgx_gen("loop_cfgx");
        CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = rd}};
        CodegenOpOperand arg1 = {.type = RvvOperandType::IMM, .value = {.imm = 0}};
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::NONE};

        auto code_ = loop_cfgx_gen.codegen(inputs, output);
        prelogue_code_.insert(prelogue_code_.end(), code_);
    }

    {
        RvvInstGenerator loop_end_gen("loop_end");
        CodegenOpOperand arg0 = {.type = RvvOperandType::IMM, .value = {.imm = 0}};
        std::vector<CodegenOpOperand> inputs = {arg0};
        CodegenOpOperand output = {.type = RvvOperandType::NONE};

        auto code_ = loop_end_gen.codegen(inputs, output);
        loop_code_.insert(loop_code_.end(), code_);
    }
}

void TmuFuseCodegen::dispatchInstBundle(std::vector<RvvInst>& inst_bundle) {
    assert(inst_bundle.size() == 2 && "the size of instruction bundle should be 2");
    auto a = inst_bundle.at(0);
    auto b = inst_bundle.at(1);

    assert(((a.isUsingVirtualVectorRegisterV23() && b.isUsingVirtualVectorRegisterV23()) ||
             (!a.isUsingVirtualVectorRegisterV23() && !b.isUsingVirtualVectorRegisterV23())) && "v23 must be used in two instructions at the same time if it is used");

    if (RvvInst::isInstADependentOnInstB(b, a)) {
        auto nop_inst = OpCodegen::genNopRvvInst();
        std::vector<RvvInst> inst_bundle_0 = {a, nop_inst};
        std::vector<RvvInst> inst_bundle_1 = {nop_inst, b};
        inst_sequence_.push_back(inst_bundle_0);
        inst_sequence_.push_back(inst_bundle_1);
    } else {
        inst_sequence_.push_back(inst_bundle);
    }

    inst_bundle.clear();
}

void TmuFuseCodegen::packExclusiveOneBundleInst(const RvvInst& inst, std::vector<RvvInst>& inst_bundle, VeuInstSlotSupportion& cur_slot) {
    // 新启动一个bundle
    auto nop_inst = OpCodegen::genNopRvvInst();
    if (cur_slot == VeuInstSlotSupportion::SLOT_1) {
        inst_bundle.push_back(nop_inst);
        dispatchInstBundle(inst_bundle);
    }

    auto support_slot_type = inst.slot_supportion;
    if (support_slot_type == VeuInstSlotSupportion::SLOT_0 || support_slot_type == VeuInstSlotSupportion::SLOT_0_1) {
        inst_bundle.push_back(inst);
        inst_bundle.push_back(nop_inst);
    } else if (support_slot_type == VeuInstSlotSupportion::SLOT_1) {
        inst_bundle.push_back(nop_inst);
        inst_bundle.push_back(inst);
    } else {
        throw std::runtime_error("unsupported slot support type for exclusive one bundle instruction");
    }

    dispatchInstBundle(inst_bundle);
    cur_slot = VeuInstSlotSupportion::SLOT_0;
}

void TmuFuseCodegen::determineVectorInstSlot() {
    VeuInstSlotSupportion using_v24_v25_inst_slot = VeuInstSlotSupportion::UNKNOWN;
    VeuInstSlotSupportion other_inst_slot = VeuInstSlotSupportion::UNKNOWN;
    for (auto& inst : loop_code_) {
        if (inst.isUsingVirtualVectorRegisterV24OrV25()) {
            assert(using_v24_v25_inst_slot == VeuInstSlotSupportion::UNKNOWN && "there should be only one instruction using v24/v25 in loop code");
            using_v24_v25_inst_slot = inst.slot_supportion;
        } else if (inst.isUsingGeneralVectorRegister() && inst.slot_supportion != VeuInstSlotSupportion::SLOT_0_1) {
            if (other_inst_slot == VeuInstSlotSupportion::UNKNOWN) {
                other_inst_slot = inst.slot_supportion;
            } else {
                assert(other_inst_slot == inst.slot_supportion && "there should be only one slot type for instructions using general vector registers in loop code");
            }
        }
    }
    assert(using_v24_v25_inst_slot != VeuInstSlotSupportion::UNKNOWN && "there should be one instruction using v24/v25 in loop code");
    assert(using_v24_v25_inst_slot != other_inst_slot && "the instruction using v24/v25 cannot be in the same slot with instructions using general vector registers");

    switch (using_v24_v25_inst_slot) {
        case VeuInstSlotSupportion::SLOT_0: {
            other_inst_slot = VeuInstSlotSupportion::SLOT_1;
            break;
        }
        case VeuInstSlotSupportion::SLOT_1: {
            other_inst_slot = VeuInstSlotSupportion::SLOT_0;
            break;
        }
        case VeuInstSlotSupportion::SLOT_0_1: {
            if (other_inst_slot == VeuInstSlotSupportion::SLOT_0_1 || other_inst_slot == VeuInstSlotSupportion::UNKNOWN) {
                using_v24_v25_inst_slot = VeuInstSlotSupportion::SLOT_0;
                other_inst_slot = VeuInstSlotSupportion::SLOT_1;
            } else {
                using_v24_v25_inst_slot = (other_inst_slot == VeuInstSlotSupportion::SLOT_0) ? VeuInstSlotSupportion::SLOT_1 : VeuInstSlotSupportion::SLOT_0;
            }
            break;
        }
        default:
            throw std::runtime_error("invalid slot support type for instruction using v24/v25");
    }
    
    for (auto& inst : loop_code_) {
        if (inst.isUsingVirtualVectorRegisterV24OrV25()) {
            inst.slot_supportion = using_v24_v25_inst_slot;
        } else if (inst.isUsingGeneralVectorRegister()) {
            inst.slot_supportion = other_inst_slot;
        }
    }
}

void TmuFuseCodegen::dispatchInst(const RvvInst& inst, bool is_last) {
    static std::vector<RvvInst> inst_bundle;
    static VeuInstSlotSupportion cur_slot = VeuInstSlotSupportion::SLOT_0;

    if (RvvInst::isExclusiveOneBundleInst(inst)) {
        packExclusiveOneBundleInst(inst, inst_bundle, cur_slot);
        return;
    }

    if (inst.slot_supportion == VeuInstSlotSupportion::SLOT_0_1 || inst.slot_supportion == cur_slot) {
        inst_bundle.push_back(inst);
        if (inst_bundle.size() == 2) {
            dispatchInstBundle(inst_bundle);
        }
        cur_slot = (cur_slot == VeuInstSlotSupportion::SLOT_0) ? VeuInstSlotSupportion::SLOT_1 : VeuInstSlotSupportion::SLOT_0;
    } else if (cur_slot == VeuInstSlotSupportion::SLOT_0) {
        auto nop_inst = OpCodegen::genNopRvvInst();
        inst_bundle.push_back(nop_inst);
        inst_bundle.push_back(inst);
        dispatchInstBundle(inst_bundle);
        cur_slot = VeuInstSlotSupportion::SLOT_0;
    } else if (cur_slot == VeuInstSlotSupportion::SLOT_1) {
        auto nop_inst = OpCodegen::genNopRvvInst();
        inst_bundle.push_back(nop_inst);
        dispatchInstBundle(inst_bundle);
        inst_bundle.push_back(inst);
        cur_slot = VeuInstSlotSupportion::SLOT_1;
    } else {
        throw std::runtime_error("invalid current slot type during instruction dispatch");
    }

    if (is_last) {
        if (cur_slot == VeuInstSlotSupportion::SLOT_1) {
            auto nop_inst = OpCodegen::genNopRvvInst();
            inst_bundle.push_back(nop_inst);
            dispatchInstBundle(inst_bundle);
        }
    }
}

void TmuFuseCodegen::scheduleInsts() {
    inst_sequence_.clear();

    // 扫描loop_code_中的向量指令, 进一步确定向量指令slot 
    determineVectorInstSlot();

    for (auto inst : prelogue_code_) {
        dispatchInst(inst);
    }

    for (int32_t i = 0; i < loop_code_.size(); ++i) {
        auto inst = loop_code_.at(i);
        dispatchInst(inst, i == loop_code_.size() - 1);
    }
}

std::vector<uint32_t> TmuFuseCodegen::toBinary() const {
    std::vector<uint32_t> binary;
    for (auto& inst_bundle : inst_sequence_) {
        for (auto& inst : inst_bundle) {
            auto raw_ = inst.raw;
            binary.push_back(raw_);
        }
    }

    return binary;
}

std::string TmuFuseCodegen::toAssembly() const {
    std::string assembly;
    for (auto& inst_bundle : inst_sequence_) {
        auto inst0 = inst_bundle.at(0);
        auto inst1 = inst_bundle.at(1);

        std::string inst0_str = inst0.assembly;
        std::string inst1_str = inst1.assembly;
        inst0_str.resize(50, ' ');
        inst1_str.resize(50, ' ');
        assembly += inst1_str + " | " + inst0_str + "\n";
    }

    return assembly;
}

} // namespace xp_tmu
} // namespace xp_mlir


