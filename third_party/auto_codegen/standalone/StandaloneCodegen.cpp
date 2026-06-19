#include "standalone/StandaloneCodegen.hpp"
#include "mlir/IR/Operation.h"
#include "standalone/RegManager.hpp"
#include "standalone/RvvInstGen.hpp"
#include "standalone/com_utils.hpp"
#include <cstdint>
#include <iostream>
#include <memory>

namespace xp_mlir {
namespace xp_std {
StandaloneCodegen::StandaloneCodegen(mlir::Operation* standalone_region_op) {
    standalone_pattern_ = std::make_shared<xp_std::StandalonePattern>(standalone_region_op);
    standalone_pattern_->parseStandalone();

    is_matmultc_ld_mode_ = standalone_pattern_->isMatmulTcLd();

    if (!is_matmultc_ld_mode_) {
        xvec_converter_ = std::make_unique<xp_std::XVectorConverter>(standalone_pattern_);
        xvec_converter_->convert();
    }
}

void StandaloneCodegen::codegen() {
    if (is_matmultc_ld_mode_) {
        genMatmulTcLdCode();
    } else {
        genOpsCode();
        xvec_converter_->destoryNewStandaloneOp();
    }

    scheduleInsts();
    std::cout << "after instruction scheduling: " << std::endl;
    std::cout << toAssembly() << std::endl;

    // reallocRegisters();
    // std::cout << "after register reallocation: " << std::endl;
    // std::cout << toAssembly() << std::endl;
}

void StandaloneCodegen::genMatmulTcLdCode() {
    assert(is_matmultc_ld_mode_ && "the standalone codegen is expected to be in matmulTcLd mode");

    auto all_ops = standalone_pattern_->getAllStandaloneRegionOps();
    auto all_op_infos = standalone_pattern_->getOp2StandaloneInfoTable();
    auto matmultc_ld_op = all_ops.front();
    auto matmultc_ld_op_info = all_op_infos.at(matmultc_ld_op);

    auto matmultc_ld_code = OpCodegen::codegen_matmulTcLd(matmultc_ld_op_info);
    loop_code_.insert(loop_code_.end(), matmultc_ld_code.begin(), matmultc_ld_code.end());
}

CodeLoc StandaloneCodegen::getOpLocInScfRegion(mlir::Operation* op) {
    assert(op && "operation cannot be null");
    auto scf_handler_ = xvec_converter_->getScfHandler();
    auto parent_op = op->getParentOp();
    if (parent_op == scf_handler_.inner_loop.getOperation()) {
        return CodeLoc::INNER_LOOP;
    } else if (parent_op == scf_handler_.outer_loop.getOperation()) {
        return CodeLoc::OUTER_LOOP;
    } else {
        assert(parent_op == scf_handler_.scf_region->getParentOp() && "the parent op of operation should be either inner loop, outer loop or scf region");
        return CodeLoc::PRELOGUE;
    }
}

void StandaloneCodegen::checkXvecOpSequence(const std::vector<mlir::Operation*> xvec_ops) {
    int32_t outer_loop_begin_index = -1, outer_loop_end_index = -1;
    int32_t inner_loop_begin_index = -1, inner_loop_end_index = -1;
    for (int32_t i = 0; i < xvec_ops.size(); ++i) {
        auto op = xvec_ops.at(i);
        if (auto loop_begin_op = llvm::dyn_cast<xvec::LoopBegin>(op)) {
            auto loop_id = loop_begin_op.getLoopId();
            if (loop_id == 0) {
                assert(outer_loop_begin_index == -1 && "there should be only one outer loop begin op in scf region");
                outer_loop_begin_index = i;
            } else if (loop_id == 1) {
                assert(inner_loop_begin_index == -1 && "there should be only one inner loop begin op in scf region");
                inner_loop_begin_index = i;
            } else {
                assert(false && "unsupported loop id for loop begin op");
            }
        } 
        
        if (auto loop_end_op = llvm::dyn_cast<xvec::LoopEnd>(op)) {
            auto loop_id = loop_end_op.getLoopId();
            if (loop_id == 0) {
                assert(outer_loop_end_index == -1 && "there should be only one outer loop end op in scf region");
                outer_loop_end_index = i;
            } else if (loop_id == 1) {
                assert(inner_loop_end_index == -1 && "there should be only one inner loop end op in scf region");
                inner_loop_end_index = i;
            } else {
                assert(false && "unsupported loop id for loop end op");
            }
        }
    }
    assert(outer_loop_begin_index != -1 && outer_loop_end_index != -1 && "there should be one outer loop in scf region");
    assert(inner_loop_begin_index != -1 && inner_loop_end_index != -1 && "there should be one inner loop in scf region");

    for (int32_t i = 0; i < xvec_ops.size(); ++i) {
        auto op_loc = getOpLocInScfRegion(xvec_ops.at(i));
        if (op_loc == CodeLoc::PRELOGUE) {
            assert(i < outer_loop_begin_index && "the operations before outer loop begin op should be in prelogue");
        } else if (op_loc == CodeLoc::OUTER_LOOP) {
            bool constraint_0 = (i >= outer_loop_begin_index && i < inner_loop_begin_index);
            bool constraint_1 = (i > inner_loop_end_index && i <= outer_loop_end_index);
            assert((constraint_0 || constraint_1) && "error outer loop op index");
        } else if (op_loc == CodeLoc::INNER_LOOP) {
            assert(i >= inner_loop_begin_index && i <= inner_loop_end_index && "the operations between inner loop begin op and inner loop end op should be in inner loop");
        } else {
            assert(false && "invalid operation location in scf region");
        }
    }
}

void StandaloneCodegen::genOpsCode() {
    auto scf_handler_ = xvec_converter_->getScfHandler();
    auto scf_region_op_ = scf_handler_.scf_region;

    std::vector<mlir::Operation*> all_xvec_ops;
    scf_region_op_->walk<WalkOrder::PreOrder>([&](mlir::Operation* op) {
        if (op->getDialect()->getNamespace() == "xvec") {
            all_xvec_ops.push_back(op);
        }
    });
    checkXvecOpSequence(all_xvec_ops);

    for (auto op : all_xvec_ops) {
        auto code_loc = getOpLocInScfRegion(op);
        OpCodegen codegen_op(op, code_loc, prelogue_code_, loop_code_);
        codegen_op.codegen();
    }
}

void StandaloneCodegen::dispatchInstBundle(std::vector<RvvInst>& inst_bundle) {
    assert(inst_bundle.size() == 2 && "the size of instruction bundle should be 2");
    auto a = inst_bundle.at(0);
    auto b = inst_bundle.at(1);
    if (RvvInst::isInstADependentOnInstB(b, a)) {
        auto nop_inst = RvvInstGenerator::genNopRvvInst();
        std::vector<RvvInst> inst_bundle_0 = {a, nop_inst};
        std::vector<RvvInst> inst_bundle_1 = {nop_inst, b};
        inst_sequence_.push_back(inst_bundle_0);
        inst_sequence_.push_back(inst_bundle_1);
    } else {
        inst_sequence_.push_back(inst_bundle);
    }

    inst_bundle.clear();
}

void StandaloneCodegen::packExclusiveOneBundleInst(const RvvInst& inst, std::vector<RvvInst>& inst_bundle, VeuInstSlotSupportion& cur_slot) {
    // 新启动一个bundle
    auto nop_inst = RvvInstGenerator::genNopRvvInst();
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

void StandaloneCodegen::dispatchInst(const RvvInst& inst, bool is_last) {
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
        auto nop_inst = RvvInstGenerator::genNopRvvInst();
        inst_bundle.push_back(nop_inst);
        inst_bundle.push_back(inst);
        dispatchInstBundle(inst_bundle);
        cur_slot = VeuInstSlotSupportion::SLOT_0;
    } else if (cur_slot == VeuInstSlotSupportion::SLOT_1) {
        auto nop_inst = RvvInstGenerator::genNopRvvInst();
        inst_bundle.push_back(nop_inst);
        dispatchInstBundle(inst_bundle);
        inst_bundle.push_back(inst);
        cur_slot = VeuInstSlotSupportion::SLOT_1;
    } else {
        throw std::runtime_error("invalid current slot type during instruction dispatch");
    }

    if (is_last) {
        if (cur_slot == VeuInstSlotSupportion::SLOT_1) {
            auto nop_inst = RvvInstGenerator::genNopRvvInst();
            inst_bundle.push_back(nop_inst);
            dispatchInstBundle(inst_bundle);
        }
    }
}

void StandaloneCodegen::scheduleInsts() {
    inst_sequence_.clear();

    for (auto inst : prelogue_code_) {
        dispatchInst(inst);
    }

    for (int32_t i = 0; i < loop_code_.size(); ++i) {
        auto inst = loop_code_.at(i);
        dispatchInst(inst, i == loop_code_.size() - 1);
    }
    
    setRvvInstLoc();
}

std::vector<uint32_t> StandaloneCodegen::toBinary() const {
    std::vector<uint32_t> binary;
    for (auto& inst_bundle : inst_sequence_) {
        for (auto& inst : inst_bundle) {
            auto raw_ = inst.raw;
            binary.push_back(raw_);
        }
    }

    return binary;
}

std::string StandaloneCodegen::toAssembly() const {
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

void StandaloneCodegen::setRvvInstLoc() {
    int32_t cur_scope_id = 0; // prelogue: 0, outer loop: 1, inner loop: 2
    for (int32_t i_line = 0; i_line < inst_sequence_.size(); ++i_line) {
        auto& inst_bundle = inst_sequence_.at(i_line);
        for (int32_t i_slot = 0; i_slot < inst_bundle.size(); ++i_slot) {
            auto& inst = inst_bundle.at(i_slot);
            inst.loc.line_id = i_line;
            inst.loc.slot_id = i_slot;

            if (RvvInst::isOuterLoopBegin(inst)) {
                // 进入outer loop
                inst.loc.scope_id = 1;
                cur_scope_id = 1;
                continue;
            } 
            if (RvvInst::isOuterLoopEnd(inst)) {
                // 离开outer loop回到prelogue
                inst.loc.scope_id = 1;
                cur_scope_id = 0;
                continue;
            } 
            if (RvvInst::isInnerLoopBegin(inst)) {
                // 进入inner loop
                inst.loc.scope_id = 2;
                cur_scope_id = 2;
                continue;
            } 
            if (RvvInst::isInnerLoopEnd(inst)) {
                // 离开inner loop回到outer loop
                inst.loc.scope_id = 2;
                cur_scope_id = 1;
                continue;
            }

            inst.loc.scope_id = cur_scope_id;
        }
    }
}

struct RegisterLiveness {
    int32_t start_line;
    int32_t end_line;
};

class RegReuse {
public:
    RegReuse(std::vector<std::vector<RvvInst>>& inst_sequence) : origin_inst_sequence(inst_sequence) {
        genRegDefUseInfo();
        genRegLiveness();
    }

    void genRegDefUseInfo() {
        outer_loop_begin_line = -1;
        outer_loop_end_line = -1;
        inner_loop_begin_line = -1;
        inner_loop_end_line = -1;

        // 记录每个寄存器的定义点和使用点
        for (int32_t i = 0; i < origin_inst_sequence.size(); ++i) {
            auto inst_bundle = origin_inst_sequence.at(i);
            for (auto& inst : inst_bundle) {
                if (RvvInst::isNopRvvInst(inst)) continue;

                if (RvvInst::isOuterLoopBegin(inst)) {
                    outer_loop_begin_line = i;
                }
                if (RvvInst::isOuterLoopEnd(inst)) {
                    outer_loop_end_line = i;
                }
                if (RvvInst::isInnerLoopBegin(inst)) {
                    inner_loop_begin_line = i;
                }
                if (RvvInst::isInnerLoopEnd(inst)) {
                    inner_loop_end_line = i;
                }

                for (auto input : inst.inputs) {
                    if (input.type == RvvOperandType::XREG) {
                        reg_use_table[input.type][input.value.x_reg.id].push_back(inst);
                        continue;
                    }

                    if (input.type == RvvOperandType::VREG) {
                        reg_use_table[input.type][input.value.v_reg.id].push_back(inst);
                        continue;
                    }
                }

                {
                    auto output = inst.output;
                    if (output.type == RvvOperandType::XREG) {
                        reg_def_table[output.type][output.value.x_reg.id].push_back(inst);
                        continue;
                    }

                    if (output.type == RvvOperandType::VREG) {
                        reg_def_table[output.type][output.value.v_reg.id].push_back(inst);
                        continue;
                    }
                }
            }
        }

        // 有def, 但无use处理
        for (auto & [reg_type, reg_defs] : reg_def_table) {
            for (auto& [reg_id, def_insts] : reg_defs) {
                if (reg_use_table.at(reg_type).count(reg_id) == 0) {
                    reg_use_table[reg_type][reg_id] = {};
                }
            }
        }

        assert(outer_loop_begin_line != -1 && outer_loop_end_line != -1 && "there should be one outer loop in scf region");
        assert(inner_loop_begin_line != -1 && inner_loop_end_line != -1 && "there should be one inner loop in scf region");
    }

    void genRegLiveness() {
        for (auto & [reg_type, reg_defs] : reg_def_table) {
            for (auto& [reg_id, def_insts] : reg_defs) {
                // start
                auto first_def_inst = def_insts.front();
                reg_liveness_table[reg_type][reg_id].start_line = first_def_inst.loc.line_id;

                // end
                auto use_insts = reg_use_table.at(reg_type).at(reg_id);
                if (use_insts.empty()) {
                    reg_liveness_table[reg_type][reg_id].end_line = first_def_inst.loc.line_id;
                    continue;
                }

                auto last_use_inst = use_insts.back();
                assert(last_use_inst.loc.scope_id >= first_def_inst.loc.scope_id && last_use_inst.loc.line_id > first_def_inst.loc.line_id && 
                         "the last use instruction should be after the first def instruction");
                if (last_use_inst.loc.scope_id > first_def_inst.loc.scope_id) {
                    if (first_def_inst.loc.scope_id == 0) {
                        // prelogue
                        reg_liveness_table[reg_type][reg_id].end_line = outer_loop_end_line;
                    } else if (first_def_inst.loc.scope_id == 1) {
                        // outer loop
                        reg_liveness_table[reg_type][reg_id].end_line = inner_loop_end_line;
                    } else {
                        throw std::runtime_error("invalid scope id for instruction");
                    }
                } else {
                    // 同一scope内，直接取最后一个use instruction的line id
                    reg_liveness_table[reg_type][reg_id].end_line = last_use_inst.loc.line_id;
                }
            }
        }
    }

    int32_t getReuseReg(RvvOperandType reg_type, int32_t logical_reg_id) {
        if (reg_type == RvvOperandType::XREG) {
            assert(RegManager::get().isLogicalScalarReg(logical_reg_id) && "the input reg should be a logical scalar register");
        } else if (reg_type == RvvOperandType::VREG) {
            assert(RegManager::get().isLogicalVectorReg(logical_reg_id) && "the input reg should be a logical vector register");
        } else {
            throw std::runtime_error("unsupported register type");
        }

        auto liveness_table = reg_liveness_table.at(reg_type);
        auto logical_reg_liveness = liveness_table.at(logical_reg_id);

        for (const auto& [candidate_reg_id, candidate_reg_liveness] : liveness_table) {
            if (reg_type == RvvOperandType::XREG) {
                if (RegManager::get().isLogicalScalarReg(candidate_reg_id) || RegManager::get().isSpecificScalarReg(candidate_reg_id)) {
                    continue;
                }
            }
            if (reg_type == RvvOperandType::VREG) {
                if (RegManager::get().isLogicalVectorReg(candidate_reg_id) || RegManager::get().isSpecificVectorReg(candidate_reg_id)) {
                    continue;
                }
            }

            // 满足重用条件：candidate register的liveness与待重用的logical register的liveness不重叠
            bool can_reuse = (candidate_reg_liveness.end_line < logical_reg_liveness.start_line) || 
                             (candidate_reg_liveness.start_line > logical_reg_liveness.end_line);
            if (can_reuse) {
                return candidate_reg_id;
            }
        }

        assert(false && "cannot find a reusable register that satisfies the reuse conditions");
        return -1;
    }

    void updateInstReg(RvvOperandType reg_type, int32_t logical_reg_id, int32_t pysical_reuse_reg_id) {
        auto def_insts = reg_def_table.at(reg_type).at(logical_reg_id);
        auto use_insts = reg_use_table.at(reg_type).at(logical_reg_id);
        
        for (const auto& def_inst : def_insts) {
            auto inst_line_id = def_inst.loc.line_id;
            auto inst_slot_id = def_inst.loc.slot_id;
            auto& to_update_inst = origin_inst_sequence.at(inst_line_id).at(inst_slot_id);
            assert(!RvvInst::isNopRvvInst(to_update_inst) && "the instruction to update should not be a nop instruction");

            auto& inst_output = to_update_inst.output;
            if (inst_output.type == reg_type) {
                if (reg_type == RvvOperandType::XREG && inst_output.value.x_reg.id == logical_reg_id) {
                    inst_output.value.x_reg.id = pysical_reuse_reg_id;
                } else if (reg_type == RvvOperandType::VREG && inst_output.value.v_reg.id == logical_reg_id) {
                    inst_output.value.v_reg.id = pysical_reuse_reg_id;
                }
            }
            to_update_inst.updateCode();
        }

        for (const auto& use_inst : use_insts) {
            auto inst_line_id = use_inst.loc.line_id;
            auto inst_slot_id = use_inst.loc.slot_id;
            auto& to_update_inst = origin_inst_sequence.at(inst_line_id).at(inst_slot_id);
            assert(!RvvInst::isNopRvvInst(to_update_inst) && "the instruction to update should not be a nop instruction");

            auto& inst_inputs = to_update_inst.inputs;
            for (auto& input : inst_inputs) {
                if (input.type == reg_type) {
                    if (reg_type == RvvOperandType::XREG && input.value.x_reg.id == logical_reg_id) {
                        input.value.x_reg.id = pysical_reuse_reg_id;
                    } else if (reg_type == RvvOperandType::VREG && input.value.v_reg.id == logical_reg_id) {
                        input.value.v_reg.id = pysical_reuse_reg_id;
                    }
                }
            }
            to_update_inst.updateCode();
        }
    }

    void realloc() {
        for (auto & [reg_type, reg_defs] : reg_def_table) {
            for (auto& [logical_reg_id, def_insts] : reg_defs) {
                if (reg_type == RvvOperandType::XREG) {
                    if (!RegManager::get().isLogicalScalarReg(logical_reg_id)) {
                        continue;
                    }
                } else if (reg_type == RvvOperandType::VREG) {
                    if (!RegManager::get().isLogicalVectorReg(logical_reg_id)) {
                        continue;
                    }
                } else {
                    throw std::runtime_error("unsupported register type");
                }

                // 分配物理register
                auto pysical_reuse_reg_id = getReuseReg(reg_type, logical_reg_id);
                // 更新指令寄存器和指令编码
                updateInstReg(reg_type, logical_reg_id, pysical_reuse_reg_id);
            }
        }
    }

private:
    std::vector<std::vector<RvvInst>>& origin_inst_sequence;

    int32_t outer_loop_begin_line = -1;
    int32_t outer_loop_end_line = -1;
    int32_t inner_loop_begin_line = -1;
    int32_t inner_loop_end_line = -1;

    std::map<RvvOperandType, std::map<int32_t, std::vector<RvvInst>>> reg_def_table; // {reg_type, {reg_id, the instructions that defines this reg}}
    std::map<RvvOperandType, std::map<int32_t, std::vector<RvvInst>>> reg_use_table; // {reg_type, {reg_id, the instructions that uses this reg}}
    std::map<RvvOperandType, std::map<int32_t, RegisterLiveness>> reg_liveness_table; // {reg_id, liveness}
};

void StandaloneCodegen::reallocRegisters() {
    RegReuse reg_reuse(inst_sequence_);
    reg_reuse.realloc();
}

} // namespace xp_std
} // namespace xp_mlir


