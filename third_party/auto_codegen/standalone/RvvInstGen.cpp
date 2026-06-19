#include "standalone/RvvInstGen.hpp"
#include "mlir/IR/OperationSupport.h"
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <stdexcept>
#include <map>
#include <set>
#include <cmath>

namespace xp_mlir {
namespace xp_std {
bool RvvInst::isExclusiveOneBundleInst(const RvvInst& inst) {
    return inst.name == "loop_cfgx" || 
           inst.name == "loop_end" ||
           inst.name == "vsetvli";
}

bool RvvInst::isNopRvvInst(const RvvInst& inst) {
    return inst.assembly == "nop";
}

bool RvvInst::isOuterLoopBegin(const RvvInst& inst) {
    if (inst.name == "loop_cfgx") {
        assert(inst.inputs.at(1).type == RvvOperandType::IMM && "the 2nd input of loop_cfgx instruction should be an immediate operand representing the loop id");
        auto loop_id = inst.inputs.at(1).value.imm;
        return loop_id == 0;
    }
    
    return false;
}

bool RvvInst::isOuterLoopEnd(const RvvInst& inst) {
    if (inst.name == "loop_end") {
        assert(inst.inputs.at(0).type == RvvOperandType::IMM && "the 1st input of loop_end instruction should be an immediate operand representing the loop id");
        auto loop_id = inst.inputs.at(0).value.imm;
        return loop_id == 0;
    }
    
    return false;
}

bool RvvInst::isInnerLoopBegin(const RvvInst& inst) {
    if (inst.name == "loop_cfgx") {
        assert(inst.inputs.at(1).type == RvvOperandType::IMM && "the 2nd input of loop_cfgx instruction should be an immediate operand representing the loop id");
        auto loop_id = inst.inputs.at(1).value.imm;
        return loop_id == 1;
    }
    
    return false;
}

bool RvvInst::isInnerLoopEnd(const RvvInst& inst) {
    if (inst.name == "loop_end") {
        assert(inst.inputs.at(0).type == RvvOperandType::IMM && "the 1st input of loop_end instruction should be an immediate operand representing the loop id");
        auto loop_id = inst.inputs.at(0).value.imm;
        return loop_id == 1;
    }
    
    return false;
}

bool RvvInst::isInstADependentOnInstB(const RvvInst& a, const RvvInst& b) {
    auto b_output = b.output;
    for (const auto& a_input : a.inputs) {
        if (a_input == b_output) {
            return true;
        }
    }
    return false;
}

void RvvInst::updateCode() {
    RvvInstGenerator generator(name);
    auto new_inst = generator.codegen(inputs, output);
    raw = new_inst.raw;
    assembly = new_inst.assembly;
}

bool RvvInst::isTwoInstsHasDependency(const RvvInst& a, const RvvInst& b) {
    return isInstADependentOnInstB(a, b) || isInstADependentOnInstB(b, a);
}

bool CodegenOpOperand::operator==(const CodegenOpOperand& other) const {
    if (type != other.type) {
        return false;
    }
    switch (type) {
        case RvvOperandType::IMM:
            return value.imm == other.value.imm;
        case RvvOperandType::XREG:
            return value.x_reg.id == other.value.x_reg.id;
        case RvvOperandType::VREG:
            return value.v_reg.id == other.value.v_reg.id;
        case RvvOperandType::VM:
            return true;
        case RvvOperandType::NONE:
            // 因为如果inst a的input含有None, inst b的output为None, 
            // 如果认为它们相等, 则会导致inst a被认为依赖于inst b, 这显然是不合理的
            return false;
        default:
            assert(false && "Unsupported RvvOperandType in equality comparison");
            return false;
    }
}

using rvv_inst_gen_func_t = std::function<uint32_t(const std::vector<uint32_t>&)>;

uint32_t gen_lui(const std::vector<uint32_t> &args) {
    return (args[1] & 0xfffff000) + (args[0] << 7) + 0x37;
};
uint32_t gen_addi(const std::vector<uint32_t> &args) {
    return ((args[2] & 0xfff) << 20) + (args[1] << 15) + (args[0] << 7) + 0x13;
};
uint32_t gen_xori(const std::vector<uint32_t> &args) {
    return ((args[2] & 0xfff) << 20) + (args[1] << 15) + (0x4 << 12) + (args[0] << 7) + 0x13;
};
uint32_t gen_ori(const std::vector<uint32_t> &args) {
    return ((args[2] & 0xfff) << 20) + (args[1] << 15) + (0x6 << 12) + (args[0] << 7) + 0x13;
};
uint32_t gen_andi(const std::vector<uint32_t> &args) {
    return ((args[2] & 0xfff) << 20) + (args[1] << 15) + (0x7 << 12) + (args[0] << 7) + 0x13;
};
uint32_t gen_slli(const std::vector<uint32_t> &args) {
    return (args[2] << 20) + (args[1] << 15) + (0x1 << 12) + (args[0] << 7) + 0x13;
};
uint32_t gen_srli(const std::vector<uint32_t> &args) {
    return (args[2] << 20) + (args[1] << 15) + (0x5 << 12) + (args[0] << 7) + 0x13;
};
uint32_t gen_srai(const std::vector<uint32_t> &args) {
    return (0x20 << 25) + (args[2] << 20) + (args[1] << 15) + (0x5 << 12) + (args[0] << 7) + 0x13;
};
uint32_t gen_add(const std::vector<uint32_t> &args) {
    return (args[2] << 20) + (args[1] << 15) + (args[0] << 7) + 0x33;
};
uint32_t gen_sub(const std::vector<uint32_t> &args) {
    return (0x20 << 25) + (args[2] << 20) + (args[1] << 15) + (args[0] << 7) + 0x33;
};
uint32_t gen_sll(const std::vector<uint32_t> &args) {
    return (args[2] << 20) + (args[1] << 15) + (0x1 << 12) + (args[0] << 7) + 0x33;
};
uint32_t gen_slt(const std::vector<uint32_t> &args) {
    return (args[2] << 20) + (args[1] << 15) + (0x2 << 12) + (args[0] << 7) + 0x33;
};
uint32_t gen_xor(const std::vector<uint32_t> &args) {
    return (args[2] << 20) + (args[1] << 15) + (0x4 << 12) + (args[0] << 7) + 0x33;
};
uint32_t gen_srl(const std::vector<uint32_t> &args) {
    return (args[2] << 20) + (args[1] << 15) + (0x5 << 12) + (args[0] << 7) + 0x33;
};
uint32_t gen_sra(const std::vector<uint32_t> &args) {
    return (0x20 << 25) + (args[2] << 20) + (args[1] << 15) + (0x5 << 12) + (args[0] << 7) + 0x33;
};
uint32_t gen_or(const std::vector<uint32_t> &args) {
    return (args[2] << 20) + (args[1] << 15) + (0x6 << 12) + (args[0] << 7) + 0x33;
};
uint32_t gen_and(const std::vector<uint32_t> &args) {
    return (args[2] << 20) + (args[1] << 15) + (0x7 << 12) + (args[0] << 7) + 0x33;
};
uint32_t gen_mul(const std::vector<uint32_t> &args) {
    return (0x1 << 25) + (args[2] << 20) + (args[1] << 15) + (args[0] << 7) + 0x33;
};
uint32_t gen_mulh(const std::vector<uint32_t> &args) {
    return (0x1 << 25) + (args[2] << 20) + (args[1] << 15) + (0x1 << 12) + (args[0] << 7) + 0x33;
};
uint32_t gen_lb(const std::vector<uint32_t> &args) {
    return ((args[1] & 0xfff) << 20) + (args[2] << 15) + (args[0] << 7) + 0x3;
};
uint32_t gen_lh(const std::vector<uint32_t> &args) {
    return ((args[1] & 0xfff) << 20) + (args[2] << 15) + (0x1 << 12) + (args[0] << 7) + 0x3;
};
uint32_t gen_lw(const std::vector<uint32_t> &args) {
    return ((args[1] & 0xfff) << 20) + (args[2] << 15) + (0x2 << 12) + (args[0] << 7) + 0x3;
};
uint32_t gen_sb(const std::vector<uint32_t> &args) {
    return ((args[1] & 0xfe0) << 20) + (args[0] << 20) + (args[2] << 15) + ((args[1] & 0x1f) << 7) + 0x23;
};
uint32_t gen_sh(const std::vector<uint32_t> &args) {
    return ((args[1] & 0xfe0) << 20) + (args[0] << 20) + (args[2] << 15) + (0x1 << 12) + ((args[1] & 0x1f) << 7) + 0x23;
};
uint32_t gen_sw(const std::vector<uint32_t> &args) {
    return ((args[1] & 0xfe0) << 20) + (args[0] << 20) + (args[2] << 15) + (0x2 << 12) + ((args[1] & 0x1f) << 7) + 0x23;
};
uint32_t gen_vsadd_vv(const std::vector<uint32_t> &args) {
    return (0x21 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (args[0] << 7) + 0x57;
};
uint32_t gen_vsadd_vx(const std::vector<uint32_t> &args) {
    return (0x21 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (0x4 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vsadd_vi(const std::vector<uint32_t> &args) {
    return (0x21 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (0x3 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vssub_vv(const std::vector<uint32_t> &args) {
    return (0x23 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (args[0] << 7) + 0x57;
};
uint32_t gen_vssub_vx(const std::vector<uint32_t> &args) {
    return (0x23 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[2] << 20) + (args[3] << 15) + (0x4 << 12) + (args[1] << 7) + 0x57;
};
uint32_t gen_vdiv_vv(const std::vector<uint32_t> &args) {
    return (0x21 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (0x2 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vdiv_vx(const std::vector<uint32_t> &args) {
    return (0x21 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (0x6 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmax_vv(const std::vector<uint32_t> &args) {
    return (0x7 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmax_vx(const std::vector<uint32_t> &args) {
    return (0x7 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (0x4 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmin_vv(const std::vector<uint32_t> &args) {
    return (0x5 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmin_vx(const std::vector<uint32_t> &args) {
    return (0x5 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (0x4 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vsll_vv(const std::vector<uint32_t> &args) {
    return (0x25 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (args[0] << 7) + 0x57;
};
uint32_t gen_vsll_vx(const std::vector<uint32_t> &args) {
    return (0x25 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (0x4 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vsll_vi(const std::vector<uint32_t> &args) {
    return (0x25 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (0x3 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vsrl_vv(const std::vector<uint32_t> &args) {
    return (0x28 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (args[0] << 7) + 0x57;
};
uint32_t gen_vsrl_vx(const std::vector<uint32_t> &args) {
    return (0x28 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (0x4 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vsrl_vi(const std::vector<uint32_t> &args) {
    return (0x28 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (0x3 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vsra_vv(const std::vector<uint32_t> &args) {
    return (0x29 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (args[0] << 7) + 0x57;
};
uint32_t gen_vsra_vx(const std::vector<uint32_t> &args) {
    return (0x29 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (0x4 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vsra_vi(const std::vector<uint32_t> &args) {
    return (0x29 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (0x3 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vredsum_vs(const std::vector<uint32_t> &args) {
    return (args[3] << 25) + ((args.size() >= 4 ? args[3] : 1) << 20) + (args[2] << 15) + (0x2 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vredmax_vs(const std::vector<uint32_t> &args) {
    return (0x7 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (0x2 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vredmin_vs(const std::vector<uint32_t> &args) {
    return (0x5 << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (0x2 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmv_v_v(const std::vector<uint32_t> &args) {
    return (0x17 << 26) + (0x1 << 25) + (args[1] << 15) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmv_v_x(const std::vector<uint32_t> &args) {
    return (0x17 << 26) + (0x1 << 25) + (args[1] << 15) + (0x4 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmv_v_i(const std::vector<uint32_t> &args) {
    return (0x17 << 26) + (0x1 << 25) + (args[1] << 15) + (0x3 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmv_x_s(const std::vector<uint32_t> &args) {
    return (0x10 << 26) + (0x1 << 25) + (args[1] << 20) + (0x2 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmv_s_x(const std::vector<uint32_t> &args) {
    return (0x10 << 26) + (0x1 << 25) + (args[1] << 15) + (0x6 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmv1r_v(const std::vector<uint32_t> &args) {
    return (0x27 << 26) + (0x1 << 25) + (args[1] << 20) + (0x3 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmerge_vvm(const std::vector<uint32_t> &args) {
    return (0x17 << 26) + (args[1] << 20) + (args[2] << 15) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmerge_vxm(const std::vector<uint32_t> &args) {
    return (0x17 << 26) + (args[1] << 20) + (args[2] << 15) + (0x4 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmerge_vim(const std::vector<uint32_t> &args) {
    return (0x17 << 26) + (args[1] << 20) + (args[2] << 15) + (0x3 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vsext_vf4(const std::vector<uint32_t> &args) {
    return (0x12 << 26) + ((args.size() >= 3 ? args[2] : 1) << 25) + (args[1] << 20) + (0x5 << 15) + (0x2 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vsext_vf2(const std::vector<uint32_t> &args) {
    return (0x12 << 26) + ((args.size() >= 3 ? args[2] : 1) << 25) + (args[1] << 20) + (0x7 << 15) + (0x2 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vslideup_vx(const std::vector<uint32_t> &args) {
    return (0xe << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (0x4 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vslideup_vi(const std::vector<uint32_t> &args) {
    return (0xe << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (0x3 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vslidedown_vx(const std::vector<uint32_t> &args) {
    return (0xf << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (0x4 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vslidedown_vi(const std::vector<uint32_t> &args) {
    return (0xf << 26) + ((args.size() >= 4 ? args[3] : 1) << 25) + (args[1] << 20) + (args[2] << 15) + (0x3 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vcompress_vm(const std::vector<uint32_t> &args) {
    return (0x17 << 26) + (0x1 << 25) + (args[1] << 20) + (args[2] << 15) + (0x2 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vle8_v(const std::vector<uint32_t> &args) {
    return (0x1 << 25) + (args[1] << 15) + (args[0] << 7) + 0x7;
};
uint32_t gen_vle16_v(const std::vector<uint32_t> &args) {
    return (0x1 << 25) + (args[1] << 15) + (0x5 << 12) + (args[0] << 7) + 0x7;
};
uint32_t gen_vle32_v(const std::vector<uint32_t> &args) {
    return (0x1 << 25) + (args[1] << 15) + (0x6 << 12) + (args[0] << 7) + 0x7;
};
uint32_t gen_vse8_v(const std::vector<uint32_t> &args) {
    return (0x1 << 25) + (args[1] << 15) + (args[0] << 7) + 0x27;
};
uint32_t gen_vse16_v(const std::vector<uint32_t> &args) {
    return (0x1 << 25) + (args[1] << 15) + (0x5 << 12) + (args[0] << 7) + 0x27;
};
uint32_t gen_vse32_v(const std::vector<uint32_t> &args) {
    return (0x1 << 25) + (args[1] << 15) + (0x6 << 12) + (args[0] << 7) + 0x27;
};
uint32_t gen_vlm_v(const std::vector<uint32_t> &args) {
    return (0x1 << 25) + (0xb << 20) + (args[1] << 15) + (args[0] << 7) + 0x7;
};
uint32_t gen_vsm_v(const std::vector<uint32_t> &args) {
    return (0x1 << 25) + (0xb << 20) + (args[1] << 15) + (args[0] << 7) + 0x27;
};
uint32_t gen_vsetvli(const std::vector<uint32_t> &raw_args) {
    // 只开放rd, rs1, vtype固定为e32 = 32, m0 = 0, ta = 1, ma = 1
    assert(raw_args.size() == 2 && "vsetvli should have exactly 2 arguments");
    assert(raw_args[0] == 30 && raw_args[1] == 15 && "for vsetvli, rd should be x30 and rs1 should be x15");

    std::vector<uint32_t> args = raw_args;
    args.push_back(32); // vtype: e32
    args.push_back(0);  // m0
    args.push_back(1);  // ta = 1
    args.push_back(1);  // ma = 1
    return (args[5] << 27) + (args[4] << 26) + (static_cast<int>(std::log2(args[2]/8)) << 23) + (args[3] << 20) + (args[1] << 15) + (0x7 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vsetivli(const std::vector<uint32_t> &raw_args) {
    // 只开放rd, rs1, vtype固定为e32 = 32, m1 = 1, ta = 1, ma = 1
    assert(raw_args.size() == 2 && "vsetvli should have exactly 2 arguments");
    assert(raw_args[0] == 30 && raw_args[1] == 15 && "for vsetivli, rd should be x30 and rs1 should be x15");

    std::vector<uint32_t> args = raw_args;
    args.push_back(32); // vtype: e32
    args.push_back(1);  // m1
    args.push_back(1);  // ta = 1
    args.push_back(1);  // ma = 1
    return (0x1 << 31) + (0x1 << 30) + (args[5] << 27) + (args[4] << 26) + (static_cast<int>(std::log2(args[2]/8)) << 23) + (args[3] << 20) + ((args[1] & 0x1f) << 15) + (0x7 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vcpop_m(const std::vector<uint32_t> &args) {
    return (0x10 << 26) + (0x1 << 25) + (args[1] << 20) + (0x10 << 15) + (0x2 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmseq_vv(const std::vector<uint32_t> &args) {
    return (0x18 << 26) + (args[3] << 25) + (args[1] << 20) + (args[2] << 15) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmseq_vx(const std::vector<uint32_t> &args) {
    return (0x18 << 26) + (args[3] << 25) + (args[1] << 20) + (args[2] << 15) + (0x4 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmseq_vi(const std::vector<uint32_t> &args) {
    return (0x18 << 26) + (args[3] << 25) + (args[1] << 20) + (args[2] << 15) + (0x3 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmslt_vv(const std::vector<uint32_t> &args) {
    return (0x1b << 26) + (args[3] << 25) + (args[1] << 20) + (args[2] << 15) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmslt_vx(const std::vector<uint32_t> &args) {
    return (0x1b << 26) + (args[3] << 25) + (args[1] << 20) + (args[2] << 15) + (0x4 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmsgt_vx(const std::vector<uint32_t> &args) {
    return (0x1f << 26) + (args[3] << 25) + (args[1] << 20) + (args[2] << 15) + (0x4 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vmsgt_vi(const std::vector<uint32_t> &args) {
    return (0x1f << 26) + (args[3] << 25) + (args[1] << 20) + (args[2] << 15) + (0x3 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vor_vv(const std::vector<uint32_t> &args) {
    return (0xa << 26) + (args[3] << 25) + (args[1] << 20) + (args[2] << 15) + (args[0] << 7) + 0x57;
};
uint32_t gen_vor_vx(const std::vector<uint32_t> &args) {
    return (0xa << 26) + (args[3] << 25) + (args[1] << 20) + (args[2] << 15) + (0x4 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vor_vi(const std::vector<uint32_t> &args) {
    return (0xa << 26) + (args[3] << 25) + (args[1] << 20) + (args[2] << 15) + (0x3 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vand_vv(const std::vector<uint32_t> &args) {
    return (0x9 << 26) + (args[3] << 25) + (args[1] << 20) + (args[2] << 15) + (args[0] << 7) + 0x57;
};
uint32_t gen_vand_vx(const std::vector<uint32_t> &args) {
    return (0x9 << 26) + (args[3] << 25) + (args[1] << 20) + (args[2] << 15) + (0x4 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vand_vi(const std::vector<uint32_t> &args) {
    return (0x9 << 26) + (args[3] << 25) + (args[1] << 20) + (args[2] << 15) + (0x3 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vxor_vv(const std::vector<uint32_t> &args) {
    return (0xb << 26) + (args[3] << 25) + (args[1] << 20) + (args[2] << 15) + (args[0] << 7) + 0x57;
};
uint32_t gen_vxor_vx(const std::vector<uint32_t> &args) {
    return (0xb << 26) + (args[3] << 25) + (args[1] << 20) + (args[2] << 15) + (0x4 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_vxor_vi(const std::vector<uint32_t> &args) {
    return (0xb << 26) + (args[3] << 25) + (args[1] << 20) + (args[2] << 15) + (0x3 << 12) + (args[0] << 7) + 0x57;
};
uint32_t gen_addiu(const std::vector<uint32_t> &args) {
    return ((args[2] & 0xfff) << 20) + (args[1] << 15) + (args[0] << 7) + 0xb;
};
uint32_t gen_vmul8_vv(const std::vector<uint32_t> &args) {
    return (args[1] << 20) + (args[2] << 15) + (0x1 << 12) + (args[0] << 7) + 0xb;
};
uint32_t gen_vmul8_vx(const std::vector<uint32_t> &args) {
    return (0x1 << 25) + (args[1] << 20) + (args[2] << 15) + (0x1 << 12) + (args[0] << 7) + 0xb;
};
uint32_t gen_vmul16_vv(const std::vector<uint32_t> &args) {
    return (0x2 << 25) + (args[1] << 20) + (args[2] << 15) + (0x1 << 12) + (args[0] << 7) + 0xb;
};
uint32_t gen_vmul16_vx(const std::vector<uint32_t> &args) {
    return (0x3 << 25) + (args[1] << 20) + (args[2] << 15) + (0x1 << 12) + (args[0] << 7) + 0xb;
};
uint32_t gen_vmul16_vf2(const std::vector<uint32_t> &args) {
    return (0x4 << 25) + (args[1] << 20) + (args[2] << 15) + (0x1 << 12) + (args[0] << 7) + 0xb;
};
uint32_t gen_vmul16_vxf2(const std::vector<uint32_t> &args) {
    return (0x5 << 25) + (args[1] << 20) + (args[2] << 15) + (0x1 << 12) + (args[0] << 7) + 0xb;
};
uint32_t gen_vmul16_xvf2(const std::vector<uint32_t> &args) {
    return (0x6 << 25) + (args[1] << 20) + (args[2] << 15) + (0x1 << 12) + (args[0] << 7) + 0xb;
};
uint32_t gen_vquant_i8ti32_vv(const std::vector<uint32_t> &args) {
    return (args[2] << 20) + (args[1] << 15) + (0x2 << 12) + (args[0] << 7) + 0xb;
};
uint32_t gen_vquant_i32ti8_vv(const std::vector<uint32_t> &args) {
    return (0x1 << 25) + (args[2] << 20) + (args[1] << 15) + (0x2 << 12) + (args[0] << 7) + 0xb;
};
uint32_t gen_vquant_i16ti32_vv(const std::vector<uint32_t> &args) {
    return (0x2 << 25) + (args[2] << 20) + (args[1] << 15) + (0x2 << 12) + (args[0] << 7) + 0xb;
};
uint32_t gen_vquant_i32ti16_vv(const std::vector<uint32_t> &args) {
    return (0x4 << 25) + (args[2] << 20) + (args[1] << 15) + (0x2 << 12) + (args[0] << 7) + 0xb;
};
uint32_t gen_vquant_i32ti10_vv(const std::vector<uint32_t> &args) {
    return (0x3 << 25) + (args[2] << 20) + (args[1] << 15) + (0x2 << 12) + (args[0] << 7) + 0xb;
};
uint32_t gen_load_lut_i8_d8(const std::vector<uint32_t> &args) {
    return (args[0] << 15) + (0x7 << 12) + 0x3;
};
uint32_t gen_load_lut_i8_d10(const std::vector<uint32_t> &args) {
    return (0x1 << 25) + (args[0] << 15) + (0x7 << 12) + 0x3;
};
uint32_t gen_load_lut_i10_d8(const std::vector<uint32_t> &args) {
    return (0x2 << 25) + (args[0] << 15) + (0x7 << 12) + 0x3;
};
uint32_t gen_load_lut_i10_d10(const std::vector<uint32_t> &args) {
    return (0x3 << 25) + (args[0] << 15) + (0x7 << 12) + 0x3;
};
uint32_t gen_vsfu_i8_d8_vv(const std::vector<uint32_t> &args) {
    return (args[2] << 20) + (args[1] << 15) + (0x3 << 12) + (args[0] << 7) + 0xb;
};
uint32_t gen_vsfu_i8_d10_vv(const std::vector<uint32_t> &args) {
    return (0x1 << 25) + (args[2] << 20) + (args[1] << 15) + (0x3 << 12) + (args[0] << 7) + 0xb;
};
uint32_t gen_vsfu_i10_d8_vv(const std::vector<uint32_t> &args) {
    return (0x2 << 25) + (args[2] << 20) + (args[1] << 15) + (0x3 << 12) + (args[0] << 7) + 0xb;
};
uint32_t gen_vsfu_i10_d10_vv(const std::vector<uint32_t> &args) {
    return (0x3 << 25) + (args[2] << 20) + (args[1] << 15) + (0x3 << 12) + (args[0] << 7) + 0xb;
};
uint32_t gen_loop_cfgi(const std::vector<uint32_t> &args) {
    return ((args[0] & 0x1ffff) << 15) + (0x3 << 12) + (args[1] << 7) + 0x63;
};
uint32_t gen_loop_cfgx(const std::vector<uint32_t> &args) {
    return (args[0] << 15) + (0x3 << 12) + (0x2 << 9) + (args[1] << 7) + 0x63;
};
uint32_t gen_loop_end(const std::vector<uint32_t> &args) {
    return (0x3 << 12) + (0x6 << 9) + (args[0] << 7) + 0x63;
};

rvv_inst_gen_func_t getRvvInstGenFunc(const std::string& inst_name) {
    static std::map<std::string, rvv_inst_gen_func_t> rvv_inst_gen_func_table = {
        {"lui", &gen_lui},
        {"addi", &gen_addi},
        {"xori", &gen_xori},
        {"ori", &gen_ori},
        {"andi", &gen_andi},
        {"slli", &gen_slli},
        {"srli", &gen_srli},
        {"srai", &gen_srai},
        {"add", &gen_add},
        {"sub", &gen_sub},
        {"sll", &gen_sll},
        {"slt", &gen_slt},
        {"xor", &gen_xor},
        {"srl", &gen_srl},
        {"sra", &gen_sra},
        {"or", &gen_or},
        {"and", &gen_and},
        {"mul", &gen_mul},
        {"mulh", &gen_mulh},
        {"lb", &gen_lb},
        {"lh", &gen_lh},
        {"lw", &gen_lw},
        {"sb", &gen_sb},
        {"sh", &gen_sh},
        {"sw", &gen_sw},
        {"vsadd.vv", &gen_vsadd_vv},
        {"vsadd.vx", &gen_vsadd_vx},
        {"vsadd.vi", &gen_vsadd_vi},
        {"vssub.vv", &gen_vssub_vv},
        {"vssub.vx", &gen_vssub_vx},
        {"vdiv.vv", &gen_vdiv_vv},
        {"vdiv.vx", &gen_vdiv_vx},
        {"vmax.vv", &gen_vmax_vv},
        {"vmax.vx", &gen_vmax_vx},
        {"vmin.vv", &gen_vmin_vv},
        {"vmin.vx", &gen_vmin_vx},
        {"vsll.vv", &gen_vsll_vv},
        {"vsll.vx", &gen_vsll_vx},
        {"vsll.vi", &gen_vsll_vi},
        {"vsrl.vv", &gen_vsrl_vv},
        {"vsrl.vx", &gen_vsrl_vx},
        {"vsrl.vi", &gen_vsrl_vi},
        {"vsra.vv", &gen_vsra_vv},
        {"vsra.vx", &gen_vsra_vx},
        {"vsra.vi", &gen_vsra_vi},
        {"vredsum.vs", &gen_vredsum_vs},
        {"vredmax.vs", &gen_vredmax_vs},
        {"vredmin.vs", &gen_vredmin_vs},
        {"vmv.v.v", &gen_vmv_v_v},
        {"vmv.v.x", &gen_vmv_v_x},
        {"vmv.v.i", &gen_vmv_v_i},
        {"vmv.x.s", &gen_vmv_x_s},
        {"vmv.s.x", &gen_vmv_s_x},
        {"vmv1r.v", &gen_vmv1r_v},
        {"vmerge.vvm", &gen_vmerge_vvm},
        {"vmerge.vxm", &gen_vmerge_vxm},
        {"vmerge.vim", &gen_vmerge_vim},
        {"vsext.vf4", &gen_vsext_vf4},
        {"vsext.vf2", &gen_vsext_vf2},
        {"vslideup.vx", &gen_vslideup_vx},
        {"vslideup.vi", &gen_vslideup_vi},
        {"vslidedown.vx", &gen_vslidedown_vx},
        {"vslidedown.vi", &gen_vslidedown_vi},
        {"vcompress.vm", &gen_vcompress_vm},
        {"vle8.v", &gen_vle8_v},
        {"vle16.v", &gen_vle16_v},
        {"vle32.v", &gen_vle32_v},
        {"vse8.v", &gen_vse8_v},
        {"vse16.v", &gen_vse16_v},
        {"vse32.v", &gen_vse32_v},
        {"vlm.v", &gen_vlm_v},
        {"vsm.v", &gen_vsm_v},
        {"vsetvli", &gen_vsetvli},
        {"vsetivli", &gen_vsetivli},
        {"vcpop.m", &gen_vcpop_m},
        {"vmseq.vv", &gen_vmseq_vv},
        {"vmseq.vx", &gen_vmseq_vx},
        {"vmseq.vi", &gen_vmseq_vi},
        {"vmslt.vv", &gen_vmslt_vv},
        {"vmslt.vx", &gen_vmslt_vx},
        {"vmsgt.vx", &gen_vmsgt_vx},
        {"vmsgt.vi", &gen_vmsgt_vi},
        {"vor.vv", &gen_vor_vv},
        {"vor.vx", &gen_vor_vx},
        {"vor.vi", &gen_vor_vi},
        {"vand.vv", &gen_vand_vv},
        {"vand.vx", &gen_vand_vx},
        {"vand.vi", &gen_vand_vi},
        {"vxor.vv", &gen_vxor_vv},
        {"vxor.vx", &gen_vxor_vx},
        {"vxor.vi", &gen_vxor_vi},
        {"addiu", &gen_addiu},
        {"vmul8.vv", &gen_vmul8_vv},
        {"vmul8.vx", &gen_vmul8_vx},
        {"vmul16.vv", &gen_vmul16_vv},
        {"vmul16.vx", &gen_vmul16_vx},
        {"vmul16.vf2", &gen_vmul16_vf2},
        {"vmul16.vxf2", &gen_vmul16_vxf2},
        {"vmul16.xvf2", &gen_vmul16_xvf2},
        {"vquant_i8ti32.vv", &gen_vquant_i8ti32_vv},
        {"vquant_i32ti8.vv", &gen_vquant_i32ti8_vv},
        {"vquant_i16ti32.vv", &gen_vquant_i16ti32_vv},
        {"vquant_i32ti16.vv", &gen_vquant_i32ti16_vv},
        {"vquant_i32ti10.vv", &gen_vquant_i32ti10_vv},
        {"load_lut_i8_d8", &gen_load_lut_i8_d8},
        {"load_lut_i8_d10", &gen_load_lut_i8_d10},
        {"load_lut_i10_d8", &gen_load_lut_i10_d8},
        {"load_lut_i10_d10", &gen_load_lut_i10_d10},
        {"vsfu_i8_d8.vv", &gen_vsfu_i8_d8_vv},
        {"vsfu_i8_d10.vv", &gen_vsfu_i8_d10_vv},
        {"vsfu_i10_d8.vv", &gen_vsfu_i10_d8_vv},
        {"vsfu_i10_d10.vv", &gen_vsfu_i10_d10_vv},
        {"loop_cfgi", &gen_loop_cfgi},
        {"loop_cfgx", &gen_loop_cfgx},
        {"loop_end", &gen_loop_end},
    };

    if (rvv_inst_gen_func_table.count(inst_name) == 0) {
        throw std::runtime_error("unsupported rvv instruction: " + inst_name);
    }

    auto func_ = rvv_inst_gen_func_table[inst_name];
    assert(func_ != nullptr && "function pointer should not be null");

    return func_;
}

std::vector<RvvOperandType> getRvvInstOperandTypes(const std::string& inst_name) {
    static std::map<std::string, std::vector<RvvOperandType>> rvv_inst_operand_types_table = {
        {"lui", {RvvOperandType::XREG, RvvOperandType::IMM}},
        {"addi", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::IMM}},
        {"xori", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::IMM}},
        {"ori", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::IMM}},
        {"andi", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::IMM}},
        {"slli", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::IMM}},
        {"srli", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::IMM}},
        {"srai", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::IMM}},
        {"add", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::XREG}},
        {"sub", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::XREG}},
        {"sll", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::XREG}},
        {"slt", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::XREG}},
        {"xor", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::XREG}},
        {"srl", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::XREG}},
        {"sra", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::XREG}},
        {"or", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::XREG}},
        {"and", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::XREG}},
        {"mul", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::XREG}},
        {"mulh", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::XREG}},
        {"lb", {RvvOperandType::XREG, RvvOperandType::IMM, RvvOperandType::XREG}},
        {"lh", {RvvOperandType::XREG, RvvOperandType::IMM, RvvOperandType::XREG}},
        {"lw", {RvvOperandType::XREG, RvvOperandType::IMM, RvvOperandType::XREG}},
        {"sb", {RvvOperandType::XREG, RvvOperandType::IMM, RvvOperandType::XREG}},
        {"sh", {RvvOperandType::XREG, RvvOperandType::IMM, RvvOperandType::XREG}},
        {"sw", {RvvOperandType::XREG, RvvOperandType::IMM, RvvOperandType::XREG}},
        {"vsadd.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VM}},
        {"vsadd.vx", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG, RvvOperandType::VM}},
        {"vsadd.vi", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::IMM, RvvOperandType::VM} },
        {"vssub.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VM}},
        {"vssub.vx", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG, RvvOperandType::VM}},
        {"vdiv.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VM}},
        {"vdiv.vx", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG, RvvOperandType::VM}},
        {"vmax.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VM}},
        {"vmax.vx", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG, RvvOperandType::VM}},
        {"vmin.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VM}},
        {"vmin.vx", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG, RvvOperandType::VM}},
        {"vsll.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VM}},
        {"vsll.vx", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG, RvvOperandType::VM}},
        {"vsll.vi", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::IMM, RvvOperandType::VM} },
        {"vsrl.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VM}},
        {"vsrl.vx", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG, RvvOperandType::VM}},
        {"vsrl.vi", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::IMM, RvvOperandType::VM} },
        {"vsra.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VM}},
        {"vsra.vx", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG, RvvOperandType::VM}},
        {"vsra.vi", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::IMM, RvvOperandType::VM} },
        {"vredsum.vs", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VM}},
        {"vredmax.vs", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VM}},
        {"vredmin.vs", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VM}},
        {"vmv.v.v", {RvvOperandType::VREG, RvvOperandType::VREG}},
        {"vmv.v.x", {RvvOperandType::VREG, RvvOperandType::XREG}},
        {"vmv.v.i", {RvvOperandType::VREG, RvvOperandType::IMM}},
        {"vmv.x.s", {RvvOperandType::XREG, RvvOperandType::VREG}},
        {"vmv.s.x", {RvvOperandType::VREG, RvvOperandType::XREG}},
        {"vmv1r.v", {RvvOperandType::VREG, RvvOperandType::VREG}},
        {"vmerge.vvm", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG}},
        {"vmerge.vxm", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG}},
        {"vmerge.vim", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::IMM}},
        {"vsext.vf4", {RvvOperandType::VREG, RvvOperandType::VREG}},
        {"vsext.vf2", {RvvOperandType::VREG, RvvOperandType::VREG}},
        {"vslideup.vx", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG, RvvOperandType::VM}},
        {"vslideup.vi", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::IMM, RvvOperandType::VM} },
        {"vslidedown.vx", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG, RvvOperandType::VM}},
        {"vslidedown.vi", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::IMM, RvvOperandType::VM} },
        {"vcompress.vm", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG}},
        {"vle8.v", {RvvOperandType::VREG, RvvOperandType::XREG}},
        {"vle16.v", {RvvOperandType::VREG, RvvOperandType::XREG}},
        {"vle32.v", {RvvOperandType::VREG, RvvOperandType::XREG}},
        {"vse8.v", {RvvOperandType::VREG, RvvOperandType::XREG}},
        {"vse16.v", {RvvOperandType::VREG, RvvOperandType::XREG}},
        {"vse32.v", {RvvOperandType::VREG, RvvOperandType::XREG}},
        {"vlm.v", {RvvOperandType::VREG, RvvOperandType::XREG}},
        {"vsm.v", {RvvOperandType::VREG, RvvOperandType::XREG}},
        // 只开放rd, rs1, vtype固定为e32 = 32, m1 = 1, ta = 1, ma = 1
        {"vsetvli", {RvvOperandType::XREG, RvvOperandType::XREG}},
        {"vsetivli", {RvvOperandType::XREG, RvvOperandType::XREG}},
        {"vcpop.m", {RvvOperandType::XREG, RvvOperandType::VREG}},
        {"vmseq.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VM}},
        {"vmseq.vx", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG, RvvOperandType::VM}},
        {"vmseq.vi", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::IMM, RvvOperandType::VM} },
        {"vmslt.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VM}},
        {"vmslt.vx", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG, RvvOperandType::VM}},
        {"vmsgt.vx", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG, RvvOperandType::VM}},
        {"vmsgt.vi", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::IMM, RvvOperandType::VM}},
        {"vor.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VM}},
        {"vor.vx", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG, RvvOperandType::VM}},
        {"vor.vi", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::IMM, RvvOperandType::VM} },
        {"vand.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VM}},
        {"vand.vx", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG, RvvOperandType::VM}},
        {"vand.vi", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::IMM, RvvOperandType::VM} },
        {"vxor.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VM}},
        {"vxor.vx", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG, RvvOperandType::VM}},
        {"vxor.vi", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::IMM, RvvOperandType::VM} },
        {"addiu", {RvvOperandType::XREG, RvvOperandType::XREG, RvvOperandType::IMM}},
        {"vmul8.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG}},
        {"vmul8.vx", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG}},
        {"vmul16.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG}},
        {"vmul16.vx", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG}},
        {"vmul16.vf2", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG}},
        {"vmul16.vxf2", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::XREG}},
        {"vmul16.xvf2", {RvvOperandType::VREG, RvvOperandType::XREG, RvvOperandType::VREG}},
        {"vquant_i8ti32.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG}},
        {"vquant_i32ti8.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG}},
        {"vquant_i16ti32.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG}},
        {"vquant_i32ti16.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG}},
        {"vquant_i32ti10.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG}},
        {"load_lut_i8_d8", {RvvOperandType::XREG}},
        {"load_lut_i8_d10", {RvvOperandType::XREG}},
        {"load_lut_i10_d8", {RvvOperandType::XREG}},
        {"load_lut_i10_d10", {RvvOperandType::XREG}},
        {"vsfu_i8_d8.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG}},
        {"vsfu_i8_d10.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG}},
        {"vsfu_i10_d8.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG}},
        {"vsfu_i10_d10.vv", {RvvOperandType::VREG, RvvOperandType::VREG, RvvOperandType::VREG}},
        {"loop_cfgi", {RvvOperandType::IMM, RvvOperandType::IMM}},
        {"loop_cfgx", {RvvOperandType::XREG, RvvOperandType::IMM}},
        {"loop_end", {RvvOperandType::IMM}},
    };

    if (rvv_inst_operand_types_table.count(inst_name) == 0) {
        throw std::runtime_error("unsupported rvv instruction: " + inst_name);
    }

    auto operand_types = rvv_inst_operand_types_table[inst_name];
    assert(operand_types.size() > 0 && "function pointer should not be null");

    return operand_types;
}

VeuInstSlotSupportion getInstSupportedSlots(const std::string& inst_name) {
    static const std::set<std::string> slot_0 = {
        "lb", "lh", "load_lut_i10_d10", "load_lut_i10_d8", "load_lut_i8_d10", "load_lut_i8_d8", "loop_cfgi",
        "loop_cfgx", "loop_end", "lw", "mul", "mulh", "sll", "slli", "slt", "slti", "sra", "srai", "srl", "srli",
        "vand.vi", "vand.vv", "vand.vx", "vle16.v", "vle32.v", "vle8.v", "vlm.v", "vmseq.vi", "vmseq.vv", "vmseq.vx",
        "vmsgt.vi", "vmsgt.vx", "vmslt.vx", "vmslt.vv", "vmul16.vf2", "vmul16.vv", "vmul16.vx", "vmul16.vxf2",
        "vmul16.xvf2", "vmul8.vv", "vmul8.vx", "vor.vi", "vor.vv", "vor.vx", "vsll.vi", "vsll.vv", "vsll.vx",
        "vsra.vi", "vsra.vv", "vsra.vx", "vsext.vf2", "vsext.vf4", "vsetivli", "vsetvli", "vsrl.vi", "vsrl.vv",
        "vsrl.vx", "vxor.vi", "vxor.vv", "vxor.vx"
    };

    static const std::set<std::string> slot_1 = {
        "and", "andi", "or", "ori", "sb", "sh", "sw", "vcompress.vm", "vcpop.m", "vdiv.vv", "vdiv.vx",
        "vquant_i16ti32.vv", "vquant_i32ti10.vv", "vquant_i32ti16.vv", "vquant_i32ti8.vv", "vquant_i8ti32.vv", "vredmax.vs",
        "vredmin.vs", "vredsum.vs", "vse16.v", "vse32.v", "vse8.v", "vsfu_i10_d10.vv", "vsfu_i10_d8.vv", "vsfu_i8_d10.vv",
        "vsfu_i8_d8.vv", "vslidedown.vi", "vslidedown.vx", "vslideup.vi", "vslideup.vx", "vsm.v", "xor"
    };

    static const std::set<std::string> slot_0_1 = {
        "add", "addi", "addiu", "lui", "sub", "vmax.vv", "vmax.vx", "vmerge.vim", "vmerge.vvm", "vmerge.vxm",
        "vmin.vv", "vmin.vx", "vmv.s.x", "vmv.v.i", "vmv.v.v", "vmv.v.x", "vmv.x.s", "vmv1r.v", "vsadd.vi",
        "vsadd.vv", "vsadd.vx", "vssub.vv", "vssub.vx"
    };
    
    assert(slot_0.count(inst_name) + slot_1.count(inst_name) + slot_0_1.count(inst_name) == 1 && "an instruction should only belong to one slot category");
                                         
    VeuInstSlotSupportion supported_type;
    if (slot_0_1.count(inst_name) > 0) {
        supported_type = VeuInstSlotSupportion::SLOT_0_1;
    } else if (slot_0.count(inst_name) > 0) {
        supported_type = VeuInstSlotSupportion::SLOT_0;
    } else if (slot_1.count(inst_name) > 0) {
        supported_type = VeuInstSlotSupportion::SLOT_1;
    } else {
        throw std::runtime_error("unsupported instruction for slot assignment: " + inst_name);
    }

    return supported_type;
}

RvvInst RvvInstGenerator::genNopRvvInst() {
    RvvInst inst;
    inst.raw = 0x00000013; // addi x0, x0, 0
    inst.name = "nop";
    inst.assembly = "nop";
    inst.slot_supportion = VeuInstSlotSupportion::SLOT_0_1; // nop指令可以和任意slot的指令打包

    CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = {0, RegStatus::USED}}};
    CodegenOpOperand arg1 = {.type = RvvOperandType::IMM, .value = {.imm = 0}};
    inst.inputs = {arg0, arg1};
    inst.output = arg0;

    return inst;
}

uint32_t RvvInstGenerator::toCode(const std::vector<uint32_t>& raw_args) const {
    auto func = getRvvInstGenFunc(name);
    return func(raw_args);
}

std::string RvvInstGenerator::toAssembly(const std::vector<uint32_t>& args) const {
    auto operand_types = getRvvInstOperandTypes(name);

    std::string assembly = name + " ";
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0 && operand_types[i] != RvvOperandType::VM) {
            assembly += ", ";
        }
        switch (operand_types[i]) {
            case RvvOperandType::XREG:
                assembly += "x" + std::to_string(args[i]);
                break;
            case RvvOperandType::VREG:
                assembly += "v" + std::to_string(args[i]);
                break;
            case RvvOperandType::IMM:
                assembly += std::to_string(args[i]);
                break;
            case RvvOperandType::VM:
                // vm operand is implicit and does not need to be printed in assembly
                break;
            default:
                throw std::runtime_error("unsupported operand type");
        }
    }

    return assembly;
}

RvvInst RvvInstGenerator::codegen(const std::vector<CodegenOpOperand>& inputs, const CodegenOpOperand& output) {
    auto operand_types = getRvvInstOperandTypes(name);

    std::vector<CodegenOpOperand> all_args;
    if (output.type != RvvOperandType::NONE) {
        // 检查是否有输出
        all_args.push_back(output);
    }
    all_args.insert(all_args.end(), inputs.begin(), inputs.end());

    if (all_args.size() != operand_types.size()) {
        assert(operand_types.size() == all_args.size() + 1 && "the last operand is vm for vector instructions");
        assert(operand_types.back() == RvvOperandType::VM && "the last operand should be vm for vector instructions");

        // vm is implicit and always 0 for our use case, so we can just push back a dummy value
        CodegenOpOperand vm_operand;
        vm_operand.type = RvvOperandType::VM;
        vm_operand.value.imm = 1;
        all_args.push_back(vm_operand); 
    }

    for (int32_t i = 0; i < all_args.size(); ++i) {
        auto arg = all_args[i];
        auto operand_type = operand_types.at(i);
        assert(arg.type == operand_type && "operand type does not match the expected type");
    }

    std::vector<uint32_t> raw_args;
    for (int32_t i = 0; i < all_args.size(); ++i) {
        switch (all_args[i].type) {
            case RvvOperandType::XREG: {
                auto reg_id = all_args[i].value.x_reg.id;
                // assert(reg_id >= 0 && reg_id < 32 && "x register index should be less than 32");
                raw_args.push_back(static_cast<uint32_t>(reg_id));
                break;
            }
            case RvvOperandType::VREG: {
                auto reg_id = all_args[i].value.v_reg.id;
                // assert(((reg_id >= 0 && reg_id < 16) || (reg_id >= 23 && reg_id <= 31)) && "v register index should be less than 16 or between 23 and 31");
                raw_args.push_back(static_cast<uint32_t>(reg_id));
                break;
            }
            case RvvOperandType::IMM: {
                auto im = all_args[i].value.imm;
                raw_args.push_back(im);
                break;
            }
            case RvvOperandType::VM: {
                auto im = all_args[i].value.imm;
                assert(im == 1 && "vm operand should always be 1 for our use case");
                raw_args.push_back(im);
                break;
            }
            default:
                throw std::runtime_error("unsupported operand type");
        }
    }

    RvvInst inst;
    inst.name = name;
    inst.assembly = toAssembly(raw_args);
    inst.raw = toCode(raw_args);
    inst.slot_supportion = getInstSupportedSlots(name);
    inst.inputs = inputs;
    inst.output = output;

    return inst;
}

} // namespace xp_std
} // namespace xp_mlir