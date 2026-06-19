#include "standalone/OpCodegen.hpp"
#include "mlir/IR/Operation.h"
#include "standalone/StandaloneOpInfo.hpp"
#include "standalone/RvvInstGen.hpp"
#include "standalone/RegManager.hpp"
#include "xp_mlir/Dialect/XMA/IR/XMAOps.h"
#include "xp_mlir/Dialect/XMA/Vector/XVectorOps.hpp"
#include <string>
#include <utility>
#include "standalone/com_utils.hpp"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

namespace xp_mlir {
namespace xp_std {
std::vector<RvvInst>& OpCodegen::getCodeContainer() {
    switch (code_loc_) {
        case CodeLoc::PRELOGUE: {
            return prelogue_code_;
        }
        case CodeLoc::OUTER_LOOP:{
            return loop_code_;
        }
        case CodeLoc::INNER_LOOP: {
            return loop_code_;
        }
        default: {
            assert(false && "invalid code location");
            return prelogue_code_;
        }
    }

    return prelogue_code_;
}

CodegenOpOperand OpCodegen::xvectorOperand2RvvOperand(mlir::Value val) {
    assert(val && "the value is expected to be valid");
    auto val_type = val.getType();
    CodegenOpOperand operand;
    if (auto scalar_reg_type = llvm::dyn_cast<xvec::ScalarRegType>(val_type)) {
        operand.type = RvvOperandType::XREG;
        operand.value.x_reg = {.id = scalar_reg_type.getId(), .status = RegStatus::USED};
        operand.bit_width = static_cast<int32_t>(scalar_reg_type.getDtype());
    } else if (auto vector_reg_type = llvm::dyn_cast<xvec::VectorRegType>(val_type)) {
        operand.type = RvvOperandType::VREG;
        operand.value.v_reg = {.id = vector_reg_type.getId(), .status = RegStatus::USED};
        operand.bit_width = static_cast<int32_t>(vector_reg_type.getDtype());
    } else if (llvm::isa<xvec::RvvImmType>(val_type)) {
        operand.type = RvvOperandType::IMM;
        auto val_def_op = val.getDefiningOp();
        assert(val_def_op && "the defining op of the value is expected to be valid");
        auto imm_value = llvm::dyn_cast<xvec::ImmOp>(val_def_op).getVal();
        operand.value.imm = imm_value;
        operand.bit_width = 32;
    } else {
        assert(false && "unsupported operand type for xvector op");
    }

    return operand;
}

std::vector<CodegenOpOperand> OpCodegen::xvecOpInputs2RvvInstInputs(mlir::Operation* op) {
    if (auto init_xreg_op = llvm::dyn_cast<xvec::InitXregOp>(op)) {
        // 无operand, 输入是attribute
        auto imm_value = init_xreg_op.getVal();
        CodegenOpOperand operand = {.type = RvvOperandType::IMM, .value = {.imm = imm_value}};
        return {operand};
    }

    auto xvec_inputs = op->getOperands();
    std::vector<CodegenOpOperand> rvv_inputs;
    for (auto operand : xvec_inputs) {
        auto operand_rvv = xvectorOperand2RvvOperand(operand);
        rvv_inputs.push_back(operand_rvv);
    }
    return rvv_inputs;
}

void OpCodegen::codegen() {
    llvm::TypeSwitch<mlir::Operation*>(xvec_op_)
        .Case<xvec::ImmOp>([&](auto op) { codegen_immOp(op); })
        .Case<xvec::InitXregOp>([&](auto op) { codegen_initXregOp(op); })
        .Case<xvec::ScalarLoadOp>([&](auto op) { codegen_scalarLoadOp(op); })
        .Case<xvec::VectorLoadOp>([&](auto op) { codegen_vectorLoadOp(op); })
        .Case<xvec::StoreOp>([&](auto op) { codegen_storeOp(op); })
        .Case<xvec::ReciprocalOp>([&](auto op) { codegen_reciprocalOp(op); })
        .Case<xvec::Xadd>([&](auto op) { codegen_xaddOp(op); })
        .Case<xvec::Xmul>([&](auto op) { codegen_xmulOp(op); })
        .Case<xvec::Xdiv>([&](auto op) { codegen_xdivOp(op); })
        .Case<xvec::Xmod>([&](auto op) { codegen_xmodOp(op); })
        .Case<xvec::Xincrement>([&](auto op) { codegen_xincrementOp(op); })
        .Case<xvec::SelectScalarFromVectorAndBroadcast>([&](auto op) { codegen_selectScalarFromVectorAndBroadcastOp(op); })
        .Case<xvec::Extend>([&](auto op) { codegen_extendOp(op); })
        .Case<xvec::Vadd>([&](auto op) { codegen_vaddOp(op); })
        .Case<xvec::Vsub>([&](auto op) { codegen_vsubOp(op); })
        .Case<xvec::Vmul>([&](auto op) { codegen_vmulOp(op); })
        .Case<xvec::Vdiv>([&](auto op) { codegen_vdivOp(op); })
        .Case<xvec::Vmax>([&](auto op) { codegen_vmaxOp(op); })
        .Case<xvec::Vmin>([&](auto op) { codegen_vminOp(op); })
        .Case<xvec::Vabs>([&](auto op) { codegen_vabsOp(op); })
        .Case<xvec::Vquant>([&](auto op) { codegen_vquantOp(op); })
        .Case<xvec::Vqlut>([&](auto op) { codegen_vqlutOp(op); })
        .Case<xvec::LoopBegin>([&](auto op) { codegen_loopBeginOp(op); })
        .Case<xvec::LoopEnd>([&](auto op) { codegen_loopEndOp(op); })
        .Case<xvec::RvvConfig>([&](auto op) { codegen_rvvConfigOp(op); })
        .Default([&](mlir::Operation* op) {
            assert(false && "unsupported xvector operation for codegen");
        });
}


void OpCodegen::codegen_immOp(mlir::Operation* op) {
    return;
}

std::vector<RvvInst> OpCodegen::codegen_matmulTcLd(const StandaloneOpInfo& op_info) {
    /*
      vsetvli rd,rs1,sew,vlmul,vta,vma (vma=1,vta=1,sew=2,vlmul=0,rs1=15,rd=30)
      
      vle32.v vd, (rs1) (rs1=24,vd=15)
      
      vle32.v vd, (rs1) (rs1=24,vd=1)
      vslideup.vx vd,vs2,rs1,vm (vm=0,vs2=15,rs1=3,vd=15)
      
      vslideup.vx vd,vs2,rs1,vm (vm=0,vs2=15,rs1=3,vd=15)
      
      vslideup.vx vd,vs2,rs1,vm (vm=0,vs2=15,rs1=3,vd=15)
      
      vslideup.vx vd,vs2,rs1,vm (vm=0,vs2=15,rs1=3,vd=15)
      
      vse32.v vs3, (rs1) (rs1=22,vs3=1)
     */

    std::vector<RvvInst> ret_code_;

    assert(op_info.input_infos.size() == 1 && "matmulTcLd op should have only one input");
    auto input_info = op_info.input_infos.at(0);
    auto base_addr = input_info.mem_value.l1_addr;

    // "lui x18, #(imm)"
    // "addiu x18, x18, #(imm)"
    auto l1_base_addr_xreg = RegManager::get().allocScalarReg();
    {
        RvvInstGenerator lui_gen("lui");
        CodegenOpOperand arg0 = {.type = RvvOperandType::IMM, .value = {.imm = static_cast<uint32_t>(base_addr)}};
        std::vector<CodegenOpOperand> inputs = {arg0};
        CodegenOpOperand output = {.type = RvvOperandType::XREG, .value = {.x_reg = l1_base_addr_xreg}};

        auto code_ = lui_gen.codegen(inputs, output);
        ret_code_.push_back(code_);
    }

    {
        RvvInstGenerator lui_gen("addiu");
        CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = l1_base_addr_xreg}};
        CodegenOpOperand arg1 = {.type = RvvOperandType::IMM, .value = {.imm = static_cast<uint32_t>(base_addr)}};
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::XREG, .value = {.x_reg = l1_base_addr_xreg}};

        auto code_ = lui_gen.codegen(inputs, output);
        ret_code_.push_back(code_);
    }

    // "addi x18, x0, #(imm)"
    auto left_shift_num_xreg = RegManager::get().allocScalarReg();
    {
        RvvInstGenerator addi_gen("addi");
        CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = {0, RegStatus::USED}}}; // x0
        CodegenOpOperand arg1 = {.type = RvvOperandType::IMM, .value = {.imm = 32}};
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::XREG, .value = {.x_reg = left_shift_num_xreg}};

        auto code_ = addi_gen.codegen(inputs, output);
        ret_code_.push_back(code_);
    }

    // vsetvli rd,rs1,sew,vlmul,vta,vma (vma=1,vta=1,sew=2,vlmul=0,rs1=15,rd=30)
    {
        auto vl_set_xreg = RegManager::get().allocSpecificScalarReg(VEU_RVV_CONFIG_VL_SET_XREG_ID);
        auto vl_return_xreg = RegManager::get().allocSpecificScalarReg(VEU_RVV_CONFIG_VL_RETURN_XREG_ID);

        // vl = 128
        {
            RvvInstGenerator addi_gen("addi");
            CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = {0, RegStatus::USED}}}; // x0
            CodegenOpOperand arg1 = {.type = RvvOperandType::IMM, .value = {.imm = 128}};
            std::vector<CodegenOpOperand> inputs = {arg0, arg1};
            CodegenOpOperand output = {.type = RvvOperandType::XREG, .value = {.x_reg = vl_set_xreg}};

            auto addi_code_ = addi_gen.codegen(inputs, output);
            ret_code_.push_back(addi_code_);
        }

        RvvInstGenerator vsetvli_gen("vsetvli");
        CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = vl_set_xreg}};
        std::vector<CodegenOpOperand> inputs = {arg0};
        CodegenOpOperand output = {.type = RvvOperandType::XREG, .value = {.x_reg = vl_return_xreg}};

        auto code_ = vsetvli_gen.codegen(inputs, output);
        ret_code_.push_back(code_);
    }

    // vle32.v vd, (rs1) (rs1=24,vd=15)
    auto real_ld_vreg = RegManager::get().allocSpecificVectorReg(VEU_RESERVED_MATMULTC_VREG_ID);
    {
        RvvInstGenerator load_gen("vle32.v");
        CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = l1_base_addr_xreg}};
        std::vector<CodegenOpOperand> inputs = {arg0};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = real_ld_vreg}}; // vd=15

        auto code_ = load_gen.codegen(inputs, output);
        ret_code_.push_back(code_);
    }
      
    // vle32.v vd, (rs1) (rs1=24,vd=1)
    auto virtual_ld_vreg = RegManager::get().allocVectorReg();
    {
        RvvInstGenerator load_gen("vle32.v");
        CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = l1_base_addr_xreg}};
        std::vector<CodegenOpOperand> inputs = {arg0};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = virtual_ld_vreg}};

        auto code_ = load_gen.codegen(inputs, output);
        ret_code_.push_back(code_);
    }

    // 0: vslideup.vx vd,vs2,rs1,vm (vm=0,vs2=15,rs1=3,vd=15)
    {
        RvvInstGenerator slideup_gen("vslideup.vx");
        CodegenOpOperand arg0 = {.type = RvvOperandType::VREG, .value = {.v_reg = real_ld_vreg}}; // vs2=15
        CodegenOpOperand arg1 = {.type = RvvOperandType::XREG, .value = {.x_reg = left_shift_num_xreg}}; // rs1=3
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = real_ld_vreg}}; // vd=15

        auto code_ = slideup_gen.codegen(inputs, output);
        ret_code_.push_back(code_);
    }

    // 1: vslideup.vx vd,vs2,rs1,vm (vm=0,vs2=15,rs1=3,vd=15)
    {
        RvvInstGenerator slideup_gen("vslideup.vx");
        CodegenOpOperand arg0 = {.type = RvvOperandType::VREG, .value = {.v_reg = real_ld_vreg}}; // vs2=15
        CodegenOpOperand arg1 = {.type = RvvOperandType::XREG, .value = {.x_reg = left_shift_num_xreg}}; // rs1=3
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = real_ld_vreg}}; // vd=15

        auto code_ = slideup_gen.codegen(inputs, output);
        ret_code_.push_back(code_);
    }

    // 2: vslideup.vx vd,vs2,rs1,vm (vm=0,vs2=15,rs1=3,vd=15)
    {
        RvvInstGenerator slideup_gen("vslideup.vx");
        CodegenOpOperand arg0 = {.type = RvvOperandType::VREG, .value = {.v_reg = real_ld_vreg}}; // vs2=15
        CodegenOpOperand arg1 = {.type = RvvOperandType::XREG, .value = {.x_reg = left_shift_num_xreg}}; // rs1=3
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = real_ld_vreg}}; // vd=15

        auto code_ = slideup_gen.codegen(inputs, output);
        ret_code_.push_back(code_);
    }

    // 3: vslideup.vx vd,vs2,rs1,vm (vm=0,vs2=15,rs1=3,vd=15)
    {
        RvvInstGenerator slideup_gen("vslideup.vx");
        CodegenOpOperand arg0 = {.type = RvvOperandType::VREG, .value = {.v_reg = real_ld_vreg}}; // vs2=15
        CodegenOpOperand arg1 = {.type = RvvOperandType::XREG, .value = {.x_reg = left_shift_num_xreg}}; // rs1=3
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = real_ld_vreg}}; // vd=15

        auto code_ = slideup_gen.codegen(inputs, output);
        ret_code_.push_back(code_);
    }

    // vse32.v vs3, (rs1) (rs1=22,vs3=1)
    {
        RvvInstGenerator store_gen("vse32.v");
        CodegenOpOperand arg0 = {.type = RvvOperandType::VREG, .value = {.v_reg = virtual_ld_vreg}};
        CodegenOpOperand arg1 = {.type = RvvOperandType::XREG, .value = {.x_reg = l1_base_addr_xreg}};
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::NONE, .value = {.imm = 0}};

        auto code_ = store_gen.codegen(inputs, output);
        ret_code_.push_back(code_);
    }

    return ret_code_;
}

void OpCodegen::codegen_initXregOp(mlir::Operation* op) {
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 1 && "init xreg op should have only one input");
    assert(rvv_inputs[0].type == RvvOperandType::IMM && "the input of init xreg op should be an immediate value");  
    auto imm_value = rvv_inputs.at(0).value.imm;

    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));

    if (is11bitImm(imm_value) || static_cast<int32_t>(imm_value) == -1) {
        // addi会将符号位扩展到高位, 所以需要限制到11bit, 否则会出现符号位扩展错误
        // 如果是-1, 符号扩展后也是-1, 不会出现错误
        // "addi x18, x0, #(imm)"
        RvvInstGenerator addi_gen("addi");
        CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = {0, RegStatus::USED}}}; // x0
        CodegenOpOperand arg1 = rvv_inputs.at(0);
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = rvv_output;

        auto code_ = addi_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    } else {
        // "lui x18, #(imm)"
        // "addiu x18, x18, #(imm)"
        {
            RvvInstGenerator lui_gen("lui");
            CodegenOpOperand arg0 = rvv_inputs.at(0);
            std::vector<CodegenOpOperand> inputs = {arg0};
            CodegenOpOperand output = rvv_output;

            auto code_ = lui_gen.codegen(inputs, output);
            getCodeContainer().push_back(code_);
        }

        {
            RvvInstGenerator lui_gen("addiu");
            CodegenOpOperand arg0 = rvv_output;
            CodegenOpOperand arg1 = rvv_inputs.at(0);
            std::vector<CodegenOpOperand> inputs = {arg0, arg1};
            CodegenOpOperand output = rvv_output;

            auto code_ = lui_gen.codegen(inputs, output);
            getCodeContainer().push_back(code_);
        }
    }
}

void OpCodegen::codegen_scalarLoadOp(mlir::Operation* op) {
    assert(false && "to impl");
}

void OpCodegen::codegen_vectorLoadOp(mlir::Operation* op) {
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 1 && "vector load op should have only one input");

    auto rvv_input = rvv_inputs.at(0);
    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));
    auto bitwidth = rvv_output.bit_width;

    auto inst_name = (bitwidth == 32) ? "vle32.v" : (is_i16(bitwidth) ? "vle16.v" : "vle8.v");
    RvvInstGenerator load_gen(inst_name);
    CodegenOpOperand arg0 = rvv_input;
    std::vector<CodegenOpOperand> inputs = {arg0};
    CodegenOpOperand output = rvv_output;

    auto code_ = load_gen.codegen(inputs, output);
    getCodeContainer().push_back(code_);
}

void OpCodegen::codegen_storeOp(mlir::Operation* op) {
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 2 && "vector store op should have only one input");

    auto input_dtype = llvm::dyn_cast<xvec::VectorRegType>(op->getOperand(0).getType()).getDtype();
    auto bitwidth = static_cast<int32_t>(input_dtype);

    auto inst_name = (bitwidth == 32) ? "vse32.v" : (is_i16(bitwidth) ? "vse16.v" : "vse8.v");
    RvvInstGenerator store_gen(inst_name);
    std::vector<CodegenOpOperand> inputs = rvv_inputs;
    CodegenOpOperand output = {.type = RvvOperandType::NONE, .value = {.imm = 0}};

    auto code_ = store_gen.codegen(inputs, output);
    getCodeContainer().push_back(code_);
}

void OpCodegen::codegen_reciprocalOp(mlir::Operation* op) {
    /*
     * 如果除0, 则分母用1替代
     * scalar = x17
     * slli x20, x17, 26

     * addi x22, x0, 1
     * slli x18, x22, 26
     * addi x18, x18, -1

     * x = v5
     * vmv.v.x v2,x18
     
     * vmv.v.i v15,imm=1 
     * vmseq.vi v0, v5, imm=0
     * vmerge.vvm v14,v5,v15, <v0>
     * vdiv.vv v6,v2,v14
     * vsadd.vx v7, v6, x20
     */
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));
    auto x = rvv_inputs.at(0);
    assert(x.type == RvvOperandType::VREG && "the input of reciprocal op should be a vector register");
    auto scalar = rvv_inputs.at(1);
    assert(scalar.type == RvvOperandType::XREG && "the second input of reciprocal op should be a scalar register");

    const uint32_t max_shift_num = 26;

    auto scalar_shift_xreg = RegManager::get().allocScalarReg();
    {
        // slli x20, x17, 26
        RvvInstGenerator slli_gen("slli");
        CodegenOpOperand arg0 = scalar;
        CodegenOpOperand arg1 = {.type = RvvOperandType::IMM, .value = {.imm = max_shift_num}};
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::XREG, .value = {.x_reg = scalar_shift_xreg}};

        auto code_ = slli_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }

    auto one_xreg = RegManager::get().allocScalarReg();
    {
        // addi x22, x0, 1
        RvvInstGenerator addi_gen("addi");
        CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = {0, RegStatus::USED}}}; // x0
        CodegenOpOperand arg1 = {.type = RvvOperandType::IMM, .value = {.imm = 1}};
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::XREG, .value = {.x_reg = one_xreg}};

        auto code_ = addi_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }

    auto one_shift_xreg = RegManager::get().allocScalarReg();
    {
        // slli x18, x22, 26
        RvvInstGenerator slli_gen("slli");
        CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = one_xreg}};
        CodegenOpOperand arg1 = {.type = RvvOperandType::IMM, .value = {.imm = max_shift_num}};
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::XREG, .value = {.x_reg = one_shift_xreg}};

        auto code_ = slli_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }

    {
        // addi x18, x18, -1
        RvvInstGenerator addi_gen("addi");
        CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = one_shift_xreg}};
        CodegenOpOperand arg1 = {.type = RvvOperandType::IMM, .value = {.imm = static_cast<uint32_t>(-1)}};
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::XREG, .value = {.x_reg = one_shift_xreg}};

        auto code_ = addi_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }

    auto one_shift_broadcast_vreg = RegManager::get().allocVectorReg();
    {
        // vmv.v.x v2,x18
        RvvInstGenerator vmv_vx_gen("vmv.v.x");
        CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = one_shift_xreg}};
        std::vector<CodegenOpOperand> inputs = {arg0};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = one_shift_broadcast_vreg}};

        auto code_ = vmv_vx_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }

    auto one_broadcast_vreg = RegManager::get().allocVectorReg();
    {
        // vmv.v.i v15,imm=1 
        RvvInstGenerator vmv_vi_gen("vmv.v.i");
        CodegenOpOperand arg0 = {.type = RvvOperandType::IMM, .value = {.imm = 1}};
        std::vector<CodegenOpOperand> inputs = {arg0};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = one_broadcast_vreg}};

        auto code_ = vmv_vi_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }

    {
        // vmseq.vi v0, v5, imm=0
        RvvInstGenerator vmseq_vi_gen("vmseq.vi");
        CodegenOpOperand arg0 = x;
        CodegenOpOperand arg1 = {.type = RvvOperandType::IMM, .value = {.imm = 0}};
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = RegManager::get().allocSpecificVectorReg(VEU_RESERVED_MASK_VREG_ID)}};

        auto code_ = vmseq_vi_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }

    auto div_denominator_vreg = RegManager::get().allocVectorReg();
    {
        // vmerge.vvm v14,v5,v15, <v0>
        RvvInstGenerator vmerge_gen("vmerge.vvm");
        CodegenOpOperand arg0 = x;
        CodegenOpOperand arg1 = {.type = RvvOperandType::VREG, .value = {.v_reg = one_broadcast_vreg}}; // v15
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = div_denominator_vreg}};
        
        auto code_ = vmerge_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }

    auto vdiv_output_vreg = RegManager::get().allocVectorReg();
    {
        // vdiv.vv v6,v2,v14
        RvvInstGenerator vdiv_gen("vdiv.vv");
        CodegenOpOperand arg0 = {.type = RvvOperandType::VREG, .value = {.v_reg = one_shift_broadcast_vreg}}; // v2
        CodegenOpOperand arg1 = {.type = RvvOperandType::VREG, .value = {.v_reg = div_denominator_vreg}}; // v14
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = vdiv_output_vreg}};

        auto code_ = vdiv_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }

    {
        // vsadd.vx v7, v6, x20
        RvvInstGenerator vsadd_gen("vsadd.vx");
        CodegenOpOperand arg0 = {.type = RvvOperandType::VREG, .value = {.v_reg = vdiv_output_vreg}}; // v6
        CodegenOpOperand arg1 = {.type = RvvOperandType::XREG, .value = {.x_reg = scalar_shift_xreg}}; // x20
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = rvv_output;

        auto code_ = vsadd_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }
}

void OpCodegen::codegen_xaddOp (mlir::Operation* op) {
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 2 && "xadd op should have two inputs");
    auto rvv_input0 = rvv_inputs.at(0);
    auto rvv_input1 = rvv_inputs.at(1);
    assert(rvv_input0.type == RvvOperandType::XREG && "the first input of xadd op should be a scalar register");
    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));
    
    if (rvv_input1.type == RvvOperandType::IMM) {
        auto imm_value = rvv_input1.value.imm;
        assert(is11bitImm(imm_value) && "the immediate value for addi instruction should be a 12 bit immediate");

        // addi rd, rs, imm
        RvvInstGenerator addi_gen("addi");
        std::vector<CodegenOpOperand> inputs = rvv_inputs;
        CodegenOpOperand output = rvv_output;

        auto code_ = addi_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    } else if (rvv_input1.type == RvvOperandType::XREG) {
        // add rd, rs1, rs2
        RvvInstGenerator add_gen("add");
        std::vector<CodegenOpOperand> inputs = rvv_inputs;
        CodegenOpOperand output = rvv_output;

        auto code_ = add_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    } else {
        assert(false && "the second input of xadd op should be either an immediate value or a scalar register");
    }   
}

void OpCodegen::codegen_xmulOp(mlir::Operation* op) {
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 2 && "xmul op should have two inputs");
    auto rvv_input0 = rvv_inputs.at(0);
    auto rvv_input1 = rvv_inputs.at(1);
    assert(rvv_input0.type == RvvOperandType::XREG && "the 1st input should be a scalar register");
    assert(rvv_input1.type == RvvOperandType::XREG && "the 2nd input should be a scalar register");
    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));

    // mul rd, rs1, rs2
    RvvInstGenerator mul_gen("mul");
    std::vector<CodegenOpOperand> inputs = rvv_inputs;
    CodegenOpOperand output = rvv_output;

    auto code_ = mul_gen.codegen(inputs, output);
    getCodeContainer().push_back(code_);
}

void OpCodegen::codegen_xdivOp(mlir::Operation* op) {
    /*
     * // x2 = x26 / x1
     * vmv.s.x v15, x26                         
     * vdiv.vx v15, v15, x1                             
     * vmv.x.s x2, v15 
     */
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 2 && "xdiv op should have two inputs");
    auto rvv_input0 = rvv_inputs.at(0);
    auto rvv_input1 = rvv_inputs.at(1);
    assert(rvv_input0.type == RvvOperandType::XREG && "the 1st input should be a scalar register");
    assert(rvv_input1.type == RvvOperandType::XREG && "the 2nd input should be a scalar register");
    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));

    auto x_broadcast_vreg = RegManager::get().allocVectorReg();
    {
        // vmv.s.x v15, x26 
        RvvInstGenerator vmv_sx_gen("vmv.s.x");
        CodegenOpOperand arg0 = rvv_inputs.at(0);
        std::vector<CodegenOpOperand> inputs = {arg0};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = x_broadcast_vreg}};

        auto code_ = vmv_sx_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }

    {
        // vdiv.vx v15, v15, x1 
        RvvInstGenerator vdiv_vx_gen("vdiv.vx");
        CodegenOpOperand arg0 = {.type = RvvOperandType::VREG, .value = {.v_reg = x_broadcast_vreg}};
        CodegenOpOperand arg1 = rvv_inputs.at(1);
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = x_broadcast_vreg}};

        auto code_ = vdiv_vx_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }

    {
        // vmv.x.s x2, v15 
        RvvInstGenerator vmv_xs_gen("vmv.x.s");
        CodegenOpOperand arg0 = {.type = RvvOperandType::VREG, .value = {.v_reg = x_broadcast_vreg}};
        std::vector<CodegenOpOperand> inputs = {arg0};
        CodegenOpOperand output = rvv_output;

        auto code_ = vmv_xs_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }
}

void OpCodegen::codegen_xmodOp(mlir::Operation* op) {
    /*
     * // mod(x26, x2, x1) = x26 - x2*x1, n = x2
     * mul x3, x2, x1
     * sub x4, x26, x3
     */
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 3 && "xmod op should have 3 inputs");
    auto rvv_input0 = rvv_inputs.at(0);
    auto rvv_input1 = rvv_inputs.at(1);
    auto rvv_input2 = rvv_inputs.at(2);
    assert(rvv_input0.type == RvvOperandType::XREG && "the 1st input should be a scalar register");
    assert(rvv_input1.type == RvvOperandType::XREG && "the 2nd input should be a scalar register");
    assert(rvv_input2.type == RvvOperandType::XREG && "the 3rd input should be a scalar register");
    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));

    auto mul_res_xreg = RegManager::get().allocScalarReg();
    {
        // mul x3, x2, x1
        RvvInstGenerator mul_gen("mul");
        CodegenOpOperand arg0 = rvv_inputs.at(1);
        CodegenOpOperand arg1 = rvv_inputs.at(2);
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::XREG, .value = {.x_reg = mul_res_xreg}};

        auto code_ = mul_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }

    {
        // sub x4, x26, x3
        RvvInstGenerator sub_gen("sub");
        CodegenOpOperand arg0 = rvv_inputs.at(0);
        CodegenOpOperand arg1 = {.type = RvvOperandType::XREG, .value = {.x_reg = mul_res_xreg}};
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = rvv_output;

        auto code_ = sub_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }
}

void OpCodegen::codegen_xincrementOp(mlir::Operation* op) {
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 1 && "xincrement op should have two inputs");
    auto rvv_input0 = rvv_inputs.at(0);
    assert(rvv_input0.type == RvvOperandType::XREG && "the 1st input should be a scalar register");
    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));

    // addi rd, rs, 1
    RvvInstGenerator addi_gen("addi");
    auto arg0 = rvv_input0;
    CodegenOpOperand arg1 = {.type = RvvOperandType::IMM, .value = {.imm = 1}};
    std::vector<CodegenOpOperand> inputs = {arg0, arg1};
    CodegenOpOperand output = rvv_output;

    auto code_ = addi_gen.codegen(inputs, output);
    getCodeContainer().push_back(code_);
}

void OpCodegen::codegen_selectScalarFromVectorAndBroadcastOp(mlir::Operation* op) {
    /*
     * // index = x1 
     * vslidedown.vx v16, v15, x1
     * vmv.x.s x9, v16;
     * vmv.v.x v5, x9;
     */
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 2 && "selectScalarFromVectorAndBroadcast op should have two inputs");
    auto rvv_input0 = rvv_inputs.at(0);
    auto rvv_input1 = rvv_inputs.at(1);
    assert(rvv_input0.type == RvvOperandType::VREG && "the 1st input should be a vector register");
    assert(rvv_input1.type == RvvOperandType::XREG && "the 2nd input should be a scalar register");
    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));

    auto slide_res_vreg = RegManager::get().allocVectorReg();
    {
        RvvInstGenerator slidown_vx_gen("vslidedown.vx");
        CodegenOpOperand arg0 = rvv_inputs.at(0);
        CodegenOpOperand arg1 = rvv_inputs.at(1);
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = slide_res_vreg}};

        auto code_ = slidown_vx_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }

    auto select_scalar_xreg = RegManager::get().allocScalarReg();
    {
        // vmv.x.s x9, v16;
        RvvInstGenerator vmv_xs_gen("vmv.x.s");
        CodegenOpOperand arg0 = {.type = RvvOperandType::VREG, .value = {.v_reg = slide_res_vreg}};
        std::vector<CodegenOpOperand> inputs = {arg0};
        CodegenOpOperand output = {.type = RvvOperandType::XREG, .value = {.x_reg = select_scalar_xreg}};

        auto code_ = vmv_xs_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }

    {
        // vmv.v.x v5, x9;
        RvvInstGenerator vmv_vx_gen("vmv.v.x");
        CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = select_scalar_xreg}};
        std::vector<CodegenOpOperand> inputs = {arg0};
        CodegenOpOperand output = rvv_output;

        auto code_ = vmv_vx_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }
}

void OpCodegen::codegen_extendOp(mlir::Operation* op) {
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 1 && "extend op should have two inputs");
    auto rvv_input0 = rvv_inputs.at(0);
    assert(rvv_input0.type == RvvOperandType::VREG && "the first input should be a vector register");
    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));

    // "vsext.vf#(C) v25, v2"
    auto input0_bit_width = rvv_input0.bit_width;
    assert((input0_bit_width == 8 || is_i16(input0_bit_width)) && "only support extending i8 or i16 input for extend op");
    const std::string inst_name = (input0_bit_width == 8) ? "vsext.vf4" : "vsext.vf2";

    RvvInstGenerator extend_gen(inst_name);
    std::vector<CodegenOpOperand> inputs = {rvv_input0};
    auto output = rvv_output;
    auto code_extend = extend_gen.codegen(inputs, output);
    getCodeContainer().push_back(code_extend);
}

void OpCodegen::codegen_vaddOp(mlir::Operation* op) {
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 2 && "vadd op should have two inputs");
    auto rvv_input0 = rvv_inputs.at(0);
    auto rvv_input1 = rvv_inputs.at(1);
    assert(rvv_input0.type == RvvOperandType::VREG && "the first input of vadd op should be a vector register");
    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));
    
    if (rvv_input1.type == RvvOperandType::IMM) {
        auto imm_value = rvv_input1.value.imm;
        assert(is5bitImm(imm_value) && "the immediate value for vadd should be a 5 bit immediate");

        // vsadd.vi vd, vs, imm
        RvvInstGenerator add_gen("vsadd.vi");
        std::vector<CodegenOpOperand> inputs = rvv_inputs;
        CodegenOpOperand output = rvv_output;

        auto code_ = add_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    } else if (rvv_input1.type == RvvOperandType::XREG) {
        // vsadd.vx vd, vs, rs
        RvvInstGenerator add_gen("vsadd.vx");
        std::vector<CodegenOpOperand> inputs = rvv_inputs;
        CodegenOpOperand output = rvv_output;

        auto code_ = add_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    } else if (rvv_input1.type == RvvOperandType::VREG) {
        // vsadd.vv vd, vs1, vs2
        RvvInstGenerator add_gen("vsadd.vv");
        std::vector<CodegenOpOperand> inputs = rvv_inputs;
        CodegenOpOperand output = rvv_output;

        auto code_ = add_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    } else {
        assert(false && "error path");
    }   
}

void OpCodegen::codegen_vsubOp(mlir::Operation* op) {
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 2 && "vsub op should have two inputs");
    auto rvv_input0 = rvv_inputs.at(0);
    auto rvv_input1 = rvv_inputs.at(1);
    assert(rvv_input0.type == RvvOperandType::VREG && "the first input of vsub op should be a vector register");
    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));
    
    if (rvv_input1.type == RvvOperandType::XREG) {
        // vssub.vx vd, vs, rs
        RvvInstGenerator sub_gen("vssub.vx");
        std::vector<CodegenOpOperand> inputs = rvv_inputs;
        CodegenOpOperand output = rvv_output;

        auto code_ = sub_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    } else if (rvv_input1.type == RvvOperandType::VREG) {
        // vssub.vv vd, vs1, vs2
        RvvInstGenerator sub_gen("vssub.vv");
        std::vector<CodegenOpOperand> inputs = rvv_inputs;
        CodegenOpOperand output = rvv_output;

        auto code_ = sub_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    } else {
        assert(false && "error path");
    }
}

void OpCodegen::codegen_vmulOp(mlir::Operation* op) {
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 2 && "vmul op should have two inputs");
    auto rvv_input0 = rvv_inputs.at(0);
    auto rvv_input1 = rvv_inputs.at(1);
    assert(rvv_input0.type == RvvOperandType::VREG && "the first input of vmul op should be a vector register");
    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));

    auto input0_bit_width = rvv_input0.bit_width;
    auto input1_bit_width = rvv_input1.bit_width;

    std::string inst_name = "";
    std::vector<CodegenOpOperand> inputs = rvv_inputs;
    if (rvv_input1.type == RvvOperandType::VREG) {
        if (input0_bit_width == 8 && input1_bit_width == 8) {
            inst_name = "vmul8.vv";
        } else if (is_i16(input0_bit_width) && is_i16(input1_bit_width)) {
            inst_name = "vmul16.vv";
        } else if (is_i16(input0_bit_width) && input1_bit_width == 8) {
            inst_name = "vmul16.vf2";
        } else if (input0_bit_width == 8 && is_i16(input1_bit_width)) {
            // 交换左右操作数
            inputs = {rvv_input1, rvv_input0};
            inst_name = "vmul16.vf2";
        } else {
            assert(false && "unsupported bit width combination for vmul.vv");
        }
    } else if (rvv_input1.type == RvvOperandType::XREG) {
        if (input0_bit_width == 8) {
            inst_name = "vmul16.xvf2";
            // 需要交换操作数, scalar在前
            inputs = {rvv_input1, rvv_input0};
        } else if (is_i16(input0_bit_width)) {
            inst_name = "vmul16.vx";
        } else {
            assert(false && "unsupported bit width combination for vmul.vx");
        }
    } else {
        assert(false && "the second input of vmul op should be either a vector register or a scalar register");
    }
    
    RvvInstGenerator mul_gen(inst_name);
    CodegenOpOperand output = rvv_output;

    auto code_ = mul_gen.codegen(inputs, output);
    getCodeContainer().push_back(code_);
}

void OpCodegen::codegen_vdivOp(mlir::Operation* op) {
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 2 && "vdiv op should have two inputs");
    auto rvv_input0 = rvv_inputs.at(0);
    auto rvv_input1 = rvv_inputs.at(1);
    assert(rvv_input0.type == RvvOperandType::VREG && "the first input of vdiv op should be a vector register");
    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));
    
    if (rvv_input1.type == RvvOperandType::XREG) {
        // vdiv.vx vd, vs, rs
        RvvInstGenerator div_gen("vdiv.vx");
        std::vector<CodegenOpOperand> inputs = rvv_inputs;
        CodegenOpOperand output = rvv_output;

        auto code_ = div_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    } else if (rvv_input1.type == RvvOperandType::VREG) {
        // vdiv.vv vd, vs1, vs2
        RvvInstGenerator div_gen("vdiv.vv");
        std::vector<CodegenOpOperand> inputs = rvv_inputs;
        CodegenOpOperand output = rvv_output;

        auto code_ = div_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    } else {
        assert(false && "error path");
    }
}

void OpCodegen::codegen_vmaxOp(mlir::Operation* op) {
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 2 && "vmax op should have two inputs");
    auto rvv_input0 = rvv_inputs.at(0);
    auto rvv_input1 = rvv_inputs.at(1);
    assert(rvv_input0.type == RvvOperandType::VREG && "the 1st input should be a vector register");
    assert(rvv_input1.type == RvvOperandType::XREG && "the 2nd input should be a scalar register");
    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));

    std::string inst_name = "vmax.vx";
    auto inputs = rvv_inputs;
    auto output = rvv_output;

    RvvInstGenerator max_gen(inst_name);
    auto code_ = max_gen.codegen(inputs, output);
    getCodeContainer().push_back(code_);
}

void OpCodegen::codegen_vminOp(mlir::Operation* op) {
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 2 && "vmin op should have two inputs");
    auto rvv_input0 = rvv_inputs.at(0);
    auto rvv_input1 = rvv_inputs.at(1);
    assert(rvv_input0.type == RvvOperandType::VREG && "the 1st input should be a vector register");
    assert(rvv_input1.type == RvvOperandType::XREG && "the 2nd input should be a scalar register");
    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));

    std::string inst_name = "vmin.vx";
    auto inputs = rvv_inputs;
    auto output = rvv_output;

    RvvInstGenerator min_gen(inst_name);
    auto code_ = min_gen.codegen(inputs, output);
    getCodeContainer().push_back(code_);
}

void OpCodegen::codegen_vabsOp(mlir::Operation* op) {
    /*
     * x = v4
     * vsext.vf#(vsext_width) v5, v4
     * addi x20, x0, -1
     * vmul#(vmul_width).vx v10, v4, x20
     * vmslt.vx v0, v5, x0
     * vmerge.vvm v6, v5, v10, v0
     * vmv.v.i v11, 1
     * vquant_i32ti#(vquant_output_width).vv v24, v6, v11
     */

    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 2 && "vmax op should have two inputs");
    auto rvv_input0 = rvv_inputs.at(0);
    assert(rvv_input0.type == RvvOperandType::VREG && "the 1st input should be a vector register");
    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));

    auto input_bitwidth = rvv_input0.bit_width;
    // 其它类型需要特殊逻辑
    assert((input_bitwidth == 8 || input_bitwidth == 10 || input_bitwidth == 16) && "only support 8/10/16 bit width for abs op");

    auto x_extend32_output_vreg = RegManager::get().allocVectorReg();
    {
        // "vsext.vf#(vsext_width) v5, v4"
        const std::string inst_name = (input_bitwidth == 8) ? "vsext.vf4" : "vsext.vf2";
        std::vector<CodegenOpOperand> inputs = {rvv_input0};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = x_extend32_output_vreg}};

        RvvInstGenerator extend_gen(inst_name);
        auto code_extend = extend_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_extend);
    }
    
    auto neg_1_scalar_reg = RegManager::get().allocScalarReg();
    {
        // "addi x20, x0, -1"
        RvvInstGenerator addi_gen("addi");
        CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = RegManager::get().allocSpecificScalarReg(VEU_RESERVED_ZERO_VALUE_XREG_ID)}};
        CodegenOpOperand arg1 = {.type = RvvOperandType::IMM, .value = {.imm = static_cast<uint32_t>(-1)}};
        std::vector<CodegenOpOperand> inputs_addi = {arg0, arg1};
        CodegenOpOperand output_addi = {.type = RvvOperandType::XREG, .value = {.x_reg = neg_1_scalar_reg}};

        auto code_addi = addi_gen.codegen(inputs_addi, output_addi);
        getCodeContainer().push_back(code_addi);
    }

    auto inverse_x_vreg = RegManager::get().allocVectorReg();
    {
        // "vmul#(vmul_width).vx v10, v4, x20"
        std::string vmul_inst_name = (input_bitwidth == 8) ? "vmul8.vx" : "vmul16.vx";
        RvvInstGenerator vmul_gen(vmul_inst_name);
        auto arg0 = rvv_input0;
        CodegenOpOperand arg1 = {.type = RvvOperandType::XREG, .value = {.x_reg = neg_1_scalar_reg}};
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = inverse_x_vreg}};
        auto code_vmul = vmul_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_vmul);
    }

    {
        // "vmslt.vx v0, v5, x0"
        CodegenOpOperand arg0 = {.type = RvvOperandType::VREG, .value = {.v_reg = x_extend32_output_vreg}};
        CodegenOpOperand arg1 = {.type = RvvOperandType::XREG, .value = {.x_reg = RegManager::get().allocSpecificScalarReg(VEU_RESERVED_ZERO_VALUE_XREG_ID)}};
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = RegManager::get().allocSpecificVectorReg(VEU_RESERVED_MASK_VREG_ID)}};
        RvvInstGenerator vmslt_gen("vmslt.vx");
        auto code_vmslt = vmslt_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_vmslt);
    }

    auto abs_i32_output_vreg = RegManager::get().allocVectorReg();
    {
        // "vmerge.vvm v6, v5, v10, [v0]"
        RvvInstGenerator vmerge_gen("vmerge.vvm");
        CodegenOpOperand arg0 = {.type = RvvOperandType::VREG, .value = {.v_reg = x_extend32_output_vreg}};
        CodegenOpOperand arg1 = {.type = RvvOperandType::VREG, .value = {.v_reg = inverse_x_vreg}};
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = abs_i32_output_vreg}};
        
        auto code_vmerge = vmerge_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_vmerge);
    }

    auto quant_param_vreg = RegManager::get().allocVectorReg();
    {
        // "vmv.v.i v11, 1"
        RvvInstGenerator vmv_gen("vmv.v.i");
        CodegenOpOperand arg0 = {.type = RvvOperandType::IMM, .value = {.imm = static_cast<uint32_t>(1)}};
        std::vector<CodegenOpOperand> inputs = {arg0};
        CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = quant_param_vreg}};

        auto code_ = vmv_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_);
    }

    {
        // "vquant_i32ti#(vquant_output_width).vv v24, v6, v11"
        auto inst_name = (input_bitwidth == 8) ? "vquant_i32ti8.vv" : (input_bitwidth == 10) ? "vquant_i32ti10.vv" : "vquant_i32ti16.vv";
        RvvInstGenerator vquant_gen(inst_name);
        CodegenOpOperand arg0 = {.type = RvvOperandType::VREG, .value = {.v_reg = abs_i32_output_vreg}};
        CodegenOpOperand arg1 = {.type = RvvOperandType::VREG, .value = {.v_reg = quant_param_vreg}};
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = rvv_output;
        
        auto code_vquant = vquant_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_vquant);
    }
}

void OpCodegen::codegen_vquantOp(mlir::Operation* op) {
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 2 && "vquant op should have two inputs");
    auto rvv_input0 = rvv_inputs.at(0);
    auto rvv_input1 = rvv_inputs.at(1);
    assert(rvv_input0.type == RvvOperandType::VREG && "the 1st input should be a vector register");
    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));

    // "vquant_i8ti32.vv", "vquant_i16ti32.vv", "vquant_i32ti10.vv", "vquant_i32ti16.vv", "vquant_i32ti8.vv"
    auto input_bitwidth = rvv_input0.bit_width;
    auto output_bitwidth = rvv_output.bit_width;
    bool constraint_0 = ((input_bitwidth == 8 || is_i16(input_bitwidth)) && output_bitwidth == 32);
    bool constraint_1 = (input_bitwidth == 32 && (output_bitwidth == 8 || output_bitwidth == 10 || output_bitwidth == 16));
    assert((constraint_0 || constraint_1) && "unsupported bit width combination for quant op");

    auto real_rvv_input1 = rvv_input1;
    if (rvv_input1.type == RvvOperandType::XREG) {
        // "vmv.v.x v11, x20"
        RvvInstGenerator vmv_vx_gen("vmv.v.x");
        CodegenOpOperand arg0 = rvv_input1;
        std::vector<CodegenOpOperand> inputs = {arg0};
        CodegenOpOperand vmv_vx_output = {.type = RvvOperandType::VREG, .value = {.v_reg = RegManager::get().allocVectorReg()}};
        auto code_vmv_vx = vmv_vx_gen.codegen(inputs, vmv_vx_output);
        getCodeContainer().push_back(code_vmv_vx);
        real_rvv_input1 = vmv_vx_output;
    }

    auto inst_name = "vquant_i" + (is_i16(input_bitwidth) ? "16" : std::to_string(input_bitwidth)) + 
                           "ti" + std::to_string(output_bitwidth) + ".vv";
    RvvInstGenerator vquant_gen(inst_name);
    std::vector<CodegenOpOperand> inputs = {rvv_input0, real_rvv_input1};
    auto output = rvv_output;
    auto code_vquant = vquant_gen.codegen(inputs, output);
    getCodeContainer().push_back(code_vquant);
}

void OpCodegen::codegen_vqlutOp(mlir::Operation* op) {
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 3 && "vqlut op should have 3 inputs");
    auto rvv_input0 = rvv_inputs.at(0);
    auto rvv_input1 = rvv_inputs.at(1);
    auto rvv_input2 = rvv_inputs.at(2);
    assert(rvv_input0.type == RvvOperandType::VREG && "the 1st input should be a vector register");
    assert(rvv_input2.type == RvvOperandType::XREG && "the 3rd input should be a scalar register");
    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));

    auto table_size = llvm::dyn_cast<xvec::Vqlut>(op).getTableSize();
    auto elem_num_ = table_size;
    auto bit_width_ = rvv_output.bit_width;
    assert((bit_width_ == 8 || bit_width_ == 10) && "the data type of the lut table should be i8 or i10");
    assert((elem_num_ == 256 || elem_num_ == 1024) && "the number of elements in the lut table should be 256 or 1024");
    std::string i_str = (elem_num_ == 256) ? "i8" : "i10";
    std::string d_str = (bit_width_ == 8) ? "d8" : "d10";

    // load_lut
    {
        std::string load_lut_inst_name = "load_lut_" + i_str + "_" + d_str;
        RvvInstGenerator load_lut_gen(load_lut_inst_name);
        CodegenOpOperand arg0 = rvv_input2;
        std::vector<CodegenOpOperand> inputs_load_lut = {arg0};
        CodegenOpOperand output_load_lut = {.type = RvvOperandType::NONE};
        
        auto code_load_lut = load_lut_gen.codegen(inputs_load_lut, output_load_lut);
        prelogue_code_.push_back(code_load_lut);
    }

    auto real_rvv_input1 = rvv_input1;
    if (rvv_input1.type == RvvOperandType::XREG) {
        // "vmv.v.x v11, x20"
        RvvInstGenerator vmv_vx_gen("vmv.v.x");
        CodegenOpOperand arg0 = rvv_input1;
        std::vector<CodegenOpOperand> inputs = {arg0};
        CodegenOpOperand vmv_vx_output = {.type = RvvOperandType::VREG, .value = {.v_reg = RegManager::get().allocVectorReg()}};
        auto code_vmv_vx = vmv_vx_gen.codegen(inputs, vmv_vx_output);
        getCodeContainer().push_back(code_vmv_vx);
        real_rvv_input1 = vmv_vx_output;
    }

    // vsfu
    {
        std::string vsfu_inst_name = "vsfu_" + i_str + "_" + d_str + ".vv";
        RvvInstGenerator vsfu_gen(vsfu_inst_name);
        std::vector<CodegenOpOperand> inputs = {rvv_input0, real_rvv_input1};
        auto output = rvv_output;
        auto code_vsfu = vsfu_gen.codegen(inputs, output);
        getCodeContainer().push_back(code_vsfu);
    }
}

void OpCodegen::codegen_loopBeginOp(mlir::Operation* op) {
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 1 && "loopBegin op should have one inputs");
    auto rvv_input0 = rvv_inputs.at(0);
    assert(rvv_input0.type == RvvOperandType::XREG && "the 1st input should be a scalar register");

    auto loop_id = llvm::dyn_cast<xvec::LoopBegin>(op).getLoopId();
    CodegenOpOperand rvv_input1 = {.type = RvvOperandType::IMM, .value = {.imm = loop_id}};

    RvvInstGenerator loop_cfgx_gen("loop_cfgx");
    CodegenOpOperand arg0 = rvv_input0;
    CodegenOpOperand arg1 = rvv_input1;
    std::vector<CodegenOpOperand> inputs = {arg0, arg1};
    CodegenOpOperand output = {.type = RvvOperandType::NONE};

    auto code_ = loop_cfgx_gen.codegen(inputs, output);
    getCodeContainer().push_back(code_);
}

void OpCodegen::codegen_loopEndOp(mlir::Operation* op) {
    auto loop_id = llvm::dyn_cast<xvec::LoopEnd>(op).getLoopId();
    CodegenOpOperand rvv_input0 = {.type = RvvOperandType::IMM, .value = {.imm = loop_id}};

    RvvInstGenerator loop_end_gen("loop_end");
    CodegenOpOperand arg0 = rvv_input0;
    std::vector<CodegenOpOperand> inputs = {arg0};
    CodegenOpOperand output = {.type = RvvOperandType::NONE};

    auto code_ = loop_end_gen.codegen(inputs, output);
    getCodeContainer().push_back(code_);
}

void OpCodegen::codegen_rvvConfigOp(mlir::Operation* op) {
    auto rvv_inputs = xvecOpInputs2RvvInstInputs(op);
    assert(rvv_inputs.size() == 1 && "loopBegin op should have one inputs");
    auto rvv_input0 = rvv_inputs.at(0);
    assert(rvv_input0.type == RvvOperandType::XREG && "the 1st input should be a scalar register");

    auto rvv_output = xvectorOperand2RvvOperand(op->getResult(0));

    RvvInstGenerator vsetvli_gen("vsetvli");
    CodegenOpOperand arg0 = rvv_input0;
    std::vector<CodegenOpOperand> inputs = {arg0};
    CodegenOpOperand output = rvv_output;

    auto code_ = vsetvli_gen.codegen(inputs, output);
    getCodeContainer().push_back(code_);
}


} // namespace xp_std
} // namespace xp_mlir