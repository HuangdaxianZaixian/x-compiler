#include "tmu_fuse/OpCodegen.hpp"
#include "tmu_fuse/FuseOpInfo.hpp"
#include "tmu_fuse/RvvInstGen.hpp"
#include "tmu_fuse/RegManager.hpp"
#include "xp_mlir/Dialect/XMA/IR/XMAOps.h"
#include <utility>

namespace xp_mlir {
namespace xp_tmu {
ScalarReg OpCodegen::genScalarLoadCode(uint32_t scalar, std::vector<RvvInst>& code_container) {
    // "lui x18, #(lut_address)"
    // "addiu x18, x18, #(lut_address)"
    auto rd = RegManager::get().allocScalarReg();
    {
        RvvInstGenerator lui_gen("lui");
        CodegenOpOperand arg0 = {.type = RvvOperandType::IMM, .value = {.imm = scalar}};
        std::vector<CodegenOpOperand> inputs = {arg0};
        CodegenOpOperand output = {.type = RvvOperandType::XREG, .value = {.x_reg = rd}};

        auto code_ = lui_gen.codegen(inputs, output);
        code_container.push_back(code_);
    }

    {
        RvvInstGenerator lui_gen("addiu");
        CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = rd}};
        CodegenOpOperand arg1 = {.type = RvvOperandType::IMM, .value = {.imm = scalar}};
        std::vector<CodegenOpOperand> inputs = {arg0, arg1};
        CodegenOpOperand output = {.type = RvvOperandType::XREG, .value = {.x_reg = rd}};

        auto code_ = lui_gen.codegen(inputs, output);
        code_container.push_back(code_);
    }

    return rd;
}

bool is_i16(int32_t bit_width) {
    return bit_width >= 10 && bit_width <= 16;
};

bool is_virtual_vector_register(const VectorReg& reg) {
    auto reg_id = reg.id;
    assert(reg_id != 26 && reg_id != 27 && reg_id != 28 && "illegal virtual vector register id");
    return reg_id >= 23 && reg_id <= 31;
}

ScalarReg OpCodegen::codegen_addi_internal(ScalarReg& input, int32_t scalar, std::vector<RvvInst>& code_container) {
    auto rd = RegManager::get().allocScalarReg();
    
    RvvInstGenerator addi_gen("addi");
    CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = input}};
    CodegenOpOperand arg1 = {.type = RvvOperandType::IMM, .value = {.imm = static_cast<uint32_t>(scalar)}};
    std::vector<CodegenOpOperand> inputs_addi = {arg0, arg1};
    CodegenOpOperand output_addi = {.type = RvvOperandType::XREG, .value = {.x_reg = rd}};

    auto code_addi = addi_gen.codegen(inputs_addi, output_addi);
    code_container.push_back(code_addi);

    return rd; 
}

VectorReg OpCodegen::codegen_vmv_vv_internal(VectorReg& input, std::vector<RvvInst>& code_container) {
    // "vmv.v.v v4, v31"
    auto vd = RegManager::get().allocVectorReg();
    RvvInstGenerator vmv_gen("vmv.v.v");
    CodegenOpOperand arg0 = {.type = RvvOperandType::VREG, .value = {.v_reg = input}};
    std::vector<CodegenOpOperand> inputs = {arg0};
    CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = vd}};

    auto code_ = vmv_gen.codegen(inputs, output);
    code_container.push_back(code_);
    return vd;
}

VectorReg OpCodegen::codegen_vmv_vi_internal(int32_t scalar, std::vector<RvvInst>& code_container) {
    // "vmv.v.i v11, 1"
    auto vd = RegManager::get().allocVectorReg();
    RvvInstGenerator vmv_gen("vmv.v.i");
    CodegenOpOperand arg0 = {.type = RvvOperandType::IMM, .value = {.imm = static_cast<uint32_t>(scalar)}};
    std::vector<CodegenOpOperand> inputs = {arg0};
    CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = vd}};

    auto code_ = vmv_gen.codegen(inputs, output);
    code_container.push_back(code_);
    return vd;
}

CodegenOpOperand OpCodegen::codegen_extend_internal(CodegenOpOperand& input, int32_t input_bitwidth, std::vector<RvvInst>& code_container) {
    // "vsext.vf#(C) v25, v2"
    assert((input_bitwidth == 8 || is_i16(input_bitwidth)) && "only support extending i8 or i16 input for extend op");
    const std::string inst_name = (input_bitwidth == 8) ? "vsext.vf4" : "vsext.vf2";

    CodegenOpOperand output = {
        .type = RvvOperandType::VREG, 
        .value = {.v_reg = {RegManager::get().allocVectorReg().id, RegStatus::USED}}
    };

    RvvInstGenerator extend_gen(inst_name);
    auto code_extend = extend_gen.codegen({input}, output);
    code_container.push_back(code_extend);

    return output;
}

RvvInst OpCodegen::genNopRvvInst() {
    RvvInst inst;
    inst.raw = 0x00000013; // addi x0, x0, 0
    inst.assembly = "nop";
    inst.slot_supportion = VeuInstSlotSupportion::SLOT_0_1; // nop指令可以和任意slot的指令打包

    CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = {0, RegStatus::USED}}};
    CodegenOpOperand arg1 = {.type = RvvOperandType::IMM, .value = {.imm = 0}};
    inst.inputs = {arg0, arg1};
    inst.output = arg0;

    return inst;
}

RvvInst OpCodegen::getEmptyVecCoreModeCode(int32_t tmu_output_bitwidth) {
    // vmv.v.v vd,vs1 (vs1=31,vd=25)
    RvvInstGenerator vmv_gen("vmv.v.v");
    CodegenOpOperand arg0 = {.type = RvvOperandType::VREG, .value = {.v_reg = {31, RegStatus::USED}}};
    std::vector<CodegenOpOperand> inputs = {arg0};

    assert(tmu_output_bitwidth <= 32 && "the output bitwidth of tmu fuse region must be less than or equal to 32");
    int32_t output_reg_id = (tmu_output_bitwidth == 32) ? 25 : 24;
    CodegenOpOperand output = {.type = RvvOperandType::VREG, .value = {.v_reg = {output_reg_id, RegStatus::USED}}};

    auto code_ = vmv_gen.codegen(inputs, output);
    return code_;
}

OpCodegen::OpCodegen(const std::vector<mlir::Operation*>& ops, const std::vector<FuseOpInfo>& fuse_op_infos, CodegenOpType codegen_type) : 
    codegen_type_(codegen_type),
    ops_(ops), 
    fuse_op_infos_(fuse_op_infos) {
    if (codegen_type_ == CodegenOpType::QUANT_LUT) {
        assert(fuse_op_infos.size() == 2 && "quant-lut pattern should have two ops");
        assert(llvm::isa<xma::QuantOp>(ops_.at(0)) && llvm::isa<xma::LutOp>(ops_.at(1)) && "only support quant-lut pattern for now");
    } else if (codegen_type_ == CodegenOpType::MATMULTC_QUANT_EXTEND) {
        assert(fuse_op_infos.size() == 2 && "matmultc-quant-extend pattern should have two ops");
        assert(llvm::isa<xma::QuantOp>(ops_.at(0)) && llvm::isa<xma::ExtI32Op>(ops_.at(1)) && "only support matmultc-quant-extend pattern for now");
    } else {
        assert(fuse_op_infos.size() == 1 && "general pattern should have one op");
    } 
}

void OpCodegen::codegen() {
    std::vector<CodegenOpOperand> inputs_;
    CodegenOpOperand output_;

    std::vector<OpOperandInfo> input_infos_ = fuse_op_infos_.at(0).input_infos;
    OpOperandInfo output_info_ = fuse_op_infos_.at(0).output_info;
    if (codegen_type_ == CodegenOpType::QUANT_LUT) {
        // quant和extend中间默认走v23
        input_infos_ = fuse_op_infos_.at(0).input_infos;
        output_info_ =fuse_op_infos_.at(1).output_info;
    }

    if (codegen_type_ == CodegenOpType::MATMULTC_QUANT_EXTEND) {
        // quant x, param
        input_infos_ = {fuse_op_infos_.at(0).input_infos.at(0), fuse_op_infos_.at(0).input_infos.at(1)};
        output_info_ =fuse_op_infos_.at(1).output_info;
    }

    // 生成输入输出寄存器信息
    // 1. 如果是xreg, 则生成load代码将scalar加载到xreg中
    for (auto& input_info : input_infos_) {
        assert(!input_info.is_none && "error input with none value is not expected in codegen");
        assert((input_info.mem_type != OpOperandMemType::UNKONWN || input_info.mem_type != OpOperandMemType::L1) && "error mem type in codegen");

        if (input_info.mem_type == OpOperandMemType::V_REG) {
            auto v_reg_id = input_info.mem_value.v_reg_id;
            VectorReg v_reg = {v_reg_id, RegStatus::USED};
            OperandWrapper operand_wrapper = {.v_reg = v_reg}; 
            CodegenOpOperand operand_ = {.type = RvvOperandType::VREG, .value = operand_wrapper};
            inputs_.push_back(operand_);
        } else if (input_info.mem_type == OpOperandMemType::X_REG) {
            auto scalar_reg = genScalarLoadCode(input_info.mem_value.x_reg_scalar, prelogue_code_);
            OperandWrapper operand_wrapper = {.x_reg = scalar_reg}; 
            CodegenOpOperand operand_ = {.type = RvvOperandType::XREG, .value = operand_wrapper};
            inputs_.push_back(operand_);
        } else {
            assert(input_info.mem_type == OpOperandMemType::IMM && "must be immediate operand");
            OperandWrapper operand_wrapper = {.imm = input_info.mem_value.imm}; 
            CodegenOpOperand operand_ = {.type = RvvOperandType::IMM, .value = operand_wrapper};
            inputs_.push_back(operand_);
        }
    } 

    {
        assert(output_info_.mem_type ==OpOperandMemType::V_REG && "only support register output");
        auto v_reg_id = output_info_.mem_value.v_reg_id;
        VectorReg v_reg = {v_reg_id, RegStatus::USED};
        OperandWrapper operand_wrapper = {.v_reg = v_reg}; 
        CodegenOpOperand operand_ = {.type = RvvOperandType::VREG, .value = operand_wrapper};
        output_ = operand_;
    }

    if (codegen_type_ == CodegenOpType::QUANT_LUT) {
        codegen_quant_lut_op(inputs_, output_);
        return;
    } 

    if (codegen_type_ == CodegenOpType::MATMULTC_QUANT_EXTEND) {
        codegen_quant_extend(inputs_, output_);
        return;
    }

    auto op_ = ops_.at(0);
    if (llvm::isa<xma::AddOp, xma::SubOp, xma::DivOp, xma::MulOp>(op_)) {
        codegen_eletwise_op(inputs_, output_);
        return;
    }

    if (llvm::isa<xma::MaxOp>(op_)) {
        codegen_max(inputs_, output_);
        return;
    }

    if (llvm::isa<xma::MinOp>(op_)) {
        codegen_min(inputs_, output_);
        return;
    }

    if (llvm::isa<xma::AbsOp>(op_)) {
        codegen_abs(inputs_, output_);
        return;
    }

    if (llvm::isa<xma::ExtI32Op>(op_)) {
        codegen_extend(inputs_, output_);
        return;
    }

    if (llvm::isa<xma::QuantOp, xma::DeQuantOp>(op_)) {
        codegen_quant(inputs_, output_);
        return;
    }

    assert(false && "unsupported op type for codegen");
}

void OpCodegen::codegen_eletwise_op(std::vector<CodegenOpOperand>& inputs, CodegenOpOperand& output) {
    auto op_ = ops_.at(0);
    assert(inputs.size() == 2 && "element-wise op should have two inputs");

    if (llvm::isa<xma::AddOp, xma::SubOp, xma::DivOp>(op_)) {
        // assert(input0_bit_width == 32 && input1_bit_width == 32 && "only support 32 bit width for add/sub/div op");
        for (int32_t i = 0; i < inputs.size(); ++i) {
            auto input_bitwidth = fuse_op_infos_.at(0).input_infos.at(i).bit_width;
            if (input_bitwidth != 32 && inputs.at(i).type == RvvOperandType::VREG) {
                // 插入extend指令
                inputs.at(i) = codegen_extend_internal(inputs.at(i), input_bitwidth, loop_code_);
            }
        }
    }

    auto input0_type = inputs.at(0).type;
    auto input1_type = inputs.at(1).type;
    assert(input0_type == RvvOperandType::VREG && "the input should be in v register");
    assert((input1_type == RvvOperandType::VREG || input1_type == RvvOperandType::XREG) && "the second input should be in v register or x register");

    std::string inst_name = "";
    if (llvm::isa<xma::AddOp>(op_)) {
        inst_name = (input1_type == RvvOperandType::VREG) ? "vsadd.vv" : "vsadd.vx";
    }
    if (llvm::isa<xma::SubOp>(op_)) {
        inst_name = (input1_type == RvvOperandType::VREG) ? "vssub.vv" : "vssub.vx";
    } 
    if (llvm::isa<xma::DivOp>(op_)) {
        inst_name = (input1_type == RvvOperandType::VREG) ? "vdiv.vv" : "vdiv.vx"; 
    }

    if (llvm::isa<xma::MulOp>(op_)) {
        auto input0_bit_width = fuse_op_infos_.at(0).input_infos.at(0).bit_width;
        auto input1_bit_width = fuse_op_infos_.at(0).input_infos.at(1).bit_width;

        if (input1_type == RvvOperandType::VREG) {
            if (input0_bit_width == 8 && input1_bit_width == 8) {
                inst_name = "vmul8.vv";
            } else if (is_i16(input0_bit_width) && is_i16(input1_bit_width)) {
                inst_name = "vmul16.vv";
            } else if (is_i16(input0_bit_width) && input1_bit_width == 8) {
                inst_name = "vmul16.vf2";
            } else if (input0_bit_width == 8 && is_i16(input1_bit_width)) {
                // 交换左右操作数
                std::swap(inputs.at(0), inputs.at(1));
                inst_name = "vmul16.vf2";
            } else {
                assert(false && "unsupported bit width combination for vmul.vv");
            }
        } else {
            // 前端保证scalar的最大值不超过int16, 所以统一使用int16表示
            auto scalar = fuse_op_infos_.at(0).input_infos.at(1).mem_value.x_reg_scalar;
            assert((scalar & 0xffff) == scalar && "the scalar value for vmul.vx should fit in 16 bits");

            if (input0_bit_width == 8) {
                inst_name = "vmul16.xvf2";
                // 需要交换操作数, scalar在前
                std::swap(inputs.at(0), inputs.at(1));
            } else if (is_i16(input0_bit_width)) {
                inst_name = "vmul16.vx";
            } else {
                assert(false && "unsupported bit width combination for vmul.vx");
            }
        }
    }

    RvvInstGenerator eletw_gen(inst_name);
    auto code_eletw = eletw_gen.codegen(inputs, output);
    loop_code_.push_back(code_eletw);
}

void OpCodegen::codegen_quant_lut_op(std::vector<CodegenOpOperand>& inputs, CodegenOpOperand& output) {
    // lut table需要load_lut
    auto lut_table = fuse_op_infos_.at(1).input_infos.at(1);
    assert(lut_table.mem_type == OpOperandMemType::X_REG && "the lut table input must be loaded to scalar register");
    auto lut_table_scalar_reg = genScalarLoadCode(lut_table.mem_value.x_reg_scalar, prelogue_code_);

    auto elem_num_ = lut_table.elem_num;
    auto bit_width_ = lut_table.bit_width;
    assert((bit_width_ == 8 || bit_width_ == 10) && "the data type of the lut table should be i8 or i10");
    assert((elem_num_ == 256 || elem_num_ == 1024) && "the number of elements in the lut table should be 256 or 1024");
    std::string i_str = (elem_num_ == 256) ? "i8" : "i10";
    std::string d_str = (bit_width_ == 8) ? "d8" : "d10";
    std::string load_lut_inst_name = "load_lut_" + i_str + "_" + d_str;
    RvvInstGenerator load_lut_gen(load_lut_inst_name);
    CodegenOpOperand arg0 = {.type = RvvOperandType::XREG, .value = {.x_reg = lut_table_scalar_reg}};
    std::vector<CodegenOpOperand> inputs_load_lut = {arg0};
    CodegenOpOperand output_load_lut = {.type = RvvOperandType::NONE};

    auto code_load_lut = load_lut_gen.codegen(inputs_load_lut, output_load_lut);
    prelogue_code_.push_back(code_load_lut);

    // vsfu
    std::string vsfu_inst_name = "vsfu_" + i_str + "_" + d_str + ".vv";
    RvvInstGenerator vsfu_gen(vsfu_inst_name);
    auto code_vsfu = vsfu_gen.codegen(inputs, output);
    loop_code_.push_back(code_vsfu);
}

void OpCodegen::codegen_max(std::vector<CodegenOpOperand>& inputs, CodegenOpOperand& output) {
    assert(inputs.size() == 2 && "max op should have two inputs");

    // vmax.vx vd,vs2,rs1,vm
    auto input0_type = inputs.at(0).type;
    auto input1_type = inputs.at(1).type;

    assert(input0_type == RvvOperandType::VREG && "the input should be in v register");
    assert(input1_type == RvvOperandType::XREG && "the second input should be scalar xreg for max op");

    std::string inst_name = "vmax.vx";

    RvvInstGenerator max_gen(inst_name);
    auto code_max = max_gen.codegen(inputs, output);
    loop_code_.push_back(code_max);
}

void OpCodegen::codegen_min(std::vector<CodegenOpOperand>& inputs, CodegenOpOperand& output) {
    assert(inputs.size() == 2 && "min op should have two inputs");

    // vmin.vx vd,vs2,rs1,vm
    auto input0_type = inputs.at(0).type;
    auto input1_type = inputs.at(1).type;

    assert(input0_type == RvvOperandType::VREG && "the input should be in v register");
    assert(input1_type == RvvOperandType::XREG && "the second input should be scalar xreg for min op");

    std::string inst_name = "vmin.vx";

    RvvInstGenerator min_gen(inst_name);
    auto code_min = min_gen.codegen(inputs, output);
    loop_code_.push_back(code_min);
}

void OpCodegen::codegen_abs(std::vector<CodegenOpOperand>& inputs, CodegenOpOperand& output) {
    // "addi x20, x0, -1"

    // "vmv.v.v v4, v31"
    // "vsext.vf#(vsext_width) v5, v4"
    // "vmul#(vmul_width).vx v10, v4, x20"
    // "vmslt.vx v0, v5, x0"
    // "vmerge.vvm v6, v5, v10, v0"
    // "vmv.v.i v11, 1"
    // "vquant_i32ti#(vquant_output_width).vv v24, v6, v11"

    assert(inputs.size() == 1 && "abs op should have one input");
    auto input0_type = inputs.at(0).type;
    assert(input0_type == RvvOperandType::VREG && "the input should be in v register");
    auto input_bitwidth = fuse_op_infos_.at(0).input_infos.at(0).bit_width;
    // 其它类型需要特殊逻辑
    assert((input_bitwidth == 8 || input_bitwidth == 10 || input_bitwidth == 16) && "only support 8/10/16 bit width for abs op");
    auto output_bitwidth = fuse_op_infos_.at(0).output_info.bit_width;
    assert(output_bitwidth == input_bitwidth && "the output bit width should be the same as input for abs op"); 

    // "addi x20, x0, -1"
    auto scalar_reg_0 = RegManager::get().getScalarReg0();
    auto neg_1_scalar_reg = codegen_addi_internal(scalar_reg_0, -1, prelogue_code_);

    // "vmv.v.v v4, v31"
    // 防止virtual register被多次使用
    auto input_vreg = inputs.at(0).value.v_reg;
    if (is_virtual_vector_register(input_vreg)) {
        input_vreg = codegen_vmv_vv_internal(inputs.at(0).value.v_reg, loop_code_);
    }

    // "vsext.vf#(vsext_width) v5, v4"
    CodegenOpOperand i32_extend_input = {.type = RvvOperandType::VREG, .value = {.v_reg = input_vreg}};
    auto i32_extend_output = codegen_extend_internal(i32_extend_input, input_bitwidth, loop_code_);

    // "vmul#(vmul_width).vx v10, v4, x20"
    CodegenOpOperand mul_neg_1_input0 = {.type = RvvOperandType::VREG, .value = {.v_reg = input_vreg}};
    CodegenOpOperand mul_neg_1_input1 = {.type = RvvOperandType::XREG, .value = {.x_reg = neg_1_scalar_reg}};
    CodegenOpOperand mul_neg_1_output = {.type = RvvOperandType::VREG, .value = {.v_reg = RegManager::get().allocVectorReg()}};
    std::string vmul_inst_name = (input_bitwidth == 8) ? "vmul8.vx" : "vmul16.vx";
    RvvInstGenerator vmul_gen(vmul_inst_name);
    auto code_vmul = vmul_gen.codegen({mul_neg_1_input0, mul_neg_1_input1}, mul_neg_1_output);
    loop_code_.push_back(code_vmul);

    // "vmslt.vx v0, v5, x0"
    CodegenOpOperand vmslt_input0 = i32_extend_output;
    CodegenOpOperand vmslt_input1 = {.type = RvvOperandType::XREG, .value = {.x_reg = RegManager::get().getScalarReg0()}};
    CodegenOpOperand vmslt_output_v0 = {.type = RvvOperandType::VREG, .value = {.v_reg = RegManager::get().allocSpecificVectorReg(0)}};
    RvvInstGenerator vmslt_gen("vmslt.vx");
    auto code_vmslt = vmslt_gen.codegen({vmslt_input0, vmslt_input1}, vmslt_output_v0);
    loop_code_.push_back(code_vmslt);

    // "vmerge.vvm v6, v5, v10, [v0]"
    CodegenOpOperand vmerge_input0 = i32_extend_output;
    CodegenOpOperand vmerge_input1 = mul_neg_1_output;
    // CodegenOpOperand vmerge_input2 = vmslt_output_v0;
    CodegenOpOperand vmerge_output = {.type = RvvOperandType::VREG, .value = {.v_reg = RegManager::get().allocVectorReg()}};
    RvvInstGenerator vmerge_gen("vmerge.vvm");
    auto code_vmerge = vmerge_gen.codegen({vmerge_input0, vmerge_input1}, vmerge_output);
    loop_code_.push_back(code_vmerge);

    // "vmv.v.i v11, 1"
    VectorReg output_quant_param_vreg = codegen_vmv_vi_internal(1, loop_code_);

    // "vquant_i32ti#(vquant_output_width).vv v24, v6, v11"
    CodegenOpOperand vquant_input0 = vmerge_output;
    CodegenOpOperand vquant_input1 = {.type = RvvOperandType::VREG, .value = {.v_reg = output_quant_param_vreg}};
    CodegenOpOperand vquant_output = output;
    auto output_quant_inst_name = (output_bitwidth == 8) ? "vquant_i32ti8.vv" : (output_bitwidth == 10) ? "vquant_i32ti10.vv" : "vquant_i32ti16.vv";
    RvvInstGenerator vquant_gen(output_quant_inst_name);
    auto code_vquant = vquant_gen.codegen({vquant_input0, vquant_input1}, vquant_output);
    loop_code_.push_back(code_vquant);
}

void OpCodegen::codegen_extend(std::vector<CodegenOpOperand>& inputs, CodegenOpOperand& output) {
    // "vsext.vf#(C) v25, v2"
    assert(inputs.size() == 1 && "extend op should have one input");
    auto input0_type = inputs.at(0).type;
    assert(input0_type == RvvOperandType::VREG && "the input should be in v register");

    auto input0_bit_width = fuse_op_infos_.at(0).input_infos.at(0).bit_width;

    assert((input0_bit_width == 8 || is_i16(input0_bit_width)) && "only support extending i8 or i16 input for extend op");
    const std::string inst_name = (input0_bit_width == 8) ? "vsext.vf4" : "vsext.vf2";

    RvvInstGenerator extend_gen(inst_name);
    auto code_extend = extend_gen.codegen(inputs, output);
    loop_code_.push_back(code_extend);
}

void OpCodegen::codegen_quant(std::vector<CodegenOpOperand>& inputs, CodegenOpOperand& output) {
    assert(inputs.size() == 2 && "quant op should have two inputs");

    auto input0_type = inputs.at(0).type;
    auto input1_type = inputs.at(1).type;
    assert(input0_type == RvvOperandType::VREG && "the input should be in v register");

    // "vquant_i8ti32.vv", "vquant_i16ti32.vv", "vquant_i32ti10.vv", "vquant_i32ti16.vv", "vquant_i32ti8.vv"
    auto input_bitwidth = fuse_op_infos_.at(0).input_infos.at(0).bit_width;
    auto output_bitwidth = fuse_op_infos_.at(0).output_info.bit_width;
    bool constraint_0 = ((input_bitwidth == 8 || is_i16(input_bitwidth)) && output_bitwidth == 32);
    bool constraint_1 = (input_bitwidth == 32 && (output_bitwidth == 8 || output_bitwidth == 10 || output_bitwidth == 16));
    assert((constraint_0 || constraint_1) && "unsupported bit width combination for quant op");


    if (codegen_type_ == CodegenOpType::MATMULTC_NOT_LAST_DIM_QUANT) {
        assert(input1_type == RvvOperandType::VREG && inputs[1].value.v_reg.id == 15 && "the quant param for matmultc_not_last_dim_quant should be in v15");
    } else {
        assert(input1_type == RvvOperandType::IMM && "the quant param for quant op should be immediate");

        // "vmv.v.i v11, 1"
        auto imm_ = inputs.at(1).value.imm;
        VectorReg output_vreg = codegen_vmv_vi_internal(static_cast<int32_t>(imm_), loop_code_);
        inputs.at(1) = {.type = RvvOperandType::VREG, .value = {.v_reg = output_vreg}};
    }

    auto inst_name = "vquant_i" + (is_i16(input_bitwidth) ? "16" : std::to_string(input_bitwidth)) + 
                           "ti" + std::to_string(output_bitwidth) + ".vv";
    RvvInstGenerator vquant_gen(inst_name);
    auto code_vquant = vquant_gen.codegen(inputs, output);
    loop_code_.push_back(code_vquant);
}

void OpCodegen::codegen_quant_extend(std::vector<CodegenOpOperand>& inputs, CodegenOpOperand& output) {
    assert(codegen_type_ == CodegenOpType::MATMULTC_QUANT_EXTEND && "codegen_quant_extend should only be used for matmultc-quant-extend pattern");
    assert(inputs.size() == 2 && "quant-extend should have two inputs");

    auto quant_input0_type = inputs.at(0).type;
    assert(quant_input0_type == RvvOperandType::VREG && "the input should be in v register");

    // "vquant_i8ti32.vv", "vquant_i16ti32.vv", "vquant_i32ti10.vv", "vquant_i32ti16.vv", "vquant_i32ti8.vv"
    auto quant_input_bitwidth = fuse_op_infos_.at(0).input_infos.at(0).bit_width;
    auto quant_output_bitwidth = fuse_op_infos_.at(0).output_info.bit_width;
    bool constraint_0 = ((quant_input_bitwidth == 8 || is_i16(quant_input_bitwidth)) && quant_output_bitwidth == 32);
    bool constraint_1 = (quant_input_bitwidth == 32 && (quant_output_bitwidth == 8 || quant_output_bitwidth == 10 || quant_output_bitwidth == 16));
    assert((constraint_0 || constraint_1) && "unsupported bit width combination for quant op");

    // 先添加extend指令 
    // "addi x20, x0, 1"
    auto scalar_reg_0 = RegManager::get().getScalarReg0();
    auto scalar_one_reg = codegen_addi_internal(scalar_reg_0, 1, prelogue_code_);

    // "vmul#(vmul_width).vx v24/v25, v23, x20"
    CodegenOpOperand mul_input0 = {.type = RvvOperandType::VREG, .value = {.v_reg = {.id = 23, .status = RegStatus::USED}}}; 
    CodegenOpOperand mul_input1 = {.type = RvvOperandType::XREG, .value = {.x_reg = scalar_one_reg}};
    CodegenOpOperand mul_output = output;
    assert(quant_output_bitwidth <= 16 && "the output bitwidth after quant should be less than or equal to 16 bits");
    std::string vmul_inst_name = (quant_output_bitwidth == 8) ? "vmul8.vx" : "vmul16.vx";
    RvvInstGenerator vmul_gen(vmul_inst_name);
    auto code_vmul = vmul_gen.codegen({mul_input0, mul_input1}, mul_output);
    loop_code_.push_back(code_vmul);

    auto inst_name = "vquant_i" + (is_i16(quant_input_bitwidth) ? "16" : std::to_string(quant_input_bitwidth)) + 
                           "ti" + std::to_string(quant_output_bitwidth) + ".vv";
    RvvInstGenerator vquant_gen(inst_name);
    CodegenOpOperand quant_output = {.type = RvvOperandType::VREG, .value = {.v_reg = {.id = 23, .status = RegStatus::USED}}};
    auto code_vquant = vquant_gen.codegen(inputs, quant_output);
    loop_code_.push_back(code_vquant);
}


} // namespace xp_tmu
} // namespace xp_mlir