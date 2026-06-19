#ifndef OP_CODEGEN__HPP
#define OP_CODEGEN__HPP

#include "mlir/IR/Operation.h"
#include "tmu_fuse/FuseOpInfo.hpp"
#include "tmu_fuse/RegManager.hpp"
#include "tmu_fuse/RvvInstGen.hpp"

namespace xp_mlir {
namespace xp_tmu {

enum class CodegenOpType {
    GENERAL = 0,
    QUANT_LUT,
    MATMULTC_NOT_LAST_DIM_QUANT,
    MATMULTC_QUANT_EXTEND
};

class OpCodegen {
public:
    static ScalarReg genScalarLoadCode(uint32_t scalar, std::vector<RvvInst>& code_container);
    static RvvInst genNopRvvInst();
    static RvvInst getEmptyVecCoreModeCode(int32_t tmu_output_bitwidth);
    static CodegenOpOperand codegen_extend_internal(CodegenOpOperand& input, int32_t input_bitwidth, std::vector<RvvInst>& code_container);
    static ScalarReg codegen_addi_internal(ScalarReg& input, int32_t scalar, std::vector<RvvInst>& code_container);
    static VectorReg codegen_vmv_vv_internal(VectorReg& input, std::vector<RvvInst>& code_container);
    static VectorReg codegen_vmv_vi_internal(int32_t scalar, std::vector<RvvInst>& code_container);
    
    OpCodegen(const std::vector<mlir::Operation*>& ops, const std::vector<FuseOpInfo>& fuse_op_infos, CodegenOpType codegen_type);
    void codegen();
    std::vector<RvvInst>& getPrelogueCode() { return prelogue_code_; }
    std::vector<RvvInst>& getLoopCode() { return loop_code_; }

private:
    void codegen_eletwise_op(std::vector<CodegenOpOperand>& inputs, CodegenOpOperand& output);
    void codegen_quant_lut_op(std::vector<CodegenOpOperand>& inputs, CodegenOpOperand& output);
    void codegen_max(std::vector<CodegenOpOperand>& inputs, CodegenOpOperand& output);
    void codegen_min(std::vector<CodegenOpOperand>& inputs, CodegenOpOperand& output);
    void codegen_abs(std::vector<CodegenOpOperand>& inputs, CodegenOpOperand& output);
    void codegen_extend(std::vector<CodegenOpOperand>& inputs, CodegenOpOperand& output);
    void codegen_quant(std::vector<CodegenOpOperand>& inputs, CodegenOpOperand& output);
    void codegen_quant_extend(std::vector<CodegenOpOperand>& inputs, CodegenOpOperand& output);

private:
    CodegenOpType codegen_type_;
    std::vector<mlir::Operation*> ops_;
    std::vector<FuseOpInfo> fuse_op_infos_;
    std::vector<RvvInst> prelogue_code_;
    std::vector<RvvInst> loop_code_;
};

} // namespace xp_tmu
} // namespace xp_mlir

#endif // OP_CODEGEN__HPP