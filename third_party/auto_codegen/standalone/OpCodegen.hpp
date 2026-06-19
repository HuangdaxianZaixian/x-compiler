#ifndef STANDALONE_OP_CODEGEN__HPP
#define STANDALONE_OP_CODEGEN__HPP

#include "mlir/IR/Operation.h"
#include "standalone/StandaloneOpInfo.hpp"
#include "standalone/RegManager.hpp"
#include "standalone/RvvInstGen.hpp"
#include "xp_mlir/Dialect/XMA/Vector/XVectorOps.hpp"

namespace xp_mlir {
namespace xp_std {

enum class CodeLoc {
    PRELOGUE,
    OUTER_LOOP,
    INNER_LOOP
};

class OpCodegen {
public:
    static std::vector<RvvInst> codegen_matmulTcLd(const StandaloneOpInfo& op_info);

    OpCodegen(mlir::Operation* xvec_op, CodeLoc code_loc, 
              std::vector<RvvInst>& prelogue_code, std::vector<RvvInst>& loop_code) 
        : xvec_op_(xvec_op), code_loc_(code_loc), 
          prelogue_code_(prelogue_code), loop_code_(loop_code) {}

    void codegen();
    std::vector<RvvInst>& getPrelogueCode() { return prelogue_code_; }
    std::vector<RvvInst>& getLoopCode() { return loop_code_; }

private:
    void codegen_immOp(mlir::Operation* op);
    void codegen_initXregOp(mlir::Operation* op);
    void codegen_scalarLoadOp(mlir::Operation* op);
    void codegen_vectorLoadOp(mlir::Operation* op);
    void codegen_storeOp(mlir::Operation* op);
    void codegen_reciprocalOp(mlir::Operation* op);
    void codegen_xaddOp(mlir::Operation* op);
    void codegen_xmulOp(mlir::Operation* op);
    void codegen_xdivOp(mlir::Operation* op);
    void codegen_xmodOp(mlir::Operation* op);
    void codegen_xincrementOp(mlir::Operation* op);
    void codegen_selectScalarFromVectorAndBroadcastOp(mlir::Operation* op);
    void codegen_extendOp(mlir::Operation* op);
    void codegen_vaddOp(mlir::Operation* op);
    void codegen_vsubOp(mlir::Operation* op);
    void codegen_vmulOp(mlir::Operation* op);
    void codegen_vdivOp(mlir::Operation* op);
    void codegen_vmaxOp(mlir::Operation* op);
    void codegen_vminOp(mlir::Operation* op);
    void codegen_vabsOp(mlir::Operation* op);
    void codegen_vquantOp(mlir::Operation* op);
    void codegen_vqlutOp(mlir::Operation* op);
    void codegen_loopBeginOp(mlir::Operation* op);
    void codegen_loopEndOp(mlir::Operation* op);
    void codegen_rvvConfigOp(mlir::Operation* op);
    
    std::vector<RvvInst>& getCodeContainer();
    CodegenOpOperand xvectorOperand2RvvOperand(mlir::Value val);
    std::vector<CodegenOpOperand> xvecOpInputs2RvvInstInputs(mlir::Operation* op);
    
private:
    mlir::Operation* xvec_op_;
    CodeLoc code_loc_;
    std::vector<RvvInst>& prelogue_code_;
    std::vector<RvvInst>& loop_code_;
};

} // namespace xp_std
} // namespace xp_mlir

#endif // STANDALONE_OP_CODEGEN__HPP