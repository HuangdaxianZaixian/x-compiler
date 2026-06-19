#ifndef STANDALONE_CODEGEN_HPP
#define STANDALONE_CODEGEN_HPP

#include "standalone/OpCodegen.hpp"
#include "standalone/XVectorConverter.hpp"
#include "mlir/IR/Operation.h"
#include "standalone/RvvInstGen.hpp"

namespace xp_mlir {
namespace xp_std {

class StandaloneCodegen {
public:
    StandaloneCodegen(mlir::Operation* standalone_region_op);
    void codegen();
    std::string toAssembly() const;
    std::vector<uint32_t> toBinary() const;

private:
    void genOpsCode();
    CodeLoc getOpLocInScfRegion(mlir::Operation* op);
    void checkXvecOpSequence(const std::vector<mlir::Operation*> xvec_ops);
    void packExclusiveOneBundleInst(const RvvInst& inst, std::vector<RvvInst>& inst_bundle, VeuInstSlotSupportion& cur_slot);
    void dispatchInstBundle(std::vector<RvvInst>& inst_bundle);
    void dispatchInst(const RvvInst& inst, bool is_last = false);
    void scheduleInsts();
    void reallocRegisters();
    void setRvvInstLoc();

    void genMatmulTcLdCode();

private:
    std::shared_ptr<xp_std::StandalonePattern> standalone_pattern_;
    std::unique_ptr<XVectorConverter> xvec_converter_;
    std::vector<RvvInst> prelogue_code_;
    std::vector<RvvInst> loop_code_;
    std::vector<std::vector<RvvInst>> inst_sequence_;
    bool is_matmultc_ld_mode_;
};

} // namespace xp_std
} // namespace xp_mlir

#endif // STANDALONE_CODEGEN_HPP