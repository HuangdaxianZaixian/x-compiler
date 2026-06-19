#ifndef TMU_FUSE_CODEGEN_HPP
#define TMU_FUSE_CODEGEN_HPP

#include "tmu_fuse/OpCodegen.hpp"
#include "tmu_fuse/TmuFusePattern.hpp"
#include "mlir/IR/Operation.h"
#include "tmu_fuse/RvvInstGen.hpp"

namespace xp_mlir {
namespace xp_tmu {

class TmuFuseCodegen {
public:
    TmuFuseCodegen(mlir::Operation* fuse_region_op, uint32_t loop_time);
    void codegen();
    std::string toAssembly() const;
    std::vector<uint32_t> toBinary() const;

private:
    void genVeuOpsCode();
    void genVsetvliCode();
    void genLoopConfigCode();
    void determineVectorInstSlot();
    void packExclusiveOneBundleInst(const RvvInst& inst, std::vector<RvvInst>& inst_bundle, VeuInstSlotSupportion& cur_slot);
    void dispatchInstBundle(std::vector<RvvInst>& inst_bundle);
    void dispatchInst(const RvvInst& inst, bool is_last = false);
    void scheduleInsts();

private:
    mlir::Operation* fuse_region_op_;
    uint32_t loop_time_;
    std::vector<RvvInst> prelogue_code_;
    std::vector<RvvInst> loop_code_;
    std::vector<std::vector<RvvInst>> inst_sequence_;
};

} // namespace xp_tmu
} // namespace xp_mlir

#endif // TMU_FUSE_CODEGEN_HPP