#ifndef TMU_FUSE_PATTERN 
#define TMU_FUSE_PATTERN

#include "mlir/IR/Operation.h"
#include "tmu_fuse/com_utils.hpp"
#include "tmu_fuse/FuseOpInfo.hpp"
#include "tmu_fuse/RegManager.hpp"

namespace xp_mlir {
namespace xp_tmu {

struct HwSlotOpPartition {
    HwSlotOpPartition() : tmu_op_(nullptr), dequant_op_(nullptr), quant_op_(nullptr) {}

    mlir::Operation* tmu_op_;
    mlir::Operation* dequant_op_;
    std::vector<mlir::Operation*> veu_core_ops_;
    mlir::Operation* quant_op_;
    bool is_dequant_input_from_tmu_online_;
};

struct QuantLutPartion {
    QuantLutPartion() : quant_op_(nullptr), lut_op_(nullptr) {}

    mlir::Operation* quant_op_;
    mlir::Operation* lut_op_;
};

struct MatmulTcPartion {
    MatmulTcPartion() : not_last_dim_quant_op_(nullptr), extend_op_(nullptr) {}

    mlir::Operation* not_last_dim_quant_op_;
    mlir::Operation* extend_op_;
};

struct VeuCoreOpPartition {
    VeuCoreOpPartition() : input_from_l1_elem_wise_op_(nullptr) {}
    
    mlir::Operation* input_from_l1_elem_wise_op_;
    QuantLutPartion quant_lut_partion_;
    MatmulTcPartion matmultc_partition_; // for matmultc
    std::vector<mlir::Operation*> other_veu_core_ops_;
};

class TmuFusePattern {
public:
    static bool isTmuFuse(mlir::Operation* op);
    static bool isNoRegionTmu(mlir::Operation *op);

    TmuFusePattern(mlir::Operation* fuse_region_op);
    void parseTmuFuse();
    HwSlotOpPartition& getHwSlotOpPartition() { return hw_slot_op_partition_; }
    VeuCoreOpPartition& getVeuCoreOpPartition() { return veu_core_op_partition_; }
    std::map<mlir::Operation*, FuseOpInfo>& getOp2FuseInfoTable() { return op_2_fuse_info_table_; }
    bool isEmptyVecCoreMode() const;
    bool isSwishMode() const;

private:
    void getAllOpsInFuseRegion();
    void genValue2DefTable();
    void genOp2InputDefsTable();
    void genOp2OutputUsesTable();
    mlir::Operation* isValueGeneratedInTmuFuse(mlir::Value value);
    void parseHwSlotOps();
    void parseVeuCoreOps();
    void genFuseOpInfo();
    void setOpUsersInputReg(mlir::Operation* op, int32_t reg_id);
    void allocVirtualVectorReg();
    void allocVectorReg();
    void checkOpInfo() const;
    void printFuseInfo() const;
    
private:
    mlir::Operation* fuse_region_op_;
    mlir::Value fuse_region_op_origin_output_value_;
    std::vector<mlir::Operation*> all_ops_in_fuse_region_;
    HwSlotOpPartition hw_slot_op_partition_;
    VeuCoreOpPartition veu_core_op_partition_;
    std::map<mlir::Value, mlir::Operation*, mlir_value_compare> value_2_def_table_;
    std::map<mlir::Operation*, std::vector<mlir::Operation*>> op_2_input_defs_table_;
    std::map<mlir::Operation*, std::vector<mlir::Operation*>> op_2_output_uses_table_;
    std::map<mlir::Operation*, FuseOpInfo> op_2_fuse_info_table_;
};

} // namespace xp_tmu
} // namespace xp_mlir


#endif // TMU_FUSE_PATTERN