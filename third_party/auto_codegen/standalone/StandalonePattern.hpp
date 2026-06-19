#ifndef STANDALONE_PATTERN 
#define STANDALONE_PATTERN

#include "mlir/IR/Operation.h"
#include "com_utils.hpp"
#include "standalone/StandaloneOpInfo.hpp"
#include "standalone/RegManager.hpp"

namespace xp_mlir {
namespace xp_std {
constexpr int32_t MATMULTCLD_INPUT_INDEX_IN_CALL_OP = 1;

struct QuantLutPartition {
    QuantLutPartition() : quant_op_(nullptr), lut_op_(nullptr) {}

    mlir::Operation* quant_op_;
    mlir::Operation* lut_op_;
};    

class StandalonePattern {
public:
    static bool isStandalone(mlir::Operation *op);

public:
    StandalonePattern(mlir::Operation* standalone_region_op);
    void parseStandalone();
    mlir::Operation* getStandaloneRegionOp() { return standalone_region_op_; }
    const std::map<mlir::Operation*, StandaloneOpInfo>& getOp2StandaloneInfoTable() const { return op_2_standalone_info_table_; }
    const std::vector<mlir::Operation*>& getAllStandaloneRegionOps() const { return all_ops_in_standalone_region_; }
    const QuantLutPartition& getQuantLutPartition() const { return quant_lut_partition_; }
    bool isMatmulTcLd();

private:
    std::vector<int64_t> getExtendShape(const std::vector<int64_t>& shape);
    void getAllOpsInStandaloneRegion();
    void genValue2DefTable();
    void genOp2InputDefsTable();
    void genOp2OutputUsesTable();
    void genQuantLutPartition();
    mlir::Operation* isValueGeneratedInStandaloneRegion(mlir::Value value);
    void genStandaloneOpInfo();
    void setOpUsersInputReg(mlir::Operation* op, int32_t reg_id);
    void allocVectorReg();
    void checkOpInfo() const;
    void printStandaloneInfo() const;
    void restoreMatmulTcLdInput();
    
private:
    mlir::Operation* standalone_region_op_;
    std::vector<mlir::Operation*> all_ops_in_standalone_region_;
    QuantLutPartition quant_lut_partition_;
    mlir::Value origin_matmulTcLd_input_value_;
    std::map<mlir::Value, mlir::Operation*, mlir_value_compare> value_2_def_table_;
    std::map<mlir::Operation*, std::vector<mlir::Operation*>> op_2_input_defs_table_;
    std::map<mlir::Operation*, std::vector<mlir::Operation*>> op_2_output_uses_table_;
    std::map<mlir::Operation*, StandaloneOpInfo> op_2_standalone_info_table_;
};

} // namespace xp_std
} // namespace xp_mlir


#endif // STANDALONE_PATTERN