#ifndef FUSE_OP_INFO__HPP
#define FUSE_OP_INFO__HPP

#include <vector>
#include <cinttypes>
#include <string>
#include "tmu_fuse/RvvInstGen.hpp"

namespace xp_mlir {
namespace xp_tmu {
enum class OpOperandMemType {
    UNKONWN = 0,
    V_REG = 1,
    X_REG,
    IMM,
    L1,
};

union OpOperandMemValue {
    int32_t v_reg_id;
    uint32_t x_reg_scalar;
    uint32_t imm;
    int32_t l1_addr;
};

struct OpOperandInfo {
    OpOperandInfo() : is_none(false), mem_type(OpOperandMemType::UNKONWN), mem_value({-1}), bit_width(0), elem_num(-1) {}
    bool is_none;
    OpOperandMemType mem_type;
    OpOperandMemValue mem_value; 
    int32_t bit_width;
    int32_t elem_num; // 只用于lut table的描述

    std::string to_string() const;
};

struct FuseOpInfo {
    std::vector<OpOperandInfo> input_infos;
    OpOperandInfo output_info;

    std::string to_string() const;
};

} // namespace xp_tmu
} // namespace xp_mlir

#endif // FUSE_OP_INFO__HPP