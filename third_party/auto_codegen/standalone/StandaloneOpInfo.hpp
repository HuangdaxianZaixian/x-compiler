#ifndef STANDALONE_OP_INFO__HPP
#define STANDALONE_OP_INFO__HPP

#include <cstdint>
#include <vector>
#include <cinttypes>
#include <string>
#include "standalone/RvvInstGen.hpp"

namespace xp_mlir {
namespace xp_std {
enum class OpOperandType {
    GENERAL = 0,
    LUT_TABLE,
    QUANT_PARAM
};

enum class OpOperandMemType {
    UNKONWN = 0,
    V_REG = 1,
    X_REG,
    IMM,
    L1,
};

union OpOperandMemValue {
    int32_t v_reg_id;
    int32_t x_reg_scalar;
    int32_t imm;
    int32_t l1_addr;
};

struct OpOperandInfo {
    OpOperandInfo() : is_none(false), operand_type(OpOperandType::GENERAL), mem_type(OpOperandMemType::UNKONWN), 
                      mem_value({-1}), bit_width(0), shape({}), quant_channel_dim(-1) {}
    bool is_none;
    OpOperandType operand_type;
    OpOperandMemType mem_type;
    OpOperandMemValue mem_value; 
    int32_t bit_width;
    std::vector<int64_t> shape;
    int32_t quant_channel_dim; // only for quant

    std::string to_string() const;
};

struct StandaloneOpInfo {
    std::vector<OpOperandInfo> input_infos;
    OpOperandInfo output_info;

    std::string to_string() const;
};

} // namespace xp_std
} // namespace xp_mlir

#endif // STANDALONE_OP_INFO__HPP