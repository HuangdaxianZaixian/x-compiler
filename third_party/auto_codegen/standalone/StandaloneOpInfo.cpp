#include "standalone/StandaloneOpInfo.hpp"
#include "Helper/XPMLIRUtils.h"
#include <string>

namespace xp_mlir {
namespace xp_std {

std::string OpOperandInfo::to_string() const {
    if (is_none) {
        return "None";
    }
    std::string mem_desc;
    switch (mem_type) {
        case OpOperandMemType::V_REG:
            mem_desc = "V-REG = " + std::to_string(mem_value.v_reg_id);
            break;
        case OpOperandMemType::X_REG:
            mem_desc = "X-REG-SCALAR = " + std::to_string(mem_value.x_reg_scalar);
            break;
        case OpOperandMemType::L1:
            mem_desc = "L1 = " + std::to_string(mem_value.l1_addr);
            break;
        case OpOperandMemType::IMM:
             mem_desc = "IMM = " + std::to_string(mem_value.imm);
             break;
        default:
            mem_desc = "UNKNOWN";
    }
    
    return mem_desc + ", bit_width = " + std::to_string(bit_width) + ", shape = " + vector2Str(shape);
}

std::string StandaloneOpInfo::to_string() const {
    std::string result = "";
    for (size_t i = 0; i < input_infos.size(); ++i) {
        result += "Input " + std::to_string(i) + ": " + input_infos[i].to_string() + "\n";
    }
    result += "Output: " + output_info.to_string() + "\n";
    return result;
}

} // namespace xp_std
} // namespace xp_mlir