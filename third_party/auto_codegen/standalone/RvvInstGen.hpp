#ifndef STANDALONE_RVV_INST_GEN_FUNCS_HPP
#define STANDALONE_RVV_INST_GEN_FUNCS_HPP

#include <cassert>
#include <vector>
#include <string>
#include <functional>
#include "standalone/RegManager.hpp"

namespace xp_mlir {
namespace xp_std {
constexpr int32_t VEU_VECTOR_LANE_NUM = 128;

enum class RvvOperandType {
    NONE = 0, // 只用于输出, 表示没有输出
    IMM = 1,
    XREG,
    VREG,
    VM
};

enum class VeuInstSlotSupportion {
    UNKNOWN = 0,
    SLOT_0,
    SLOT_1,
    SLOT_0_1
};

union OperandWrapper {
    ScalarReg x_reg;
    VectorReg v_reg;
    uint32_t imm;
};

struct CodegenOpOperand {
    RvvOperandType type;
    OperandWrapper value;
    int32_t bit_width;

    bool operator==(const CodegenOpOperand& other) const;
};

struct RvvInstLoc {
    RvvInstLoc() : line_id(-1), slot_id(-1), scope_id(-1) {}

    int32_t line_id;
    int32_t slot_id;
    int32_t scope_id;  // prelogue: 0, outer loop: 1, inner loop: 2
};

struct RvvInst {
    static bool isExclusiveOneBundleInst(const RvvInst& inst);
    static bool isInstADependentOnInstB(const RvvInst& a, const RvvInst& b);
    static bool isTwoInstsHasDependency(const RvvInst& a, const RvvInst& b);
    static bool isNopRvvInst(const RvvInst& inst);
    static bool isOuterLoopBegin(const RvvInst& inst);
    static bool isOuterLoopEnd(const RvvInst& inst);
    static bool isInnerLoopBegin(const RvvInst& inst);
    static bool isInnerLoopEnd(const RvvInst& inst);

    void updateCode();

    std::string name;
    uint32_t raw;
    std::string assembly;
    VeuInstSlotSupportion slot_supportion;
    std::vector<CodegenOpOperand> inputs; 
    CodegenOpOperand output;
    RvvInstLoc loc;
};

struct RvvInstGenerator {
    static RvvInst genNopRvvInst();

    RvvInstGenerator(const std::string& inst_name) : name(inst_name) {}
    std::string name;
    uint32_t toCode(const std::vector<uint32_t>& args) const;
    std::string toAssembly(const std::vector<uint32_t>& args) const;
    RvvInst codegen(const std::vector<CodegenOpOperand>& inputs, const CodegenOpOperand& output);
};

} // namespace xp_std
} // namespace xp_mlir

#endif // STANDALONE_RVV_INST_GEN_FUNCS_HPP