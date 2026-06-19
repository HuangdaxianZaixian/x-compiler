#ifndef RVV_INST_GEN_FUNCS_HPP
#define RVV_INST_GEN_FUNCS_HPP

#include <cassert>
#include <vector>
#include <string>
#include <functional>
#include "tmu_fuse/RegManager.hpp"

namespace xp_mlir {
namespace xp_tmu {
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

    bool operator==(const CodegenOpOperand& other) const;
};

struct RvvInst {
    static bool isExclusiveOneBundleInst(const RvvInst& inst);
    static bool isInstADependentOnInstB(const RvvInst& a, const RvvInst& b);
    static bool isTwoInstsHasDependency(const RvvInst& a, const RvvInst& b);
    static bool isNopInst(const RvvInst& inst);

    bool isUsingGeneralVectorRegister() const;
    bool isUsingVirtualVectorRegisterV24OrV25() const;
    bool isUsingVirtualVectorRegisterV23() const;

    uint32_t raw;
    std::string assembly;
    VeuInstSlotSupportion slot_supportion;
    std::vector<CodegenOpOperand> inputs; 
    CodegenOpOperand output;
};

struct RvvInstGenerator {
    RvvInstGenerator(const std::string& inst_name) : name(inst_name) {}
    std::string name;
    uint32_t toCode(const std::vector<uint32_t>& args) const;
    std::string toAssembly(const std::vector<uint32_t>& args) const;
    RvvInst codegen(const std::vector<CodegenOpOperand>& inputs, const CodegenOpOperand& output);
};

} // namespace xp_tmu
} // namespace xp_mlir

#endif // RVV_INST_GEN_FUNCS_HPP