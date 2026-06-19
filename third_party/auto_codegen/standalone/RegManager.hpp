#ifndef STANDALONE_REG_MANAGER__HPP
#define STANDALONE_REG_MANAGER__HPP

#include <cinttypes>
#include <cstdint>
#include <vector>
#include <set>


namespace xp_mlir {
namespace xp_std {
constexpr int32_t VEU_RESERVED_ZERO_VALUE_XREG_ID = 0; 
constexpr int32_t VEU_RVV_CONFIG_VL_SET_XREG_ID = 15;
constexpr int32_t VEU_RVV_CONFIG_VL_RETURN_XREG_ID = 30;
constexpr int32_t VEU_OUTER_LOOP_TIMES_XREG_ID = 31;
constexpr int32_t VEU_INNER_LOOP_TIMES_XREG_ID = 29;

constexpr int32_t VEU_RESERVED_MASK_VREG_ID = 0;
constexpr int32_t VEU_RESERVED_MATMULTC_VREG_ID = 15;

const std::set<int32_t> VEU_SPECIFIC_SCALAR_REG_IDS = {VEU_RESERVED_ZERO_VALUE_XREG_ID, VEU_RVV_CONFIG_VL_SET_XREG_ID, VEU_RVV_CONFIG_VL_RETURN_XREG_ID,
                                                       VEU_OUTER_LOOP_TIMES_XREG_ID, VEU_INNER_LOOP_TIMES_XREG_ID};
const std::set<int32_t> VEU_SPECIFIC_VECTOR_REG_IDS = {VEU_RESERVED_MASK_VREG_ID, VEU_RESERVED_MATMULTC_VREG_ID};

constexpr int32_t VEU_MAX_LOGICAL_SCALAR_REG_NUM = 32;
constexpr int32_t VEU_MAX_PHYSICAL_SCALAR_REG_NUM = 32;

constexpr int32_t VEU_MAX_LOGICAL_VECTOR_REG_NUM = 16;
constexpr int32_t VEU_MAX_PHYSICAL_VECTOR_REG_NUM = 16;

enum class RegStatus {
    UNKNOWN = 0,
    IDLE = 1,
    USED = 2,
};

struct ScalarReg {
    int32_t id;
    RegStatus status;
};

struct VectorReg {
    int32_t id;
    RegStatus status;
};

class RegManager {
public:
    static RegManager& get() {
        static RegManager instance;
        return instance;
    }

    ScalarReg allocSpecificScalarReg(int32_t reg_id);
    ScalarReg allocScalarReg();
    VectorReg allocSpecificVectorReg(int32_t reg_id);
    VectorReg allocVectorReg();
    void reset();
    bool isUsedScalarReg(int32_t reg_id);
    bool isUsedVectorReg(int32_t reg_id);

    bool isPhysicalScalarReg(int32_t reg_id);
    bool isPhysicalVectorReg(int32_t reg_id);
    bool isLogicalScalarReg(int32_t reg_id);
    bool isLogicalVectorReg(int32_t reg_id);
    bool isSpecificScalarReg(int32_t reg_id);
    bool isSpecificVectorReg(int32_t reg_id);
    bool isVirtualScalarReg(int32_t reg_id);

private:
    RegManager();

private:
    std::vector<ScalarReg> idel_scalar_regs_;
    std::vector<ScalarReg> used_scalar_regs_;
    std::vector<VectorReg> idel_vector_regs_;
    std::vector<VectorReg> used_vector_regs_;
};

} // namespace xp_std
} // namespace xp_mlir

#endif // STANDALONE_REG_MANAGER__HPP