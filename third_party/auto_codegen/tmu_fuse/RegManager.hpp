#ifndef REG_MANAGER__HPP
#define REG_MANAGER__HPP

#include <cinttypes>
#include <cstdint>
#include <vector>

namespace xp_mlir {
namespace xp_tmu {
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

    ScalarReg getScalarReg0();
    ScalarReg allocSpecificScalarReg(int32_t reg_id);
    ScalarReg allocScalarReg();
    VectorReg allocSpecificVectorReg(int32_t reg_id);
    VectorReg allocVectorReg();
    void reset();

private:
    RegManager();

private:
    std::vector<ScalarReg> idel_scalar_regs_;
    std::vector<ScalarReg> used_scalar_regs_;
    std::vector<VectorReg> idel_vector_regs_;
    std::vector<VectorReg> used_vector_regs_;
};

} // namespace xp_tmu
} // namespace xp_mlir

#endif // REG_MANAGER__HPP