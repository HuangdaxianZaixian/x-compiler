#include "tmu_fuse/RegManager.hpp"
#include <cassert>

namespace xp_mlir {
namespace xp_tmu {
RegManager::RegManager() {
    reset();
}

ScalarReg RegManager::getScalarReg0() {
    // x0寄存器固定为0, 内部保留, 不做分配
    return ScalarReg{0, RegStatus::USED};
}

void RegManager::reset() {
    idel_scalar_regs_.clear();
    used_scalar_regs_.clear();
    idel_vector_regs_.clear();
    used_vector_regs_.clear();

    for (int32_t i = 0; i < 32; ++i) {
        // x0内部保留
        // x30, x15固定给vsetvli使用
        if (i == 0 || i == 15 || i == 30) {
            continue;
        }
        idel_scalar_regs_.push_back(ScalarReg{i, RegStatus::IDLE});
    }
    
    // 虚拟寄存器内部有固定用法, 不做分配
    // 优先从v14开始分配, 因为v0有特殊用途(mask vector register), v15也有特殊用途(matmultc)
    idel_vector_regs_.push_back(VectorReg{0, RegStatus::IDLE});
    idel_vector_regs_.push_back(VectorReg{15, RegStatus::IDLE});
    for (int32_t i = 1; i < 15; ++i) {
        idel_vector_regs_.push_back(VectorReg{i, RegStatus::IDLE});
    }
}

ScalarReg RegManager::allocSpecificScalarReg(int32_t reg_id) {
    for (auto it = idel_scalar_regs_.begin(); it != idel_scalar_regs_.end(); ++it) {
        if (it->id == reg_id) {
            ScalarReg allocated_reg = *it;
            allocated_reg.status = RegStatus::USED;
            used_scalar_regs_.push_back(allocated_reg);
            idel_scalar_regs_.erase(it);
            return allocated_reg;
        }
    }
    
    assert(false && "Requested scalar register is not available");
}

ScalarReg RegManager::allocScalarReg() {
    if (idel_scalar_regs_.empty()) {
        assert(false && "No idle scalar registers available");
    }
    ScalarReg allocated_reg = idel_scalar_regs_.back();
    allocated_reg.status = RegStatus::USED;
    used_scalar_regs_.push_back(allocated_reg);
    idel_scalar_regs_.pop_back();
    return allocated_reg;
}

VectorReg RegManager::allocSpecificVectorReg(int32_t reg_id) {
    for (auto it = idel_vector_regs_.begin(); it != idel_vector_regs_.end(); ++it) {
        if (it->id == reg_id) {
            VectorReg allocated_reg = *it;
            allocated_reg.status = RegStatus::USED;
            used_vector_regs_.push_back(allocated_reg);
            idel_vector_regs_.erase(it);
            return allocated_reg;
        }
    }
    
    assert(false && "Requested vector register is not available");
}

VectorReg RegManager::allocVectorReg() {
    if (idel_vector_regs_.empty()) {
        assert(false && "No idle vector registers available");
    }
    VectorReg allocated_reg = idel_vector_regs_.back();
    allocated_reg.status = RegStatus::USED;
    used_vector_regs_.push_back(allocated_reg);
    idel_vector_regs_.pop_back();
    return allocated_reg;
}

} // namespace xp_tmu
} // namespace xp_mlir