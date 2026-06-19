#include "standalone/RegManager.hpp"
#include "standalone/com_utils.hpp"
#include <cassert>

namespace xp_mlir {
namespace xp_std {
RegManager::RegManager() {
    reset();
}

void RegManager::reset() {
    idel_scalar_regs_.clear();
    used_scalar_regs_.clear();
    idel_vector_regs_.clear();
    used_vector_regs_.clear();

    for (int32_t i = VEU_MAX_LOGICAL_SCALAR_REG_NUM - 1; i >= 0; --i) {
        if (VEU_SPECIFIC_SCALAR_REG_IDS.count(i) == 1) {
            continue;
        }
        idel_scalar_regs_.push_back(ScalarReg{i, RegStatus::IDLE});
    }
    
    for (int32_t i = VEU_MAX_LOGICAL_VECTOR_REG_NUM - 1; i >= 0; --i) {
        if (VEU_SPECIFIC_VECTOR_REG_IDS.count(i) == 1) {
            continue;
        }
        idel_vector_regs_.push_back(VectorReg{i, RegStatus::IDLE});
    }
}

bool RegManager::isUsedScalarReg(int32_t reg_id) {
    for (const auto& reg : used_scalar_regs_) {
        if (reg.id == reg_id) {
            return true;
        }
    }
    return false;
}

bool RegManager::isUsedVectorReg(int32_t reg_id) {
    for (const auto& reg : used_vector_regs_) {
        if (reg.id == reg_id) {
            return true;
        }
    }
    return false;
}

ScalarReg RegManager::allocSpecificScalarReg(int32_t reg_id) {
    assert(VEU_SPECIFIC_SCALAR_REG_IDS.count(reg_id) == 1 && "the specific scalar register id should be in the predefined set");
    ScalarReg allocated_reg;
    allocated_reg.id = reg_id;
    allocated_reg.status = RegStatus::USED;
    return allocated_reg;
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
    assert(VEU_SPECIFIC_VECTOR_REG_IDS.count(reg_id) == 1 && "the specific vector register id should be in the predefined set");
    VectorReg allocated_reg;
    allocated_reg.id = reg_id;
    allocated_reg.status = RegStatus::USED;
    return allocated_reg;
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

bool RegManager::isPhysicalScalarReg(int32_t reg_id) {
    return reg_id >= 0 && reg_id < VEU_MAX_PHYSICAL_SCALAR_REG_NUM;
}

bool RegManager::isPhysicalVectorReg(int32_t reg_id) {
    return reg_id >= 0 && reg_id < VEU_MAX_PHYSICAL_VECTOR_REG_NUM;
}

bool RegManager::isLogicalScalarReg(int32_t reg_id) {
    return reg_id >= VEU_MAX_PHYSICAL_SCALAR_REG_NUM && reg_id < VEU_MAX_LOGICAL_SCALAR_REG_NUM;
}

bool RegManager::isLogicalVectorReg(int32_t reg_id) {
    return reg_id >= VEU_MAX_PHYSICAL_VECTOR_REG_NUM && reg_id < VEU_MAX_LOGICAL_VECTOR_REG_NUM;
}

bool RegManager::isSpecificScalarReg(int32_t reg_id) {
    return is_in_container(VEU_SPECIFIC_SCALAR_REG_IDS, reg_id);
}

bool RegManager::isSpecificVectorReg(int32_t reg_id) {
    return is_in_container(VEU_SPECIFIC_VECTOR_REG_IDS, reg_id);
}


} // namespace xp_std
} // namespace xp_mlir