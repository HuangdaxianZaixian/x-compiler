#ifndef TMU_FUSE_COM_UTILS_HPP
#define TMU_FUSE_COM_UTILS_HPP

#include <algorithm>
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "xp_mlir/Dialect/XPCommon.h"

namespace xp_mlir {
namespace xp_tmu {

template<typename T, template <typename> class Container>
bool is_in_container(const Container<T>& vec, const T& value) {
    return vec.end() != std::find(vec.begin(), vec.end(), value);
}

struct mlir_value_compare {
  bool operator()(mlir::Value a, mlir::Value b) const {
    if (a.getImpl() < b.getImpl()) {
      return true;
    }
    return false;
  }
};

int32_t getScalarAttr(mlir::Operation *op, const std::string &attrName);
int32_t getXDataTypeBitWidth(xp_mlir::XDataType dtype);
xp_mlir::XDataType getValueDataType(mlir::Value val);
std::vector<int64_t> getInputOperandIndices(mlir::Operation* op);
int64_t getOutputOperandIndex(mlir::Operation* op);
std::vector<mlir::Value> getOpInputValues(mlir::Operation* op);
mlir::Value getOpOutputValue(mlir::Operation* op);
bool isLastDimQuant(mlir::Operation *op);
bool isPerTensorQuant(mlir::Operation* op);
int32_t getTmuOnlineOutputBitwidth(mlir::Operation* fuse_region_op);

} // namespace xp_tmu
} // namespace xp_mlir

#endif // TMU_FUSE_COM_UTILS_HPP