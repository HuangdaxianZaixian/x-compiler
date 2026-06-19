#ifndef STANDALONE_COM_UTILS_HPP
#define STANDALONE_COM_UTILS_HPP

#include <algorithm>
#include <cstdint>
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "xp_mlir/Dialect/XPCommon.h"

namespace xp_mlir {
namespace xp_std {

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
std::vector<int64_t> getXmaBufferShape(mlir::Value value);
int32_t getXmaBufferAlignValue(int32_t bitWidth);
mlir::Operation* cloneOpAndAppendNewRegion(mlir::Operation *op);
int64_t align_up(int64_t value, int64_t align_value);
bool is_i16(int32_t bit_width);
bool is16bitImm(uint32_t value);
bool is11bitImm(uint32_t value);
bool is5bitImm(uint32_t value);

} // namespace xp_std
} // namespace xp_mlir

#endif // STANDALONE_COM_UTILS_HPP