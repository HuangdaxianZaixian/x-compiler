#include "tmu_fuse/com_utils.hpp"
#include "xp_mlir/Dialect/XMA/IR/XMAOps.h"
#include "Helper/TensorUtils.h"
#include "llvm/Support/Casting.h"
#include <cstdint>

using namespace mlir;

namespace xp_mlir {
namespace xp_tmu {
int32_t getScalarAttr(Operation *op, const std::string &attrName) {
  auto attr = op->getAttr(attrName);
  auto intAttr = llvm::dyn_cast_if_present<mlir::IntegerAttr>(attr);
  assert(intAttr && "The attribute is expected to be an IntegerAttr");
  return intAttr.getInt();
}

int32_t getXDataTypeBitWidth(xp_mlir::XDataType dtype) {
    switch (dtype) {
        case xp_mlir::XDataType::U8:
        case xp_mlir::XDataType::I8:
            return 8;
        case xp_mlir::XDataType::I10:
            return 10;
        case xp_mlir::XDataType::I11:
            return 11;
        case xp_mlir::XDataType::I12:
            return 12;
        case xp_mlir::XDataType::I13:
            return 13;
        case xp_mlir::XDataType::I14:
            return 14;
        case xp_mlir::XDataType::I16:
            return 16;
        case xp_mlir::XDataType::U32:
        case xp_mlir::XDataType::I32:
            return 32;
        case xp_mlir::XDataType::I64:
            return 64;
        default:
            assert(false && "Unsupported XDataType for bit width retrieval");
            return -1;
    }
}

xp_mlir::XDataType getValueDataType(mlir::Value val) {
    assert(val && !isNone(val) && "The value is expected to be non-null and not None");
    auto input_def_op = val.getDefiningOp();
    auto data_type_attr = input_def_op->getAttr("data_type");
    assert(data_type_attr && "data_type attribute is expected in the defining op of the input value");
    auto data_type_val = llvm::dyn_cast<mlir::IntegerAttr>(data_type_attr).getInt();
    xp_mlir::XDataType data_type = xp_mlir::symbolizeXDataType(data_type_val).value_or(xp_mlir::XDataType::None);
    assert(data_type != xp_mlir::XDataType::None && "unsupported data type for the input value");

    return data_type;
}

std::vector<int64_t> getInputOperandIndices(mlir::Operation* op) {
    if (llvm::isa<xma::MatMulOp>(op)) {
        return {1, 2, 3};
    }

    if (llvm::isa<xma::MatMulTCOp, xma::ConvOp, xma::WinogradOp, xma::DepthwiseOp, xma::FCOp, xma::ColWiseOp>(op)) {
        return {1, 2, 3};
    }

    if (llvm::isa<xma::DeQuantOp, xma::QuantOp>(op)) {
        return {1, 2};
    }

    if (llvm::isa<xma::AddOp, xma::SubOp, xma::MulOp, xma::DivOp>(op)) {
        return {1, 2};
    }

    if (llvm::isa<xma::MaxOp, xma::MinOp>(op)) {
        return {1, 2};
    }

    if (llvm::isa<xma::LutOp>(op)) {
        return {1, 2};
    }

    if (llvm::isa<xma::AbsOp>(op)) {
        return {1};
    }

    if (llvm::isa<xma::ExtI32Op>(op)) {
        return {1};
    }

    if (llvm::isa<xma::MatrixMaxPoolingOp, xma::AvgPoolingOp, xma::MaxPoolingOp, xma::GlobalAvgPoolingOp, xma::GlobalMaxPoolingOp>(op)) {
        return {1};
    }

    if (llvm::isa<xma::ResizeOp, xma::AffineOp>(op)) {
        return {1};
    }

    if (llvm::isa<xma::GridSampleOp>(op)) {
        return {1, 2, 3};
    }

    assert(false && "Unsupported operation type for input operand indices");
    return {};
}

int64_t getOutputOperandIndex(mlir::Operation* op) {
    if (llvm::isa<xma::DeQuantOp, xma::QuantOp>(op)) {
        return 3;
    }

    if (llvm::isa<xma::AddOp, xma::SubOp, xma::MulOp>(op)) {
        return 3;
    }

    if (llvm::isa<xma::LutOp, xma::MaxOp, xma::MinOp>(op)) {
        return 3;
    }

    if (llvm::isa<xma::MatMulOp>(op)) {
        return 8;
    }

    if (llvm::isa<xma::MatMulTCOp, xma::ConvOp, xma::WinogradOp, xma::DepthwiseOp, xma::FCOp, xma::ColWiseOp>(op)) {
        return 8;
    }

    if (llvm::isa<xma::AbsOp>(op)) {
        return 3;
    }

    if (llvm::isa<xma::ExtI32Op>(op)) {
        return 3;
    }

    if (llvm::isa<xma::MatrixMaxPoolingOp, xma::AvgPoolingOp, xma::MaxPoolingOp, xma::GlobalAvgPoolingOp, xma::GlobalMaxPoolingOp>(op)) {
        return 5;
    }

    if (llvm::isa<xma::ResizeOp, xma::AffineOp>(op)) {
        return 8;
    }

    if (llvm::isa<xma::GridSampleOp>(op)) {
        return 8;
    }

    assert(false && "Unsupported operation type for output operand index");
    return {};
}

std::vector<mlir::Value> getOpInputValues(mlir::Operation* op) {
    std::vector<mlir::Value> input_values;
    auto input_indices = getInputOperandIndices(op);
    for (auto index : input_indices) {
        input_values.push_back(op->getOperand(index));
    }
    return input_values;
}

mlir::Value getOpOutputValue(mlir::Operation* op) {
    assert(op && "op is nullptr");
    auto output_index = getOutputOperandIndex(op);
    auto res = op->getOperand(output_index);
    assert(!isNone(res) && "The output operand is None");
    return res;
}

bool isLastDimQuant(Operation *op) {
    assert((op && llvm::isa<xma::QuantOp, xma::DeQuantOp>(op)) && "The operation must be QuantOp or DeQuantOp");
    auto channel_dim_attr = op->getAttr("channel_dim");
    assert(channel_dim_attr && "QuantOp and DeQuantOp should have channel_dim attribute");
    auto output_value = getOpOutputValue(op);
    auto def_op = output_value.getDefiningOp();
    assert(def_op && "The output value should have a defining operation");
    auto shape_attr = def_op->getAttr("shape");
    auto quant_output_shape_size = llvm::dyn_cast<mlir::ArrayAttr>(shape_attr).size();
    auto channel_dim = llvm::dyn_cast<mlir::IntegerAttr>(channel_dim_attr).getInt();
    if (channel_dim < 0) {
        channel_dim += quant_output_shape_size;
    }
    return channel_dim == quant_output_shape_size - 1;
}

bool isPerTensorQuant(mlir::Operation* op) {
    assert(op && (llvm::isa<xma::QuantOp, xma::DeQuantOp>(op)) && "Error: The operation must be QuantOp or DeQuantOp.");
    // if (op->hasAttr("scalar_param_for_opt")) return true;
    if (isNone(op->getOperand(2))) {
        assert(op->hasAttr("scalar_param") && "The per-tensor quantization op should have scalar_param attribute");
        return true;
    }

    return false;
}

int32_t getTmuOnlineOutputBitwidth(mlir::Operation* fuse_region_op) {
    assert(fuse_region_op && "fuse_region_op is nullptr");

    if (llvm::isa<xma::MatMulOp, xma::MatMulTCOp, xma::FCOp,
                  xma::ConvOp, xma::WinogradOp, xma::DepthwiseOp,
                  xma::ColWiseOp>(fuse_region_op)) {
        return 32;
    }

    if (llvm::isa<xma::MatrixMaxPoolingOp, xma::AvgPoolingOp, 
                  xma::MaxPoolingOp, xma::GlobalAvgPoolingOp, 
                  xma::GlobalMaxPoolingOp, xma::ResizeOp, 
                  xma::AffineOp, xma::GridSampleOp>(fuse_region_op)) {
        auto input_value = getOpInputValues(fuse_region_op).at(0);
        auto data_type = getValueDataType(input_value);
        return getXDataTypeBitWidth(data_type);
    }

    assert(false && "unsupported fuse region op type for retrieving tmu online output bitwidth");
    return -1;
}

} // namespace xp_tmu
} // namespace xp_mlir