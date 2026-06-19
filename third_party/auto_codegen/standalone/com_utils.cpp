#include "com_utils.hpp"
#include "xp_mlir/Dialect/XMA/IR/XMAOps.h"
#include "Helper/TensorUtils.h"
#include "llvm/Support/Casting.h"
#include <cstdint>
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace xp_mlir {
namespace xp_std {
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

void copyRegionFromOldOp(Operation* op, Operation* new_op) {
    if (op->getNumRegions() == 0)
        return;
    mlir::IRMapping mapper;
    op->getRegion(0).cloneInto(&new_op->getRegion(0), mapper);
}

Operation* cloneOpAndAppendNewRegion(Operation *op) {
    auto ctx = op->getContext();
    OpBuilder builder(op->getContext());
    builder.setInsertionPointAfter(op);
    int region_num = op->getNumRegions();
    OperationState state(op->getLoc(), op->getName());
    state.addTypes(op->getResultTypes());
    state.addOperands(op->getOperands());
    state.addAttributes(op->getAttrs());
    state.addSuccessors(op->getSuccessors());
    for (unsigned i = 0; i < op->getNumRegions() + 1; ++i)
        state.addRegion();
    auto* new_op = Operation::create(state);
    new_op->setLoc(getNameLoc(ctx, getOpName(op)));
    builder.insert(new_op);
    if (region_num) {
        copyRegionFromOldOp(op, new_op);
    }
    mlir::Block *new_block = new mlir::Block();
    new_op->getRegion(region_num).push_back(new_block);
    
    return new_op;
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
    if (llvm::isa<xma::DeQuantOp, xma::QuantOp>(op)) {
        return {1, 2};
    }

    if (llvm::isa<xma::AddOp, xma::SubOp, xma::MulOp, xma::DivOp, xma::ReciprocalOp>(op)) {
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

    if (llvm::isa<xma::MatrixMaxPoolingOp>(op)) {
        return {1};
    }

    if (llvm::isa<xma::CallOp>(op)) {
        return {};
    }

    if(llvm::isa<xma::MatMulTCLDOp>(op)) {
        return {2}; // 假输入, 实际输入buffer是callOp的第2个operand
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

    if (llvm::isa<xma::AbsOp>(op)) {
        return 3;
    }

    if (llvm::isa<xma::ExtI32Op>(op)) {
        return 3;
    }

    if (llvm::isa<xma::ReciprocalOp>(op)) {
        return 3;
    }

    if (llvm::isa<xma::MatrixMaxPoolingOp>(op)) {
        return 5;
    }

    if (llvm::isa<xma::CallOp>(op)) {
        return op->getNumOperands() - 1;
    }

    if(llvm::isa<xma::MatMulTCLDOp>(op)) {
        return 3; // 假输出
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

std::vector<int64_t> getXmaBufferShape(mlir::Value value) {
    assert(value && !isNone(value) && "The value is expected to be non-null and not None");
    auto def_op = value.getDefiningOp();
    assert(def_op && "The output value should have a defining operation");
    auto shape_attr = def_op->getAttr("shape");
    assert(shape_attr && "shape attribute is expected in the defining op of the input value");
    auto shape_array_attr = llvm::dyn_cast<mlir::ArrayAttr>(shape_attr);
    std::vector<int64_t> shape;
    for (auto dim : shape_array_attr) {
        shape.push_back(llvm::dyn_cast<mlir::IntegerAttr>(dim).getInt());
    }

    return shape;
}

int32_t getXmaBufferAlignValue(int32_t bitWidth) {
    assert(bitWidth <= 32 && "the bit width of the data type is expected to be no more than 32");
    return (bitWidth == 8 ? 128 : 64);
}

int64_t align_up(int64_t value, int64_t align_value) {
    return ((value + align_value - 1) / align_value) * align_value;
}

bool is_i16(int32_t bit_width) {
    return bit_width >= 10 && bit_width <= 16;
};

bool is16bitImm(uint32_t value) {
    uint32_t mask = 0xFFFF; // 16-bit mask
    return value == uint32_t(value & mask);
}

bool is11bitImm(uint32_t value) {
    uint32_t mask = 0x7FF; // 11-bit mask
    return value == uint32_t(value & mask);
}

bool is5bitImm(uint32_t value) {
    uint32_t mask = 0x1F; // 5-bit mask
    return value == uint32_t(value & mask);
}

} // namespace xp_std
} // namespace xp_mlir