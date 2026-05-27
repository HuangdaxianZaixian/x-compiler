#include "dialect/top/IR/TopOps.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include "mlir/IR/BuiltinTypes.h"
#include "utils/com_utils.hpp"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace xc {
namespace top {

::mlir::SmallVector<::mlir::utils::IteratorType> AddOp::getLoopIteratorTypes() {
    auto output = getResult();
    auto output_shape = utils::getTensorShape(output);
    return llvm::SmallVector<::mlir::utils::IteratorType>(output_shape.size(), ::mlir::utils::IteratorType::parallel);
}

::mlir::SmallVector<::mlir::Range> AddOp::getIterationDomain(mlir::OpBuilder& builder) {
    auto output = getResult();
    auto output_shape = utils::getTensorShape(output);
    llvm::SmallVector<mlir::Range> iteration_domain;

    for (uint32_t i = 0; i < output_shape.size(); ++i) {
        auto dim_size = output_shape[i];
        // offset, size, stride
        iteration_domain.push_back({builder.getIndexAttr(0), builder.getIndexAttr(dim_size), builder.getIndexAttr(1)});
    }

    return iteration_domain;
}

::mlir::FailureOr<::mlir::TilingResult> AddOp::getTiledImplementation(mlir::OpBuilder& builder, llvm::ArrayRef<mlir::OpFoldResult> offsets, llvm::ArrayRef<mlir::OpFoldResult> sizes) {
    auto loc = getLoc();
    auto operands = getOperands();

    llvm::SmallVector<mlir::Operation*> slice_ops;
    llvm::SmallVector<mlir::Value> sliced_operands;
    for (uint32_t i = 0; i < operands.size(); ++i) {
        auto operand = operands[i];
        auto strides = llvm::SmallVector<mlir::OpFoldResult>(sizes.size(), builder.getIndexAttr(1));
        auto slice = builder.create<mlir::tensor::ExtractSliceOp>(loc, operand, offsets, sizes, strides);
        sliced_operands.push_back(slice->getResult(0));
        slice_ops.push_back(slice);
    }

    auto origin_output_type = llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
    std::vector<int64_t> new_output_shape;
    // for (auto size : sizes) {
    //     auto size_attr = llvm::dyn_cast<mlir::Attribute>(size);
    //     assert(size_attr && "size should be an attribute");
    //     auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(size_attr);
    //     assert(intAttr && "size should be an integer attribute");
    //     new_output_shape.push_back(intAttr.getInt());
    // }
    for (auto size : sizes) {
        if (auto attr = llvm::dyn_cast<mlir::Attribute>(size)) {
            auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr);
            assert(intAttr && "size should be an integer attribute");
            new_output_shape.push_back(intAttr.getInt());
        } else if (auto value = llvm::dyn_cast<mlir::Value>(size)) {
            new_output_shape.push_back(mlir::ShapedType::kDynamic);
        }
    }
    auto new_output_type = mlir::RankedTensorType::get(new_output_shape, origin_output_type.getElementType(), origin_output_type.getEncoding());
    auto tiledAddOp = builder.create<AddOp>(loc, new_output_type, sliced_operands);

    return mlir::TilingResult{{tiledAddOp}, {tiledAddOp->getResult(0)}, slice_ops};
}

::llvm::LogicalResult AddOp::getResultTilePosition(::mlir::OpBuilder &builder, unsigned resultNumber, ::mlir::ArrayRef<::mlir::OpFoldResult> offsets, ::mlir::ArrayRef<::mlir::OpFoldResult> sizes, ::mlir::SmallVector<::mlir::OpFoldResult> &resultOffsets, ::mlir::SmallVector<::mlir::OpFoldResult> &resultSizes) {
    resultOffsets = llvm::SmallVector<mlir::OpFoldResult>(offsets.begin(), offsets.end());
    resultSizes = llvm::SmallVector<mlir::OpFoldResult>(sizes.begin(), sizes.end());
    return llvm::success();
}

::mlir::SmallVector<::mlir::utils::IteratorType> MatmulOp::getLoopIteratorTypes() {
    return {};
}

::mlir::SmallVector<::mlir::Range> MatmulOp::getIterationDomain(mlir::OpBuilder&) {
    return {};
}

::mlir::FailureOr<::mlir::TilingResult> MatmulOp::getTiledImplementation(mlir::OpBuilder&, llvm::ArrayRef<mlir::OpFoldResult>, llvm::ArrayRef<mlir::OpFoldResult>) {
    return {};
}

::llvm::LogicalResult MatmulOp::getResultTilePosition(mlir::OpBuilder&, unsigned int, llvm::ArrayRef<mlir::OpFoldResult>, llvm::ArrayRef<mlir::OpFoldResult>, llvm::SmallVector<mlir::OpFoldResult, 6u>&, llvm::SmallVector<mlir::OpFoldResult, 6u>&) {
    return llvm::failure();
}

} // namespace top
} // namespace xc