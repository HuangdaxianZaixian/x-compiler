#include "dialect/top/IR/TopOps.hpp"
#include "utils/com_utils.hpp"

namespace xc {
namespace top {

void AddOp::inferShapes() {
    auto input0_shape = xc::utils::getTensorShape(getOperand(0));
    auto input1_shape = xc::utils::getTensorShape(getOperand(1));
    assert(input0_shape.size() == input1_shape.size() && "input shapes must have the same rank");
    std::vector<int64_t> output_shape;
    for (size_t i = 0; i < input0_shape.size(); ++i) {
        output_shape.push_back(std::max(input0_shape[i], input1_shape[i]));
    }

    auto output_val = getResult();
    xc::utils::setTensorType(output_val, output_shape, xc::utils::getTensorElementType(output_val), xc::utils::getTensorEncodingAttr(output_val));
}

void MatmulOp::inferShapes() {
    auto ifm_shape = xc::utils::getTensorShape(getOperand(0));
    auto wgts_shape = xc::utils::getTensorShape(getOperand(1));
    assert(ifm_shape.size() >= 2 && "ifm_shape must have rank >= 2");
    assert(wgts_shape.size() >= 2 && "wgts_shape must have rank >= 2");

    std::vector<int64_t> output_shape = ifm_shape.size() > wgts_shape.size() ? ifm_shape : wgts_shape;
    output_shape[output_shape.size() - 1] = wgts_shape[wgts_shape.size() - 1];
    output_shape[output_shape.size() - 2] = ifm_shape[ifm_shape.size() - 2];

    auto output_val = getResult();
    xc::utils::setTensorType(output_val, output_shape, xc::utils::getTensorElementType(output_val), xc::utils::getTensorEncodingAttr(output_val));
}

} // namespace top
} // namespace xc