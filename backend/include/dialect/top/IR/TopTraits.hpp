#ifndef TOP_TRAITS__HPP
#define TOP_TRAITS__HPP

#include "mlir/IR/Attributes.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"

namespace xc {
namespace top {

// 框架不会自动生成Trait定义, 需要手动实现
template <typename ConcreteAttr>
class NPUTrait
    : public ::mlir::OpTrait::TraitBase<ConcreteAttr, NPUTrait> {
public:
  static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
    llvm::outs() << "op NPUTrait verify" << "\n";
    return ::mlir::success();
  }
};

template <typename ConcreteAttr>
class ShardingTrait
    : public ::mlir::AttributeTrait::TraitBase<ConcreteAttr, ShardingTrait> {
public:
  static ::mlir::LogicalResult verifyTrait(::mlir::Attribute attr) {
    llvm::outs() << "attribute ShardingTrait verify" << "\n";
    return ::mlir::success();
  }
};

} // namespace top
} // namespace xc

#endif