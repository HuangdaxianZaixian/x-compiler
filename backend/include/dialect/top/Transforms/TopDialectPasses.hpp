#ifndef TOP_DIALECT_PASSES__HPP
#define TOP_DIALECT_PASSES__HPP

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "dialect/top/IR/TopOps.hpp"

using namespace mlir;

namespace xc {
namespace top {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createTopOpCheckPass();

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "dialect/top/Transforms/TopDialectPasses.h.inc"

} // namespace top
} // namespace xc

#endif // TOP_DIALECT_PASSES__HPP