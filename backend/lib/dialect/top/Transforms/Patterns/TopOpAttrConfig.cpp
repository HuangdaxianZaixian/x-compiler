#include "TopOpAttrConfig.hpp"
#include "dialect/top/IR/TopTraits.hpp"
#include "mlir/Interfaces/TilingInterface.h"
#include "utils/com_utils.hpp"
#include "dialect/top/IR/TopOps.hpp"
#include "llvm/Support/raw_ostream.h"

namespace xc {
namespace top {

mlir::LogicalResult TopOpAttrConfig::matchAndRewrite(mlir::Operation *op,
                                 mlir::PatternRewriter &rewriter) const {
  llvm::outs() << "sim op code = " << op->getName().getStringRef() << "\n";
  if (op) {
    if (op->getNumResults() > 0) {
      auto result = op->getResult(0);

      auto op_index = OpIndexType::get(op->getContext(), 11);
      op->setAttr("op_index", OpIndexAttr::get(op->getContext(), op_index));

      shardingAttr shard_attr = shardingAttr::get(op->getContext(), 4, {32, 32, 32, 16});
      xc::utils::setValueEncodingAttr(result, rewriter.getNamedAttr("sharding", shard_attr));
      xc::utils::updateGraphOutputType(result);

      if (shard_attr.hasTrait<ShardingTrait>()) {
        llvm::outs() << "sharding trait found" << "\n";
      }
    }

    if (llvm::isa<top::MatmulOp>(op)) {
      auto tiling_interface = llvm::dyn_cast<TilingInterface>(op);
      if (tiling_interface) {
        llvm::outs() << "TilingInterface found for MatmulOp" << "\n";
      } else {
        llvm::outs() << "TilingInterface not found for MatmulOp" << "\n";
      }
    }

    return mlir::success();
  }
  
  return mlir::failure();
}

} // namespace top
} // namespace xc