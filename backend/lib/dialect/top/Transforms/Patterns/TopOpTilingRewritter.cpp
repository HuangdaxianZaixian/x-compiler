#include "TopOpTilingRewritter.hpp"
#include "dialect/top/IR/TopTraits.hpp"
#include "mlir/Interfaces/TilingInterface.h"
#include "utils/com_utils.hpp"
#include "dialect/top/IR/TopOps.hpp"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

namespace xc {
namespace top {

mlir::LogicalResult TopOpTilingRewritter::matchAndRewrite(mlir::Operation *op,
                                 mlir::PatternRewriter &rewriter) const {
  llvm::outs() << "tiling op = " << op->getName().getStringRef() << "\n";
  if (op && llvm::isa<AddOp>(op)) {
    auto tilingInterface = llvm::dyn_cast<TilingInterface>(op);
        
    scf::SCFTilingOptions options;
    options.setTileSizes({rewriter.getIntegerAttr(rewriter.getIntegerType(32), 1), rewriter.getIntegerAttr(rewriter.getIntegerType(32), 16), rewriter.getIntegerAttr(rewriter.getIntegerType(32), 16)});
    
    FailureOr<scf::SCFTilingResult> result = scf::tileUsingSCF(rewriter, tilingInterface, options);
    
    if (failed(result))
        return failure();
    
    rewriter.replaceOp(op, result->replacements);
    return success();
  }
  
  return mlir::failure();
}

} // namespace top
} // namespace xc