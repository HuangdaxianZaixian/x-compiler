#ifndef SPLIT_PATTERNS__H
#define SPLIT_PATTERNS__H

#include "Dialect/XRT/Rewrites/OpSplit/OpSplitUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/IR/Module.h"

namespace xp_mlir {
namespace xrt {

void pattern_matmul_k6144_n1536(mlir::ModuleOp moduleOp);
void pattern_matmul_k1536_n6144(mlir::ModuleOp moduleOp);
void pattern_pertoken(mlir::ModuleOp moduleOp);
void pattern_softmax(mlir::ModuleOp moduleOp);
void pattern_moe_0(mlir::ModuleOp moduleOp);
}
}

#endif // SPLIT_PATTERNS__H