#ifndef COMMON_UTILS__HPP
#define COMMON_UTILS__HPP

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

namespace xc {
namespace utils {

void setValueEncodingAttr(mlir::Value val, mlir::NamedAttribute attr);
void updateGraphInputType(const mlir::Value &val);
void updateGraphOutputType(const mlir::Value &val);
void moduleDump(const mlir::ModuleOp &module, const std::string &output_file_path, bool if_dump_large_data);

} // namespace utils
} // namespace xc

#endif // COMMON_UTILS__HPP