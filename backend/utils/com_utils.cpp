#include "utils/com_utils.hpp"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Support/FileUtilities.h"
#include <cassert>

namespace xc {
namespace utils {

void setValueEncodingAttr(mlir::Value val, mlir::NamedAttribute attr) {
    auto type = val.getType();
    const std::string attr_name = attr.getName().getValue().str();

    auto ctx = type.getContext();
    auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(type);
    assert(tensorType && "error value type");

    llvm::SmallVector<mlir::NamedAttribute> attrList;
    if (auto encoding = tensorType.getEncoding()) {
        auto dictAttr = llvm::dyn_cast<mlir::DictionaryAttr>(encoding);
        assert(dictAttr != nullptr && "missing tensor attr!");
        for (auto attr_ : dictAttr) {
            if (attr_.getName().getValue() != attr_name) attrList.emplace_back(attr_);
        }
    }
    attrList.emplace_back(attr);
    auto newDictAttr = mlir::DictionaryAttr::get(ctx, attrList);
    auto new_type = mlir::RankedTensorType::get(tensorType.getShape(), tensorType.getElementType(), newDictAttr);
    val.setType(new_type);
}

void updateGraphInputType(const mlir::Value &val) {
  mlir::BlockArgument arg = llvm::dyn_cast_if_present<mlir::BlockArgument>(val);
  if (arg == nullptr)
    return;

  auto funcOp =
      llvm::dyn_cast<mlir::func::FuncOp>(arg.getParentBlock()->getParentOp());
  if (funcOp == nullptr)
    return;
  auto allIns = funcOp.getArguments();
  std::vector<mlir::Type> newTypes;
  for (auto in : allIns) {
    if (in == val) {
      newTypes.push_back(val.getType());
    } else {
      newTypes.push_back(in.getType());
    }
  }
  auto funcType = mlir::FunctionType::get(val.getContext(), newTypes,
                                          funcOp.getResultTypes());
  funcOp.setFunctionType(funcType);
}

void updateGraphOutputType(const mlir::Value &val) {
  mlir::func::ReturnOp returnOp{nullptr};
  for (auto oneUser : val.getUsers()) {
    auto tmp = llvm::dyn_cast<mlir::func::ReturnOp>(oneUser);
    if (tmp != nullptr) {
      returnOp = tmp;
      break;
    }
  }
  if (returnOp == nullptr)
    return;

  std::vector<mlir::Type> newFuncTypes;
  for (mlir::Value returnIns : returnOp.getOperands()) {
    newFuncTypes.push_back(returnIns.getType());
  }

  mlir::func::FuncOp funcOp =
      llvm::dyn_cast<mlir::func::FuncOp>(returnOp->getParentOp());
  if (funcOp == nullptr)
    return;

  auto funcInputTypes = funcOp.getArgumentTypes();
  auto funcType =
      mlir::FunctionType::get(val.getContext(), funcInputTypes, newFuncTypes);

  funcOp.setFunctionType(funcType);
}

void moduleDump(const mlir::ModuleOp &module, const std::string &output_file_path, bool if_dump_large_data) {
    assert(module != nullptr && "invalid module!");
    std::string errorMessage;
    auto output = mlir::openOutputFile(output_file_path, &errorMessage);
    assert(output && "can't open output file!");
    mlir::OpPrintingFlags flags;
    flags.enableDebugInfo(true);
    if (!if_dump_large_data) {
        flags.elideLargeElementsAttrs(10);
    }
    module->print(output->os(), flags);
    output->keep();
}

std::vector<int64_t> getTensorShape(mlir::Value val) {
  assert(val && "invalid value!");
  auto tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(val.getType());
  assert(tensor_type && "invalid tensor type!");

  return tensor_type.getShape();
}

void setTensorType(mlir::Value val, const std::vector<int64_t>& shape, mlir::Type data_type, mlir::Attribute attr) {
  auto rank_tensor = mlir::RankedTensorType::get(shape, data_type, attr);
  val.setType(rank_tensor);
}

mlir::Type getTensorElementType(mlir::Value val) {
  assert(val && "invalid value!");
  auto tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(val.getType());
  assert(tensor_type && "invalid tensor type!");
  return tensor_type.getElementType();
}

int64_t getTensorElementTypeBitWidth(mlir::Value val) {
  auto elem_type = getTensorElementType(val);
  assert(elem_type.isIntOrIndexOrFloat() && "invalid element type!");
  return elem_type.getIntOrFloatBitWidth();
}

mlir::Attribute getTensorEncodingAttr(mlir::Value val) {
  auto tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(val.getType());
  assert(tensor_type && "invalid tensor type!");
  return tensor_type.getEncoding();
}

} // namespace utils
} // namespace xc
