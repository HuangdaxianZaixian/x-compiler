#ifndef TILING_CONFIG_MANAGER_H
#define TILING_CONFIG_MANAGER_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/DenseMap.h"
#include <functional>

namespace mlir {

/// Tiling 配置
struct TilingSpec {
  SmallVector<int64_t> tileSizes;
  bool interchangeLoops = false;
  SmallVector<int64_t> interchange;
};

/// 配置管理器 - 支持注册自定义配置函数
class TilingConfigManager {
public:
  using ConfigFn = std::function<TilingSpec(linalg::LinalgOp)>;

  static TilingConfigManager &getInstance() {
    static TilingConfigManager instance;
    return instance;
  }

  /// 注册特定 op 类型的配置函数
  template <typename OpTy>
  void registerConfig(ConfigFn fn) {
    configs[OpTy::getOperationName()] = std::move(fn);
  }

  /// 获取 op 的 tiling 配置
  TilingSpec getConfig(linalg::LinalgOp op) {
    auto it = configs.find(op->getName().getStringRef());
    if (it != configs.end()) {
      return it->second(op);
    }
    return defaultConfig(op);
  }

  /// 设置默认配置函数
  void setDefaultConfig(ConfigFn fn) {
    defaultConfigFn = std::move(fn);
  }

private:
  TilingSpec defaultConfig(linalg::LinalgOp op) {
    if (defaultConfigFn) {
      return defaultConfigFn(op);
    }
    return TilingSpec{};
  }

  llvm::StringMap<ConfigFn> configs;
  ConfigFn defaultConfigFn;
};

void registerLinalgTilingPass();

} // namespace mlir

#endif // TILING_CONFIG_MANAGER_H