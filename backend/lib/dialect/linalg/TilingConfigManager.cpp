#include "TilingConfigManager.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"

using namespace mlir;

namespace {

/// 初始化 tiling 配置
void initializeTilingConfigs() {
  auto &manager = TilingConfigManager::getInstance();
  
  // MatMul 配置：根据形状动态决定
  manager.registerConfig<linalg::MatmulOp>([](linalg::LinalgOp op) {
    TilingSpec spec;
    auto matmul = llvm::dyn_cast<linalg::MatmulOp>(op.getOperation());
    
    // 获取输入形状
    auto aType = llvm::dyn_cast<ShapedType>(matmul.getInputs()[0].getType());
    auto bType = llvm::dyn_cast<ShapedType>(matmul.getInputs()[1].getType());
    
    int64_t M = aType.getDimSize(0);
    int64_t K = aType.getDimSize(1);
    int64_t N = bType.getDimSize(1);
    
    // 复杂的 tiling 策略
    // 例如：根据 cache 大小优化
    constexpr int64_t L1_CACHE_SIZE = 32 * 1024;  // 32KB
    constexpr int64_t ELEMENT_SIZE = 4;  // f32
    
    // 计算最优 tile 大小使得 tiles 适合 L1 cache
    int64_t tileM = std::min(M, (int64_t)32);
    int64_t tileN = std::min(N, (int64_t)32);
    int64_t tileK = std::min(K, (int64_t)16);
    
    // 确保 tile 大小合理
    while (tileM * tileK + tileK * tileN + tileM * tileN > 
           L1_CACHE_SIZE / ELEMENT_SIZE && tileM > 4) {
      tileM /= 2;
      tileN /= 2;
    }
    
    spec.tileSizes = {tileM, tileN, tileK};

    // 启用循环交换
    spec.interchangeLoops = true;
    // 交换循环顺序: [0, 1, 2] -> [0, 2, 1] 
    // 这会把 K 循环移到 N 循环之前，有助于缓存友好性
    spec.interchange = {0, 2, 1};

    return spec;
  });
  
  // Add 配置
  manager.registerConfig<linalg::AddOp>([](linalg::LinalgOp op) {
    TilingSpec spec;
    auto resultType = llvm::dyn_cast<ShapedType>(op->getResult(0).getType());
    auto shape = resultType.getShape();
    
    // 向量化友好的 tile 大小
    spec.tileSizes.reserve(shape.size());
    for (int64_t dim : shape) {
      spec.tileSizes.push_back(std::min(dim, (int64_t)16));
    }
    return spec;
  });
  
  // Softmax 配置：不 tile reduction 维度
  manager.registerConfig<linalg::SoftmaxOp>([](linalg::LinalgOp op) {
    TilingSpec spec;
    auto softmax = llvm::dyn_cast<linalg::SoftmaxOp>(op.getOperation());
    int64_t reductionDim = softmax.getDimension();
    
    auto inputType = llvm::dyn_cast<ShapedType>(softmax.getInput().getType());
    auto shape = inputType.getShape();
    
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i == (size_t)reductionDim) {
        spec.tileSizes.push_back(0);  // 不 tile reduction 维度
      } else {
        spec.tileSizes.push_back(std::min(shape[i], (int64_t)4));
      }
    }
    return spec;
  });
}

struct LinalgTilingPass
    : public PassWrapper<LinalgTilingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgTilingPass)

  StringRef getArgument() const override { return "custom-linalg-tiling"; }
  StringRef getDescription() const override {
    return "Custom Linalg tiling with flexible configuration system";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect, 
                    tensor::TensorDialect>();
    // 注册 TilingInterface 实现
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  LinalgTilingPass() {
    // 初始化配置
    initializeTilingConfigs();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(&getContext());
    auto &manager = TilingConfigManager::getInstance();
    
    SmallVector<linalg::LinalgOp> linalgOps;
    funcOp.walk([&](linalg::LinalgOp op) {
      linalgOps.push_back(op);
    });
    
    for (auto op : llvm::reverse(linalgOps)) {
        TilingSpec spec = manager.getConfig(op);
        
        if (spec.tileSizes.empty() || 
            llvm::all_of(spec.tileSizes, [](int64_t s) { return s == 0; })) {
            continue;
        }
        
        scf::SCFTilingOptions options;
        SmallVector<OpFoldResult> tileSizesOfr;
        for (int64_t size : spec.tileSizes) {
            tileSizesOfr.push_back(rewriter.getIndexAttr(size));
        }
        options.setTileSizes(tileSizesOfr);

        // 设置循环交换
        if (spec.interchangeLoops && !spec.interchange.empty()) {
            options.setInterchange(spec.interchange);
        }

        auto tiling_inft = llvm::dyn_cast<TilingInterface>(op.getOperation());
        assert(tiling_inft && "Operation does not implement TilingInterface");
        
        rewriter.setInsertionPoint(op);
        auto tilingResult = scf::tileUsingSCF(
            rewriter, llvm::dyn_cast<TilingInterface>(op.getOperation()), options);
        
        if (succeeded(tilingResult)) {
            rewriter.replaceOp(op, tilingResult->replacements);
        }
    }
  }
};

} // namespace

namespace mlir {
void registerLinalgTilingPass() {
  PassRegistration<LinalgTilingPass>();
}
} // namespace mlir