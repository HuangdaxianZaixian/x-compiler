#include "TilingConfigManager.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;

namespace {

/// 初始化 tiling 配置
void initializeTilingConfigs() {
  auto &manager = TilingConfigManager::getInstance();
  
  // MatMul 配置：根据形状动态决定
  manager.registerConfig<linalg::MatmulOp>([](mlir::Operation* op) {
    TilingSpec spec;
    auto matmul = llvm::dyn_cast<linalg::MatmulOp>(op);
    
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
  manager.registerConfig<linalg::AddOp>([](mlir::Operation* op) {
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
  manager.registerConfig<linalg::SoftmaxOp>([](mlir::Operation* op) {
    TilingSpec spec;
    auto softmax = llvm::dyn_cast<linalg::SoftmaxOp>(op);
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
    
    SmallVector<mlir::Operation*> linalgOps;
    funcOp.walk([&](mlir::Operation* op) {
        if (op && llvm::isa<TilingInterface>(op)) {
            linalgOps.push_back(op);
        }
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

        auto tiling_inft = llvm::dyn_cast<TilingInterface>(op);
        assert(tiling_inft && "Operation does not implement TilingInterface");
        
        rewriter.setInsertionPoint(op);
        auto tilingResult = scf::tileUsingSCF(
            rewriter, llvm::dyn_cast<TilingInterface>(op), options);
        
        if (succeeded(tilingResult)) {
            rewriter.replaceOp(op, tilingResult->replacements);
        }
    }
  }
};

struct LinalgTileAndFusePass
    : public PassWrapper<LinalgTileAndFusePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgTileAndFusePass)

  StringRef getArgument() const override { return "custom-linalg-tile-and-fuse"; }
  StringRef getDescription() const override {
    return "Custom Linalg tile and fuse pass";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect, 
                    tensor::TensorDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  LinalgTileAndFusePass() {
    initializeTilingConfigs();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(&getContext());
    auto &manager = TilingConfigManager::getInstance();
    
    // 找到所有的 "root" 操作（没有被其他 TilingInterface op 消费的操作）
    SmallVector<Operation *> rootOps;
    funcOp.walk([&](Operation *op) {
      if (!isa<TilingInterface>(op))
        return;
      
      // 检查这个 op 的结果是否只被非 TilingInterface op 使用
      bool isRoot = true;
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (isa<TilingInterface>(user)) {
            isRoot = false;
            break;
          }
        }
        if (!isRoot) break;
      }

      if (isRoot) {
        rootOps.push_back(op);
      }
    });
    
    // 对每个 root op 执行 tile and fuse
    for (Operation *rootOp : rootOps) {
      tileAndFuseFromRoot(rewriter, rootOp, manager);
    }
  }

private:
  void tileAndFuseFromRoot(IRRewriter &rewriter, Operation *rootOp,
                           TilingConfigManager &manager) {
    TilingSpec spec = manager.getConfig(rootOp);
    
    if (spec.tileSizes.empty() || 
        llvm::all_of(spec.tileSizes, [](int64_t s) { return s == 0; })) {
      return;
    }
    
    // 构建 tile sizes
    SmallVector<OpFoldResult> tileSizesOfr;
    for (int64_t size : spec.tileSizes) {
      tileSizesOfr.push_back(rewriter.getIndexAttr(size));
    }
    
    // 设置 SCFTileAndFuseOptions
    scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.setTilingOptions(
        scf::SCFTilingOptions().setTileSizes(tileSizesOfr));
    
    // 设置融合控制函数：融合所有可融合的生产者
    tileAndFuseOptions.setFusionControlFn(
      [](tensor::ExtractSliceOp candidateSlice, // 从生产者结果中提取的切片操作，表示消费者需要的数据切片
                    OpResult originalProducer, // 生产者操作的结果值，即产生被切片数据的那个操作的输出
                    // 在linalg中, 正常的op输入被称为input operand, 输出被称为destination operand
                    // linalg中采用dps(destination-passing style)风格, 输出也是需要传入的, 所以需要区分operand的类型
                    // 比如
                    // %matmul_result = linalg.matmul 
                    //        ins(%A, %B : tensor<128x256xf32>, tensor<256x512xf32>)
                    //        outs(%zero : tensor<128x512xf32>) -> tensor<128x512xf32>
                    // 这里的outs来自
                    // %zero = linalg.fill ins(%c0 : f32) outs(%init : tensor<128x512xf32>) -> tensor<128x512xf32>
                    // 而对于destination operand, 它通常作为op的iter_args初始值, 需要yield这个值来更新iter_args
                    // 所以这里需要区分producer是作为input operand, 还是destination operand, 来决定是否需要yield producer的结果
                    bool isDestinationOperand) 
            -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
          // 通常可以根据candidataSlice, 判断切片大小, 觉得是否融合
          // 可以根据originalProducer的生产者op类型, 觉得是否融合
          // 当前没有做特殊判断, 返回值表示"无条件融合所有生产者"
          return scf::SCFTileAndFuseOptions::ControlFnResult{
              /*yieldProducerReplacement=*/isDestinationOperand};
        });
    
    rewriter.setInsertionPoint(rootOp);
    
    auto tilingInterfaceOp = cast<TilingInterface>(rootOp);
    auto tileAndFuseResult = scf::tileConsumerAndFuseProducersUsingSCF(
        rewriter, tilingInterfaceOp, tileAndFuseOptions);
    
    if (failed(tileAndFuseResult)) {
      return;
    }
    
    // 替换原始操作
    for (auto [origVal, replacement] : tileAndFuseResult->replacements) {
      rewriter.replaceAllUsesWith(origVal, replacement);
    }
    
    // 删除已被融合的操作
    SmallVector<Operation *> opsToDelete;
    opsToDelete.push_back(rootOp);
    
    for (auto fusedOp : tileAndFuseResult->fusedProducers) {
      opsToDelete.push_back(fusedOp);
    }
    
    for (Operation *op : opsToDelete) {
      if (op->use_empty()) {
        rewriter.eraseOp(op);
      }
    }
  }
};

} // namespace

namespace mlir {
void registerLinalgTilingPass() {
  PassRegistration<LinalgTilingPass>();
}

void registerLinalgTileAndFusePass() {
    PassRegistration<LinalgTileAndFusePass>();
}
} // namespace mlir