#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Threading.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "dialect/top/IR/TopOps.hpp"
#include "dialect/top/Transforms/TopDialectPasses.hpp"
#include "dialect/linalg/TilingConfigManager.hpp"

#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

struct IncrementalPassTimingInstrumentation final : public mlir::PassInstrumentation {
  struct ActivePass {
    std::string passName;
    std::string opName;
    std::chrono::steady_clock::time_point startTime;
  };

  llvm::DenseMap<uint64_t, llvm::SmallVector<ActivePass, 4>> activePasses;
  std::mutex outputMutex;

  static std::string getPassDisplayName(mlir::Pass *pass) {
    llvm::StringRef argument = pass->getArgument();
    if (!argument.empty()) {
      return argument.str();
    }
    return std::string(pass->getName());
  }

  void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override {
    activePasses[llvm::get_threadid()].push_back(
        {getPassDisplayName(pass),
         op ? op->getName().getStringRef().str() : std::string("<null-op>"),
         std::chrono::steady_clock::now()});
  }

  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override {
    printCompletedPassTiming(/*failed=*/false);
  }

  void runAfterPassFailed(mlir::Pass *pass, mlir::Operation *op) override {
    printCompletedPassTiming(/*failed=*/true);
  }

  void printCompletedPassTiming(bool failed) {
    auto &stack = activePasses[llvm::get_threadid()];
    if (stack.empty()) {
      return;
    }

    ActivePass activePass = std::move(stack.back());
    stack.pop_back();

    auto elapsedSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      activePass.startTime)
            .count();

    std::lock_guard<std::mutex> lock(outputMutex);
    llvm::errs() << "[mlir-timing] "
                 << (failed ? "FAIL" : "PASS")
                 << " pass='" << activePass.passName << "'"
                 << " op='" << activePass.opName << "'"
                 << " time_s=" << llvm::format("%.4f", elapsedSeconds) << "\n";
  }
};

void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::func::FuncDialect,
                  mlir::tensor::TensorDialect,
                  mlir::scf::SCFDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::arith::ArithDialect,
                  xc::top::TopDialect>();
}

void registerAllPasses() {
  xc::top::registerTopPasses();
  mlir::registerLinalgPasses();
  mlir::registerLinalgTilingPass();
  
}

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    registerAllDialects(registry);
  
    mlir::registerCanonicalizer();
    mlir::registerCSEPass();
    registerAllPasses();
  
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
  
    std::vector<char *> args(argv, argv + argc);
  
  
    std::string inputFilename, outputFilename;
    std::tie(inputFilename, outputFilename) = mlir::registerAndParseCLIOptions(
        args.size(), args.data(), "XP MLIR module optimizer driver\n", registry);
    
    // 把全局注册的CLI选项快照进MlirOptMainConfig对象
    // 真正构建pass pipeline发生mlir-opt内部调用config.setupPassPipeline(pm)时
    mlir::MlirOptMainConfig config = mlir::MlirOptMainConfig::createFromCLOptions();
    auto basePipelineSetup = [config](mlir::PassManager &pm) mutable { return config.setupPassPipeline(pm); };
    // 此处是定制setupPassPipeline函数的行为
    // 比如下面在每个pass后面增加一个Instrumentation来统计每个pass的执行时间
    config.setPassPipelineSetupFn(
        [basePipelineSetup](mlir::PassManager &pm) mutable {
          if (llvm::failed(basePipelineSetup(pm))) {
            return mlir::failure();
          }
          pm.addInstrumentation(
            std::make_unique<IncrementalPassTimingInstrumentation>());
          return mlir::success();
        });
  
    std::string errorMessage;
    auto file = mlir::openInputFile(inputFilename, &errorMessage);
    if (!file) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }
  
    auto output = mlir::openOutputFile(outputFilename, &errorMessage);
    if (!output) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }
  
    if (llvm::failed(mlir::MlirOptMain(output->os(), std::move(file), registry, config))) {
      return 1;
    }
  
    output->keep();
    return 0;
  }