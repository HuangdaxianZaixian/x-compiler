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

#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::func::FuncDialect,
                  xc::top::TopDialect>();
}

void registerAllPasses() {
  xc::top::registerTopPasses();
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
  
    mlir::MlirOptMainConfig config = mlir::MlirOptMainConfig::createFromCLOptions();
    auto basePipelineSetup =
        [config](mlir::PassManager &pm) mutable { return config.setupPassPipeline(pm); };
    config.setPassPipelineSetupFn(
        [basePipelineSetup](mlir::PassManager &pm) mutable {
          if (llvm::failed(basePipelineSetup(pm))) {
            return mlir::failure();
          }
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