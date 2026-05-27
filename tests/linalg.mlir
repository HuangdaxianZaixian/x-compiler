// RUN: mlir-opt %s -convert-linalg-to-loops -convert-scf-to-cf -convert-func-to-llvm | FileCheck %s

module {
  // 矩阵加法示例
  func.func @matmul(%A: memref<4x8xf32>, %B: memref<8x16xf32>, %C: memref<4x16xf32>) {
    linalg.matmul ins(%A, %B : memref<4x8xf32>, memref<8x16xf32>)
                  outs(%C : memref<4x16xf32>)
    return
  }
}