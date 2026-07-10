// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

#map_2d = affine_map<(d0, d1) -> (d0, d1)>
#map_row = affine_map<(d0, d1) -> (d0)>

module {
  func.func @matmul_add_softmax(
      %A: tensor<4x8xf32>,
      %B: tensor<8x16xf32>,
      %bias: tensor<4x16xf32>
  ) -> tensor<4x16xf32> {
    // 初始化 matmul 输出 tensor
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<4x16xf32>
    %zero = linalg.fill ins(%c0 : f32) outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
    
    // Step 1: matmul_result = A @ B
    %matmul_result = linalg.matmul 
      ins(%A, %B : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%zero : tensor<4x16xf32>) -> tensor<4x16xf32>
    
    // Step 2: add_result = matmul_result + bias
    %add_init = tensor.empty() : tensor<4x16xf32>
    %add_result = linalg.add 
      ins(%matmul_result, %bias : tensor<4x16xf32>, tensor<4x16xf32>)
      outs(%add_init : tensor<4x16xf32>) -> tensor<4x16xf32>
    
    // Step 3: softmax_result = softmax(add_result)
    %softmax_init = tensor.empty() : tensor<4x16xf32>
    %softmax_result = linalg.softmax dimension(1)
      ins(%add_result : tensor<4x16xf32>)
      outs(%softmax_init : tensor<4x16xf32>) -> tensor<4x16xf32>
    
    return %softmax_result : tensor<4x16xf32>
  }
}