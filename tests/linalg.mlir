// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

module {
  func.func @matmul_add_softmax(
      %A: tensor<128x256xf32>,
      %B: tensor<256x512xf32>,
      %bias: tensor<128x512xf32>
  ) -> tensor<128x512xf32> {
    // 初始化 matmul 输出 tensor
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128x512xf32>
    %zero = linalg.fill ins(%c0 : f32) outs(%init : tensor<128x512xf32>) -> tensor<128x512xf32>
    
    // Step 1: matmul_result = A @ B
    %matmul_result = linalg.matmul 
      ins(%A, %B : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%zero : tensor<128x512xf32>) -> tensor<128x512xf32>
    
    // Step 2: add_result = matmul_result + bias
    %add_init = tensor.empty() : tensor<128x512xf32>
    %add_result = linalg.add 
      ins(%matmul_result, %bias : tensor<128x512xf32>, tensor<128x512xf32>)
      outs(%add_init : tensor<128x512xf32>) -> tensor<128x512xf32>
    
    // Step 3: softmax_result = softmax(add_result)
    %softmax_init = tensor.empty() : tensor<128x512xf32>
    %softmax_result = linalg.softmax dimension(1)
      ins(%add_result : tensor<128x512xf32>)
      outs(%softmax_init : tensor<128x512xf32>) -> tensor<128x512xf32>
    
    return %softmax_result : tensor<128x512xf32>
  }
}