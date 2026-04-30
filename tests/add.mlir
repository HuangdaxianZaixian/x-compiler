module {
  func.func @eletw_func(%a: tensor<1x255x32xf32>, %b: tensor<1x255x32xf32>) -> tensor<1x255x255xf32> {
    %0 = "top.constant"() : () -> tensor<1x32x255xf32>
    %1 = "top.add"(%a, %b) : (tensor<1x255x32xf32>, tensor<1x255x32xf32>) -> tensor<1x255x32xf32>
    %2 = "top.matmul"(%1, %0) : (tensor<1x255x32xf32>, tensor<1x32x255xf32>) -> tensor<1x255x255xf32>
    return %2 : tensor<1x255x255xf32>
  }
}