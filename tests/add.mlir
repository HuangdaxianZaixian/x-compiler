module {
  func.func @eletw_func(%a: tensor<1x255x255x32xf32>, %b: tensor<1x255x255x32xf32>) -> tensor<1x255x255x32xf32> {
    %0 = "top.add"(%a, %b) : (tensor<1x255x255x32xf32>, tensor<1x255x255x32xf32>) -> tensor<1x255x255x32xf32>
    return %0 : tensor<1x255x255x32xf32>
  }
}