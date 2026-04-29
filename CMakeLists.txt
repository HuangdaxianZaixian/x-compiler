cmake_minimum_required(VERSION 3.13)

project(xc-flow)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CMAKE_BUILD_TYPE Debug)

set(LLVM_DIR /dataset-cpfs1/compiler/huangy43/repos/pro_mlir/third_party/llvm-install/lib/cmake/llvm)
set(MLIR_DIR /dataset-cpfs1/compiler/huangy43/repos/pro_mlir/third_party/llvm-install/lib/cmake/mlir)

find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})


include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_compile_options(-fno-rtti)

include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/backend/lib")
include_directories("${PROJECT_SOURCE_DIR}/backend/include")
include_directories("${CMAKE_BINARY_DIR}")
include_directories("${CMAKE_BINARY_DIR}/backend/lib")
include_directories("${CMAKE_BINARY_DIR}/backend/include")

add_subdirectory(backend/include)
add_subdirectory(backend/lib)
add_subdirectory(backend/tools)
