cmake_minimum_required(VERSION 3.13)

project(xc-flow)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CMAKE_BUILD_TYPE Debug)

# generate build/compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LLVM_DIR ${PROJECT_SOURCE_DIR}/third_party/llvm-project/build/lib/cmake/llvm)
set(MLIR_DIR ${PROJECT_SOURCE_DIR}/third_party/llvm-project/build/lib/cmake/mlir)

find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

message(STATUS "LLVM_DEBUG: ${LLVM_DEBUG}")


include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_compile_options(-fno-rtti)

include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/backend")
include_directories("${PROJECT_SOURCE_DIR}/backend/lib")
include_directories("${PROJECT_SOURCE_DIR}/backend/include")
include_directories("${CMAKE_BINARY_DIR}")
include_directories("${CMAKE_BINARY_DIR}/backend/lib")
include_directories("${CMAKE_BINARY_DIR}/backend/include")

add_subdirectory(backend)
