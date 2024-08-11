CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

# C++ 17
set(CMAKE_CXX_STANDARD 17)
PROJECT(PVSK_DATABASE)

# 启用调试信息
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/include)
CONFIGURE_FILE(${PROJECT_SOURCE_DIR}/src/include/MACRO.h.in
    ${PROJECT_SOURCE_DIR}/src/include/MACRO.h) # 获取CMake项目根目录，并传递到执行文件中

set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")

file(
    GLOB_RECURSE SRC_FILES
    "${SOURCE_DIR}/*.cpp"
    "${SOURCE_DIR}/*.cc"
    "${SOURCE_DIR}/*.h"
)

file(
    GLOB UTILS_FILES
    "{SOURCE_DIR}/Utils/*.cpp"
    "{SOURCE_DIR}/Utils/*.cc"
    "{SOURCE_DIR}/Utils/*.h"
)

set(MAIN_EXECUTABLE_NAME Onehot_Generator) 
set(PREDICT_EXECUTABLE_NAME Predictor)

add_executable(${MAIN_EXECUTABLE_NAME} ${SRC_FILES} main.cc)
add_executable(${PREDICT_EXECUTABLE_NAME}  ${UTILS_FILES} data_generator.cc)

# 禁用所有报警(gcc)
TARGET_COMPILE_OPTIONS(${MAIN_EXECUTABLE_NAME} PUBLIC -w)
TARGET_COMPILE_OPTIONS(${PREDICT_EXECUTABLE_NAME} PUBLIC -w)
# 设置编译选项
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -Wextra")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything -Wno-c++98-compat")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-c++98-compat-pedantic")
endif()

# -----------------------------------------------
# 测试类

# 设置测试可执行文件输出目录
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test_bin)

# 测试方法
# add_executable(concat_test concat_vec_to_str.cc) # 测试一维vector拼接为字符串拼接性能
# add_executable(swap_test swap_vector_col.cc)
# add_executable(swap_test_2 swap_vector_col_2.cc)
# add_executable(swap_test_3 swap_vector_col_v3.cc)
# add_executable(transpose_test transpose_mat.cc)
# add_executable(split_str_test split_str.cc)
# add_executable(vec_type_cvt_test vec_type_convert.cc)

# 恢复默认输出目录
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
