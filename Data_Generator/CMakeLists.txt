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

set(MAIN_EXECUTABLE_NAME Onehot_Generator) 
set(PREDICT_EXECUTABLE_NAME Predictor)

add_executable(${MAIN_EXECUTABLE_NAME} ${SRC_FILES} main.cc)
# add_executable(${PREDICT_EXECUTABLE_NAME} ${SRC_FILES} data_generator.cc)
add_executable(${PREDICT_EXECUTABLE_NAME} ${SOURCE_DIR}/utils/utils_2.cc ${SOURCE_DIR}/read_params/Config.cc data_generator.cc)

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

