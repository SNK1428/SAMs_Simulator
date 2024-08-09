#include "num_encode.h"
#include "../utils/Utils_2.h"
#include <algorithm>
#include <cstddef>
#include <ios>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// 构建数字padding
// constraint_block_size : 规定 | 和 >> 切分区块最大值
// constraint_padding_length : 规定每一个block（区块）元素最大值，即每个block中以;切分的数据的最大值，会补全所有区块
std::vector<std::vector<std::string>> build_num_onehot(const std::vector<std::string> &str_array, const size_t constraint_padding_length, const size_t constraint_block_size)
{
    // 依据 | 和 >> 切分数据
    auto split_2d_vec = [](const std::vector<std::string> &str_array) {
        const static std::vector<std::string> inner_delimiter = {"|", ">>"};
        std::vector<std::vector<std::string>> splited_results(str_array.size());
        for (size_t i = 0; i < str_array.size(); i += 1)
            splited_results[i] = split_str_multi_sep_trim(str_array[i], inner_delimiter);
        return splited_results;
    };

    // 依据 ; 切分，并存到一个规定大小的新矩阵中
    auto split_elements = [](const std::vector<std::vector<std::string>> &str_array, const size_t str_array_max_columns_size) {
        std::vector<std::vector<std::vector<std::string>>> each_padding_unit(str_array.size(), std::vector<std::vector<std::string>>(str_array_max_columns_size));
        for (size_t i = 0; i < str_array.size(); i += 1)
            for (size_t j = 0; j < std::min(str_array[i].size(), str_array_max_columns_size); j += 1)
                each_padding_unit[i][j] = split_string_trim(str_array[i][j], ";");
        return each_padding_unit;
    };

    // 切分block, 如果没有规定限制block值，则自行决定
    std::vector<std::vector<std::string>> splited_arr = split_2d_vec(str_array);
    size_t actual_block_size = constraint_block_size;
    for (auto ele : splited_arr)
        if (ele.size() > actual_block_size)
            actual_block_size = ele.size();

    // 排除空数组问题
    if (splited_arr.size() == 0 || actual_block_size == 0)
        throw std::runtime_error("Empty data columns");

    if (constraint_block_size != 0)
        actual_block_size = constraint_block_size;

    // 如果规定了每个区块的补齐尺寸，则判断其乘积是否会溢出
    if (constraint_padding_length != 0 && actual_block_size > std::numeric_limits<size_t>::max() / constraint_padding_length)
        throw std::runtime_error("Num overflow in method build_num_onehot");

    // 构建待onehot的初始数据
    auto each_padding_block = split_elements(splited_arr, actual_block_size);
    // 如果没有规定补齐尺寸，则通过获取拼接后可能的最大尺寸，判断是否溢出
    std::vector<std::vector<std::string>> return_results;

    // 未定义每一列补齐的尺度
    if (constraint_padding_length == 0)
    {
        size_t actual_onehot_eigen_size = 0;
        for (size_t i = 0; i < each_padding_block[0].size(); i += 1)
        {
            // 为保证数组存在，padding值最少也要为1
            size_t actual_onehot_eigen_size_block = 1;
            for (size_t j = 0; j < each_padding_block.size(); j += 1)
                if (each_padding_block[j][i].size() > actual_onehot_eigen_size_block)
                    actual_onehot_eigen_size_block = each_padding_block[j][i].size();
            // 如果尺寸已经溢出，则终止此代码
            if (actual_onehot_eigen_size > std::numeric_limits<size_t>::max() - actual_onehot_eigen_size_block)
                throw std::runtime_error("Num overflow in method build_num_onehot");

            // 构建补全数组
            std::vector<std::vector<std::string>> block_arr(each_padding_block.size(), std::vector<std::string>(actual_onehot_eigen_size_block, "0"));
            // 进行补全
            for (size_t j = 0; j < each_padding_block.size(); j += 1)
                for (size_t k = 0; k < each_padding_block[j][i].size(); k += 1)
                    block_arr[j][k] = each_padding_block[j][i][k];
            // 拼接
            if (return_results.size() == 0)
                return_results = block_arr;
            else
                return_results = concatenate_matrices(return_results, std::move(block_arr), 1);
            // 重新计数
            actual_onehot_eigen_size += actual_onehot_eigen_size_block;
        }
    }
    // 定义了每一列补齐的长度
    else
    {
        // 固定补齐每一行的尺寸
        for (size_t i = 0; i < each_padding_block[0].size(); i += 1)
        {
            std::vector<std::vector<std::string>> block_arr(each_padding_block.size(), std::vector<std::string>(constraint_padding_length, "0"));
            for (size_t j = 0; j < each_padding_block.size(); j += 1)
                for (size_t k = 0; k < std::min(each_padding_block[j][i].size(), constraint_padding_length); k += 1)
                    block_arr[j][k] = each_padding_block[j][i][k];
            if (return_results.size() == 0)
                return_results = block_arr;
            else
                return_results = concatenate_matrices(return_results, std::move(block_arr), 1);
        }
    }
    std::cout << ", blocks: " << constraint_block_size << ", maps: " << constraint_padding_length;

    return return_results;
}
