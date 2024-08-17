#include "src/read_params/Config.h"
#include "src/include/MACRO.h"
#include "src/utils/Utils_2.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>

std::string config_file_path = PROJECT_PATH "/config.txt";

// Templated function to generate and return a random series
template <typename T>
std::vector<T> random_vec_generator(size_t length, T min_value, T max_value) {
    static_assert(std::is_integral<T>::value, "T must be an integral type");            // static_assert is a check happend in compile stage
    assert(min_value <= max_value);
    assert(length != 0); 
    
    std::random_device rd; // Hardware random number generator
    std::uniform_int_distribution<T> distrib(min_value, max_value);

    std::vector<T> series;
    series.reserve(length); // Reserve memory to avoid reallocations

    for (size_t i = 0; i < length; ++i) 
        series.emplace_back(distrib(rd)); // Generate and store random number

    return series;
}

int main()
{
    size_t generate_size = 1000000;
    size_t min_index;
    size_t max_index;
    // 读取参数
    Config::initialize_config(config_file_path);
    auto raw_data_path = Config::get_config_value<std::string>("predict_data_read_path");
    auto data_saving_path = Config::get_config_value<std::string>("predict_data_saving_path");
    auto data_cls = Config::get_config_matrix<size_t>("data_type");
    // 读取数据
    auto raw_data = read_2d_datas<std::string>(raw_data_path);
    raw_data.assign(raw_data.begin()+1, raw_data.end());            // 去除首行
    if(raw_data.size() < 1)
        throw std::runtime_error("Empty raw datas");
    std::unordered_map<size_t, size_t> maps;
    for(size_t i = 0; i < data_cls.size(); i += 1)
        for(size_t j = 0; j < data_cls[i].size(); j+=1)
            maps[data_cls[i][j]] = i;
    // 判别类型
    decltype(raw_data) pin_data;
    decltype(raw_data) nip_data;
    pin_data.reserve(raw_data.size());
    nip_data.reserve(raw_data.size());
    min_index = 0;
    max_index = raw_data.size();

    auto build_new_data = [&](std::vector<std::vector<std::string>>& data_structure) -> std::vector<std::string>
    {
        // 产生随机数
        std::vector<size_t> random_pos = random_vec_generator(maps.size(), size_t(0), data_structure.size()); 
        std::vector<std::string> new_result_line;
        new_result_line.reserve(data_structure[0].size());
        // 构建随机选择的新数据
        for(size_t i = 0; i < random_pos.size(); i+=1)
            new_result_line.emplace_back(data_structure[random_pos[i]][i]);  
        return new_result_line;
    };

    // 随机构建
    for(auto line : raw_data)
        if(line[6].compare("nip") == 0)
            nip_data.emplace_back(line);
        else
            pin_data.emplace_back(line);
    decltype(raw_data) results;
    results.reserve(generate_size);

    // 正型，反型数据各构建一半
    for(size_t i = 0; i < generate_size/2; i+=1)
    {
        results.emplace_back(build_new_data(nip_data));
        results.emplace_back(build_new_data(pin_data));
    }

    // 存储数据
    write_content_to_file(results, data_saving_path);
    return 0;
}

