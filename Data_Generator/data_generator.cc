#include "src/read_params/Config.h"
#include "src/include/MACRO.h"
#include "src/utils/Utils_2.h"
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

std::string config_file_path = PROJECT_PATH "/config.txt";


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
    if(raw_data.size() < 0)
        throw std::runtime_error("Empty raw datas");
    // 判别类型
    decltype(raw_data) pin_data;
    decltype(raw_data) nip_data;
    pin_data.reserve(raw_data.size());
    nip_data.reserve(raw_data.size());
    min_index = 0;
    max_index = raw_data.size();
    std::vector<size_t> pos(raw_data[0].size());

    auto random_pos_geenrator = [&]()
    {
        std::cout << data_cls.size() << std::endl;
        for(auto line : data_cls)
        {
             
        }
    };

    auto build_new_data = [&](std::vector<std::vector<std::string>>& raw_data) -> std::vector<std::string>
    {

    };

    // 随机构建
    for(auto line : raw_data)
        if(line[10].compare("nip") == 0)
            nip_data.push_back(line);
        else
            pin_data.push_back(line);

    decltype(raw_data) results;
    results.reserve(generate_size);

    // 正型，反型数据各构建一半
    for(size_t i = 0; i < generate_size/2; i+=1)
    {
        random_pos_geenrator();
        results.push_back(build_new_data(nip_data));
        random_pos_geenrator();
        results.push_back(build_new_data(pin_data));
    }

    // 存储数据
    write_content_to_file(results, data_saving_path);
    return 0;
}

