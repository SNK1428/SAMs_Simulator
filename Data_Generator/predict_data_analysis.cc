#include "src/read_params/Config.h"
#include "src/include/MACRO.h"
#include "src/utils/Utils_2.h"
#include "src/include/IMPORT_LIB.h"
#include <cassert>
#include <cstddef>
#include <string>

int main()
{
    auto predict_data_raw_path = Config::get_config_value<std::string>("predict_data_saving_path");
    // read data
    auto predict_results = read_2d_datas<double>("");
    // read raw_data
    auto raw_data = read_2d_datas<std::string>(predict_data_raw_path);   
    // re_rank_data
    assert(predict_results.size() == raw_data.size()); 
    
    for(size_t i = 0; i < predict_results.size(); i+=1)
    {
        raw_data[i].push_back(std::to_string(predict_results[i][0]));
        raw_data[i].push_back(std::to_string(predict_results[i][1]));
    }
    //record marked data
    write_content_to_file(raw_data, ""); 
    return 0;
}
