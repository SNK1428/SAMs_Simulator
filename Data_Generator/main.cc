#include "src/include/MACRO.h"
#include "src/utils/Utils_2.h"

#include "src/encode_code/mapping_code.h"
#include "src/encode_code/num_encode.h"
#include "src/encode_code/regex_23.h"
#include "src/fix_data/fix_num.h"
#include "src/fix_data/fix_str.h"
#include "src/fix_data/fix_years.h"
#include "src/read_params/Config.h"
#include "src/include/IMPORT_LIB.h"
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>
#include <filesystem> // C++17及以上

// 原始数据中存在大量不合理的JV数据，进行补全过滤，保证输入模型的数据中，Jsc*Voc*FF=PCE
void device_jv_check(std::vector<std::vector<std::string>>& device_data)
{
    const auto pce_range = Config::get_config_values_vec<size_t>("PCE_range");
    const auto ff_range =  Config::get_config_values_vec<size_t>("FF_range");
    const auto isc_range = Config::get_config_values_vec<size_t>("Isc_range");
    const auto voc_range = Config::get_config_values_vec<size_t>("Voc_range");
    const auto jv_col = Config::get_config_values_vec<size_t>("JV_col");

    auto statistic_data = [](const std::vector<std::string>::iterator &begin, const std::vector<std::string>::iterator& end) -> size_t
    {
        auto it = begin;
        size_t cnt = 0;
        while (it != end)
        {
            if(it->compare("") == 0)
                cnt+=1;
            it++;
        }
        return cnt;
    };


    auto is_valid_data = [](const size_t cnt, const std::vector<std::string>::iterator & data_begin, const std::vector<std::string>::iterator &data_end,
            const std::vector<std::string>::iterator &ref_data_begin, const std::vector<std::string>::iterator &ref_data_end) -> bool
    {
        // 修正JV数据
        if(cnt == 4)
        {
            // 4项数据完整
        }
        else if (cnt == 3)
        {
            // 3项数据完整
        }
        else
        {
            // 其他情况：使用平均值补全，再进行检查，如能通过检查，则仍然为合理数据
        }
    };

    auto global_copy = [](const bool first, const bool second, std::vector<std::string>::iterator data_begin) -> void
    {
        auto first_begin = data_begin;
        auto second_begin = data_begin;
        if(first && second)
            return;
        else if (first)
        {
            first_begin = data_begin;
            second_begin = data_begin+4;
        }
        else
        {
            first_begin = data_begin;
            second_begin = data_begin+4;

        }
        for(size_t i = 0; i < 4; i+=1)
            *first_begin = *second_begin;
    };

    auto check_rational = [](const std::vector<std::string>::iterator& begin) -> bool
    {
        auto first_begin = begin;
        auto second_begin = begin+4;
        if(stod(*first_begin)* stod(*(first_begin+1))*stod(*(first_begin+2)) == stod(*(first_begin+3)))
            ;

        return true;
    };

    auto fixed_device_data = device_data;
    std::vector<std::vector<std::string>> return_results = std::remove_reference<decltype(device_data)>::type();
    return_results.reserve(device_data.size());
    size_t count = 0;
    bool is_reserved = false;
    bool first_is_lost = false;
    bool second_is_lost = false;
    for(size_t i = 0; i < device_data.size(); i+=1)
    {
        auto line = device_data[i];
        auto ref_line = fixed_device_data[i];
        size_t first_cnt = statistic_data(line.begin(), line.begin()+4);
        size_t second_cnt = statistic_data(line.begin()+4, line.end());
        // 修正JV数据
        bool is_valid_first = is_valid_data(first_cnt, line.begin(), line.begin()+4, ref_line.begin() , ref_line.begin()+4);
        bool is_valid_second = is_valid_data(second_cnt, line.begin()+4, line.end(), ref_line.begin()+4 , ref_line.end());
        if(!is_valid_first && !is_valid_second)
            continue;
        global_copy(is_valid_first, is_valid_second, line.begin());
        // 四项范围检查
        // JV合理性检查
        bool first_ratio = check_rational(line.begin());
        bool second_ratio = check_rational(line.begin()+4);

        if(first_ratio || second_ratio)
        {
            return_results.emplace_back(line);
            count+=1;
        }

    }
    return_results.reserve(count);
}



void preprocess_device_data(std::vector<std::vector<std::string>> &device_data)
{
    // 转置
    device_data = new_transpose_matrix(device_data);
    // 删除最后一列
    device_data.assign(device_data.begin(), device_data.end() - 1);
    device_data = new_transpose_matrix(device_data);
}

// 统一device数据格式，变为适应输入的数据格式
void preprocess_mole_device_data(std::vector<std::vector<std::string>> &mole_data, std::vector<std::vector<std::string>> &device_data)
{

    // 删除标题列
    device_data.assign(device_data.begin() + 1, device_data.end());
    for (size_t i = 0; i < device_data.size(); i += 1)
    {
        // 插入SAMs位置
        mole_data[i].emplace_back(device_data[i][2]);
        // 移除元素：前三行和最后一行
        device_data[i].assign(device_data[i].begin() + 3, device_data[i].end() - 1);
    }
}

/**
 * 2024年7月 重构新处理分子指纹方法
 */
auto build_mole_onehot(std::vector<std::vector<std::string>> &mole_data) -> std::vector<std::vector<std::string>>
{
    // 设备参数
    size_t mole_size = 0;
    size_t max_size = 0;
    for (size_t i = 0; i < mole_data.size(); i += 1)
        if (mole_data[i].size() == 2)
        {
            max_size = mole_data[i][1].size();
            break;
        }
    // 补全空值
    for (size_t i = 0; i < mole_data.size(); i += 1)
        if (mole_data[i].size() == 1)
            mole_data[i].emplace_back(std::string(max_size, '0'));
    std::vector<std::vector<std::string>> mole_info(mole_data.size(), std::vector<std::string>(max_size));
    // 获取分子计算信息
    mole_info.reserve(mole_data.size());
    for (size_t i = 0; i < mole_data.size(); i += 1)
        mole_info[i] = split_string_ch(mole_data[i][0], '_');
    std::vector<std::vector<double>> results(mole_data.size()); // 最终信息
    std::vector<int> results_article(mole_data.size());
    // 开始计算
    size_t ptr = 0;
    results[0] = split_one_hot_to_vector<double>(mole_data[0][1]);
    results_article[0] = stod(mole_info[0][0]);
    for (size_t i = 1; i < mole_data.size(); i += 1)
        // 内部更新
        if (mole_info[i][0].compare(mole_info[i - 1][0]) == 0)
            results[ptr] = vector_add(results[ptr], vector_multiply(split_one_hot_to_vector<double>(mole_data[i][1]), stod(mole_info[i][2]) * 0.01));
        else // 下一个
        {
            // 初始化下一个组
            ptr += 1;
            results_article[ptr] = stod(mole_info[i][0]);
            results[ptr] = split_one_hot_to_vector<double>(mole_data[i][1]);
            if (mole_info[i].size() == 3)
                results[ptr] = vector_multiply(results[ptr], stod(mole_info[i][2]) * 0.01);
        }
    results.resize(ptr + 1);
    for (size_t i = 0; i < results.size(); i += 1)
        results[i].insert(results[i].begin(), results_article[i]);
    // 比较合并后的分子和文章数量是否对的上
    return convert_vector_to_string(results);
}

// 筛选分子指纹
void precheck_mole_data(std::vector<std::vector<std::string>> &mole_onehot_data, std::vector<std::vector<std::string>> &device_data)
{
    // 合并数据
    std::vector<std::vector<std::string>> merge_data(device_data.begin(), device_data.end());
    std::vector<std::string> tmp;
    for (size_t i = 0; i < merge_data.size(); i += 1)
        merge_data[i].insert(merge_data[i].end(), mole_onehot_data[i].begin() + 1, mole_onehot_data[i].end());
    // 通过JV方式限制范围，过滤数据
    decltype(merge_data) filted_data;
    filted_data.reserve(merge_data.size());
    auto JV_col_data = Config::get_config_values_vec<size_t>("JV_col");
    auto PCE_range = Config::get_config_values_vec<double>("PCE_range");
    auto FF_range = Config::get_config_values_vec<double>("FF_range");
    auto Isc_range = Config::get_config_values_vec<double>("Isc_range");
    auto Voc_range = Config::get_config_values_vec<double>("Voc_range");
    std::vector<double> JV_avg_data(JV_col_data.size() / 2);
    for (size_t i = 0; i < merge_data.size(); i += 1)
    {
        for (size_t i = 0; i < JV_avg_data.size(); i += 1)
            JV_avg_data[i] = (stod(merge_data[i][JV_col_data[i]]) + stod(merge_data[i][JV_col_data[i] + JV_avg_data.size()])) / 2;
        if (JV_avg_data[0] > Voc_range[0] && JV_avg_data[0] < Voc_range[1] && JV_avg_data[1] > Isc_range[0] && JV_avg_data[1] < Isc_range[1] && JV_avg_data[2] > FF_range[0] &&
                JV_avg_data[2] < FF_range[1] && JV_avg_data[3] > PCE_range[0] && JV_avg_data[3] < PCE_range[1])
            filted_data.emplace_back(merge_data[i]);
    }
    mole_onehot_data.resize(filted_data.size());
    device_data.resize(filted_data.size());
    // 重新写入数据
    for (size_t i = 0; i < filted_data.size(); i += 1)
    {
        device_data[i].assign(filted_data[i].begin(), filted_data[i].begin() + 64);
        mole_onehot_data[i].assign(filted_data[i].begin() + 65, filted_data[i].end());
    }
}

// device_data总数据 which_block: 当前修正的是超参数sequence_col中的哪个变量
void fix_multi_sequence(std::vector<std::vector<std::string>> &device_data, const std::vector<size_t> &num_vec)
{
    const static std::vector<std::string> delimiter = {"|", ">>"};
    // 首先修正空缺值和非法数字
    // 通用修正方法：遍历修正
    // 只修正有序列关系的
    std::vector<size_t> length_each_col(num_vec.size());
    std::vector<std::vector<std::string>> splited_data(num_vec.size());
    // 主函数遍历：遍历每一个数据
    for (size_t i = 0; i < device_data[0].size(); i += 1)
    {
        // 列遍历
        // 找到最大值
        for (size_t j = 0; j < num_vec.size(); j += 1)
        {
            splited_data[j] = manual_split(device_data[num_vec[j]][i], delimiter);
            length_each_col[j] = splited_data[j].size();
        }
        auto max_num = *std::max_element(length_each_col.begin(), length_each_col.end());
        // 重新遍历，直到找到补齐到最大值，补齐的方式，是将原先的最后一项序列项复制
        for (size_t j = 0; j < num_vec.size(); j += 1)
        {
            // 查询当前行长度，如果不到最长，则进行补全
            if (max_num == splited_data[j].size())
                continue;
            // 补全完成后，写入原始数据中（device_data）
            for (size_t cnt = splited_data[j].size(); cnt < max_num; cnt += 1)
                splited_data[j].emplace_back(splited_data[j].back());
            // 插入分隔符
            splited_data[j] = insert_separator_to_vec(splited_data[j], ">>");
            device_data[num_vec[j]][i] = vector_to_string(splited_data[j]);
        }
    }
}

// 修正缺失数值
void preprocess(std::vector<std::vector<std::string>> &device_data)
{
    const static auto num_col = Config::get_config_values_vec<size_t>("num_col");
    const static auto year_col = Config::get_config_value<size_t>("years_col");
    const static auto sequence_col = Config::get_config_matrix<size_t>("sequence_block");
    auto is_in_sequence = [](size_t num) -> int {
        for (size_t i = 0; i < sequence_col.size(); i += 1)
            if (std::find(sequence_col[i].begin(), sequence_col[i].end(), num) != sequence_col[i].end())
                return static_cast<int>(i);
        return -1;
    };

    // 转置为列，便于进行遍历
    device_data = transpose_matrix(device_data);
    // 非法值，空值修正
    int which_sequence_num;
    for (size_t i = 0; i < device_data.size(); i += 1)
    {
        // 进行空值非法值修正，只有两种种类的数据串：数字和字符串
        if (std::find(num_col.begin(), num_col.end(), i) != num_col.end())
        {
            // 年份首先修正为时间戳，再作为数字进行处理
            if (i == year_col)
                device_data[i] = convert_to_days_since_custom(device_data[i], "2009/1/1");
            // 其他按一般数字修正
            device_data[i] = replace_invalid_values(device_data[i]);
        }
        else
        {
            // 修正单行字符串
            replace_empty_values_with_count(device_data[i]);
        }
        // sequence值单独修正：缺失值补全
        which_sequence_num = is_in_sequence(i); // 判断为是否需要特殊处理的序列
        if (which_sequence_num != -1 && which_sequence_num != 0)
        {
            replace_empty_values_with_count(device_data[i]);
            if (i == sequence_col[which_sequence_num][0]) // 一个block只修正一次
                fix_multi_sequence(device_data, sequence_col[which_sequence_num]);
        }
    }
    // 转置回来
    device_data = transpose_matrix(device_data);
}

// 构建map不需要特殊技巧，全部进行符号拆分即可，只需要分清数字和字符串
auto build_datas(std::vector<std::vector<std::string>> &device_data)
{
    const static std::vector<size_t> maps_constrained_size = Config::get_config_values_vec<size_t>("maps_size");
    const static std::vector<int> onehot_padding_size = Config::get_config_values_vec<int>("onehot_size");
    const static std::vector<size_t> onehot_block_size = Config::get_config_values_vec<size_t>("block_size");
    const static std::vector<size_t> num_col = Config::get_config_values_vec<size_t>("num_col"); // 数字数据所在列
    const static std::vector<std::vector<size_t>> sequence_col = Config::get_config_matrix<size_t>("sequence_block");

    // 找到了匹配的层，则返回层数，否则返回sequence_col尺寸
    auto which_seq_layer = [](const size_t num) -> size_t {
        for (size_t i = 0; i < sequence_col.size(); i += 1)
            if (std::find(sequence_col[i].begin(), sequence_col[i].end(), num) != sequence_col[i].end())
                return i;
        return sequence_col.size();
    };

    // 单独处理的 tl层，两个都要叠加在一起
    auto build_tl_data = [&](size_t col_1, size_t col_2) -> std::vector<std::string> {
        std::vector<std::string> result;
        result.reserve(device_data[col_1].size() + device_data[col_2].size()); // 先分配内存，性能会略微高一些
        result.insert(result.end(), device_data[col_1].begin(), device_data[col_1].end());
        result.insert(result.end(), device_data[col_2].begin(), device_data[col_2].end());
        return result;
    };

    // 如果希望使用已经存在的maps，则跳过maps的构建
    if (Config::get_config_value<size_t>("using_exist_maps") == 0)
    {
        // 构建map（针对字符串)
        for (size_t i = 0; i < device_data.size(); i += 1)
        {
            std::string map_file_path = Config::get_config_value<std::string>("map_file_root") + "/map_" + std::to_string(i) + ".txt";
            std::cout << i << std::endl;
            size_t which_tl = which_seq_layer(i);
            // 处理tl
            if (which_tl == 1)
            {
                // 11 12
                if (i == sequence_col[which_tl][0])
                {
                    std::vector<std::string> col_data = build_tl_data(sequence_col[which_tl][0], sequence_col[4][0]);
                    build_maps(col_data, map_file_path, maps_constrained_size[i]);
                    // 复制一份给39
                    std::string another_tl_maps = Config::get_config_value<std::string>("map_file_root") + "/map_" + std::to_string(sequence_col[4][0]) + ".txt";
                    copy_file(map_file_path, another_tl_maps);
                    // 清空内存占用
                    decltype(col_data)().swap(col_data);
                }
                else if (i == sequence_col[which_tl][1])
                {
                    std::vector<std::string> col_data = build_tl_data(sequence_col[which_tl][1], sequence_col[4][1]);
                    build_maps(col_data, map_file_path, maps_constrained_size[i]);
                    // 复制一份给41
                    std::string another_tl_maps = Config::get_config_value<std::string>("map_file_root") + "/map_" + std::to_string(sequence_col[4][1]) + ".txt";
                    copy_file(map_file_path, another_tl_maps);
                    // 清空内存占用
                    decltype(col_data)().swap(col_data);
                }
            }
            else if (which_tl == 4)
                // 39 41 跳过
                continue;
            else if (i == Config::get_config_value<size_t>("expression_col"))
                // 23列，钙钛矿成分：单独构建
                continue;
            // 处理非tl的字符串数据
            else
                // 字符串
                if (std::find(num_col.begin(), num_col.end(), i) == num_col.end())
                    build_maps(device_data[i], map_file_path, maps_constrained_size[i]);
        }
    }
    else
        std::cout << "Skip maps build\n";
    std::cout << "Maps build successed" << std::endl;
    //------------------------------------------------------
    std::string onehot_dir = Config::get_config_value<std::string>("onehot_dir_root");
    // 构建encode
    std::vector<std::vector<std::vector<float>>> one_hot_data(device_data.size());
    for (size_t i = 0; i < device_data.size(); i += 1)
    {
        if (i != Config::get_config_value<size_t>("full_structure_col"))
            std::cout << "Onehot info : " << i;

        if (i == Config::get_config_value<size_t>("expression_col"))
        {
            // 构建23号预处理数据
            std::string map_file_path = Config::get_config_value<std::string>("map_file_root") + "/map_" + std::to_string(i) + ".txt";
            auto results = build_23_raw_results(Config::get_config_value<std::string>("col_23_config_file_path"), device_data[i]);
            // 构建23号maps和 onehot
            one_hot_data[i] = build_23_maps_and_encode(results, onehot_padding_size[i], map_file_path, Config::get_config_value<size_t>("using_exist_maps") == 0);
            decltype(results)().swap(results); // 内存清空
        }
        else if (std::find(num_col.begin(), num_col.end(), i) != num_col.end())
        {
            // 处理数字字符串
            one_hot_data[i] = convert_2D_string_array<float>(build_num_onehot(device_data[i], 10, 3));
        }
        else if (i == Config::get_config_value<size_t>("full_structure_col"))
            // 在构建时，跳过第三列，使得其为空串
            one_hot_data[i] = std::vector<std::vector<float>>();
        else
        {
            // 处理普通字符串
            std::string map_path = Config::get_config_value<std::string>("map_file_root") + "/map_" + std::to_string(i) + ".txt";
            one_hot_data[i] = convert_2D_vector<int, float>(build_str_onehot(device_data[i], map_path, onehot_padding_size[i]));
            // if(i == 65)
            // write_content_to_file(device_data[i], PROJECT_PATH "/65_tmp.txt");
        }
        std::string write_file_path = onehot_dir + "/onehot_data_" + std::to_string(i) + ".txt";
        write_content_to_file(one_hot_data[i], write_file_path);

        if (i != Config::get_config_value<size_t>("full_structure_col"))
            std::cout << ", total features:\t\t" << one_hot_data[i][0].size() << "\n";

        std::remove_reference<decltype(device_data[i])>::type().swap(device_data[i]); // 删除原始数据，减轻内存占用
    }
    std::cout << "Onehot build successed" << std::endl;
    //------------------------------------------------------------
    // 数据重组，分为输入数据，目标数据和用于SAMs模型的数据
    // 因为one_hot_data是一个三维数组，我们需要将其中的元素依次取出，重构成为一个二维数组
    std::vector<size_t> jv_col = Config::get_config_values_vec<size_t>("JV_col");      // JV数据所在列
    std::vector<size_t> sams_col = {one_hot_data.size() - 2, one_hot_data.size() - 1}; // SAMs数据所在列
    // 寻找特征列的第一列数据
    size_t first_input_eigen_col = 0;
    for (size_t i = 0; i < one_hot_data.size(); i += 1)
        if (std::find(jv_col.begin(), jv_col.end(), i) == jv_col.end())
        {
            first_input_eigen_col = i;
            break;
        }
    decltype(one_hot_data) results;                         // 返回数据集合
    auto input_eigen = one_hot_data[first_input_eigen_col]; // 输入数据
    auto target_data = one_hot_data[jv_col[0]];             // 目标数据
    auto sams_adjacent_data = one_hot_data[sams_col[0]];    // SAMs数据
    std::vector<size_t> onehot_size;
    std::vector<size_t> onehot_size_sams;
    onehot_size.push_back(input_eigen[0].size());
    onehot_size_sams.push_back(sams_adjacent_data[0].size());
    for (size_t i = 0; i < one_hot_data.size(); i += 1)
    {

        if (std::find(jv_col.begin(), jv_col.end(), i) != jv_col.end())
        {
            if (i != jv_col[0])
                target_data = concatenate_matrices(target_data, one_hot_data[i], 1);
        }
        else if (std::find(sams_col.begin(), sams_col.end(), i) != sams_col.end())
        {
            if (i != sams_col[0])
            {
                sams_adjacent_data = concatenate_matrices(sams_adjacent_data, one_hot_data[i], 1);
                onehot_size_sams.push_back(one_hot_data[i][0].size());
            }
        }
        else
        {
            // 跳过第一列，描述整体结构的列（第三列）
            if (i != first_input_eigen_col && i != Config::get_config_value<size_t>("full_structure_col"))
            {
                input_eigen = concatenate_matrices(input_eigen, one_hot_data[i], 1);
                onehot_size.push_back(one_hot_data[i][0].size());
            }
        }
        std::remove_reference<decltype(one_hot_data[i])>::type().swap(one_hot_data[i]);
    }
    std::cout << "Data reconsructed successed" << std::endl;
    write_content_to_file(onehot_size, Config::get_config_value<std::string>("output_dir") + "/input_onehot_size.txt");
    write_content_to_file(onehot_size_sams, Config::get_config_value<std::string>("output_dir") + "/input_sams_onehot_size.txt");
    results.emplace_back(std::move(input_eigen));
    results.emplace_back(std::move(target_data));
    results.emplace_back(std::move(sams_adjacent_data));
    // 返回完整的重构数据
    return results;
}

// 将靠近钙钛矿的一层传输层单独取出，统一为单词TL
void reconstruct_input_data(std::vector<std::vector<std::string>> &device_data)
{
    const static std::vector<std::string> delimiter_sets = {"|", ">>"};
    device_data = transpose_matrix(device_data);
    // 两个传输层的编号，两个电极的编号
    size_t t_tl_layer_num = Config::get_config_value<size_t>("tl_layer_transparent");
    size_t m_tl_layer_num = Config::get_config_value<size_t>("tl_layer_metal_Pole");
    size_t t_p_layer_num = Config::get_config_value<size_t>("transparent_pole_layer");
    size_t m_p_layer_num = Config::get_config_value<size_t>("metal_pole_layer");

    // 两个传输层
    auto t_tl_layer = device_data[t_tl_layer_num];
    auto m_tl_layer = device_data[m_tl_layer_num];
    auto t_p_layer = device_data[t_p_layer_num];
    auto m_p_layer = device_data[m_p_layer_num];
    // 用于SAMs输入的新传输层
    auto new_t_tl_layer = decltype(t_tl_layer)(t_tl_layer.size());
    auto new_m_tl_layer = decltype(t_tl_layer)(t_tl_layer.size());
    // 去除透明电极传输层中的相关参数
    for (size_t i = 0; i < t_tl_layer.size(); i += 1)
    {
        auto split_data = split_str_multi_sep(t_tl_layer[i], delimiter_sets);
        if (split_data.size() == 1)
            // 从金属电极里拿信息
            new_t_tl_layer[i] = split_str_multi_sep(t_p_layer[i], delimiter_sets).back();
        else
            new_t_tl_layer[i] = split_data[split_data.size() - 2];
        trim(new_t_tl_layer[i]);
        split_data.back() = "TL_Materials";
        t_tl_layer[i] = "";
        // 重构输入项
        for (size_t j = 0; j < split_data.size(); j += 1)
        {
            t_tl_layer[i] += split_data[j];
            if (j != split_data.size() - 1)
                t_tl_layer[i] += "|";
        }
    }
    // 去除电极传输层中的相关参数
    for (size_t i = 0; i < m_tl_layer.size(); i += 1)
    {
        auto split_data = split_str_multi_sep(m_tl_layer[i], delimiter_sets); 
        if (split_data.size() == 1)
            // 从金属电极里拿信息
            new_m_tl_layer[i] = split_str_multi_sep(m_p_layer[i], delimiter_sets)[0];
        else
            new_m_tl_layer[i] = split_data[1];
        trim(new_m_tl_layer[i]);
        split_data[0] = "TL_Materials";
        m_tl_layer[i] = "";
        // 重构输入项
        for (size_t j = 0; j < split_data.size(); j += 1)
        {
            m_tl_layer[i] += split_data[j];
            if (j != split_data.size() - 1)
                m_tl_layer[i] += "|";
        }
        // std::cout << new_m_tl_layer[i] << std::endl;
    }
    // 将处理过的数据写回原始数据中
    device_data[t_tl_layer_num] = t_tl_layer;
    device_data[m_tl_layer_num] = m_tl_layer;
    // 将最后两行也添加到原始数组中
    device_data.emplace_back(new_t_tl_layer);
    device_data.emplace_back(new_m_tl_layer);
    // 矩阵转置回来
    device_data = transpose_matrix(device_data);
}


// 判断配置文件路径
std::string get_config_file_path(int argc,  char * argv[])
{
    if (argc > 1)
    {
        std::string config_file_path_str = std::string(argv[1]);
        if(std::filesystem::exists(config_file_path_str))
            return config_file_path_str;
    }
    return std::string("config.txt");
}

// 配置文件位置（核心初始化参数）
std::string config_file_path = PROJECT_PATH "/config.txt";

int main(int argc, char *argv[])
{
    clock_t start = clock(); // 计时器
    auto file_path = get_config_file_path(argc, argv);    // 配置文件默认与可执行文件放在一起
    file_path = config_file_path;
    Config::initialize_config(config_file_path);
    std::vector<std::vector<std::string>> available_contents; // 进入主循环的数据
    std::cout << "Initializing data..." << std::endl;
    // 读取和预处理数据
    if (Config::get_config_value<size_t>("device_data") == 0) // 判断写入器件信息还是分子信息（初始化输入数据）
    {
        // A. 分子指纹信息
        available_contents = read_2d_datas<std::string>(Config::get_config_value<std::string>("mole_device_data_path"), "\t");
        std::vector<std::vector<std::string>> mole_data = read_2d_datas<std::string>(Config::get_config_value<std::string>("mole_onehot_data_path"));
        // 处理分子信息
        std::vector<std::vector<std::string>> mole_onehot_result = build_mole_onehot(mole_data);
        // 必须满足：分子指纹与device数据能一条一条对上
        assert(available_contents.size() - 1 == mole_onehot_result.size());
        // 预处理数据
        preprocess_mole_device_data(mole_onehot_result, available_contents);
        preprocess(available_contents);
        // 筛选，并且限制JV数据范围
        precheck_mole_data(mole_onehot_result, available_contents);
        // 写入分子指纹
        write_content_to_file(mole_onehot_result, Config::get_config_value<std::string>("output_dir") + "/molecule_fingerprinter.txt");
    }
    else if (Config::get_config_value<size_t>("device_data") == 1)
    {
        // B. 器件信息
        available_contents = read_2d_datas<std::string>(Config::get_config_value<std::string>("device_data_path"), "\t");
        // 压缩信息比例（乱序压缩）
        available_contents.assign(available_contents.begin(),
                available_contents.begin() + (int)(available_contents.size() * Config::get_config_value<double>("device_data_reserve_ratio")));
        // 删除标题
        available_contents.assign(available_contents.begin() + 1, available_contents.end());
        // 预处理数据
        preprocess_device_data(available_contents);
        // 限制JV范围
        // device_jv_check(available_contents);
        // 修正缺失值
        preprocess(available_contents);
    }
    else
        throw std::runtime_error("Device_data invalid value: " + Config::get_config_value<std::string>("device_data") + "(Only support 0 or 1)");
    // 重构矩阵，将用于SAMs模型输入的两列分离出来
    reconstruct_input_data(available_contents);
    // write_content_to_file(available_contents, PROJECT_PATH "/tmp.txt");
    // exit(0);
    std::cout << "Initialized finished" << std::endl;
    // 转置，便于后续处理 进入主循环，构建map和编码
    available_contents = transpose_matrix(available_contents);
    // 构建，获取onehot结果
    auto onehot_result = build_datas(available_contents);
    // 完成数据处理，写入onehot数据
    std::cout << "Writting data to files..." << std::endl;
    Timer write_timer;
    write_timer.start();
    write_content_to_file(onehot_result[0], Config::get_config_value<std::string>("output_dir") + "/eigen_onehot_data.txt");
    std::remove_reference<decltype(onehot_result[0])>::type().swap(onehot_result[0]);
    write_content_to_file(onehot_result[1], Config::get_config_value<std::string>("output_dir") + "/output_onehot_data.txt");
    std::remove_reference<decltype(onehot_result[1])>::type().swap(onehot_result[1]);
    write_content_to_file(onehot_result[2], Config::get_config_value<std::string>("output_dir") + "/sams_onehot_data.txt");
    std::remove_reference<decltype(onehot_result[2])>::type().swap(onehot_result[2]);
    std::cout << "Write data time: " << write_timer.stop() << "s\n";
    std::cout << "Build finished, time cost : " << static_cast<double>(clock() - start) / CLOCKS_PER_SEC * 1000 << " ms " << "(" << (double)(clock() - start) / CLOCKS_PER_SEC
        << " s)" << std::endl; // 计时器结束
    return EXIT_SUCCESS;
}

