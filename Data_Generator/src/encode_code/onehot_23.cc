#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <sstream>

// 提取唯一字符串及其出现次数
std::unordered_map<std::string, std::size_t> extract_string_counts(const std::vector<std::vector<std::string>>& data_lists) noexcept {
    std::unordered_map<std::string, std::size_t> string_counts;
    string_counts.reserve(data_lists.size()/2);
    for (const auto& data : data_lists) 
        for (std::size_t i = 0; i < data.size(); i += 2) 
            string_counts[data[i]]++;
    return string_counts;
}

// 创建字符串到索引的映射，保留高频字符串，低频字符串归类为“other”
std::unordered_map<std::string, std::size_t> create_string_to_index_mapping(const std::unordered_map<std::string, std::size_t>& string_counts, std::size_t max_features) noexcept {
    std::vector<std::pair<std::string, std::size_t>> sorted_counts(string_counts.begin(), string_counts.end());
    std::sort(sorted_counts.begin(), sorted_counts.end(), [](const auto& a, const auto& b) {
            return b.second > a.second; // 按照出现次数降序排列
            });

    std::unordered_map<std::string, std::size_t> string_to_index;
    string_to_index.reserve(sorted_counts.size());
    std::size_t index = 0;
    for (std::size_t i = 0; i < sorted_counts.size() && index < max_features; ++i) {
        string_to_index[sorted_counts[i].first] = index++;
    }
    string_to_index["other"] = index;

    return string_to_index;
}

// 保存映射表到文件
void save_string_to_index_mapping(const std::unordered_map<std::string, std::size_t>& string_to_index, const std::string& filepath) noexcept {
    std::ofstream ofs(filepath);
    if (ofs.is_open()) {
        for (const auto& pair : string_to_index) {
            ofs << pair.first << "," << pair.second << "\n";
        }
        ofs.close();
    } else {
        std::cerr << "failed to open file for writing: " << filepath << std::endl;
    }
}

// 从文件中读取映射表
std::unordered_map<std::string, std::size_t> load_string_to_index_mapping(const std::string& filepath) noexcept {
    std::unordered_map<std::string, std::size_t> string_to_index;
    std::ifstream ifs(filepath);
    if (ifs.is_open()) {
        std::string line;
        while (std::getline(ifs, line)) {
            std::istringstream iss(line);
            std::string key;
            std::size_t value;
            if (std::getline(iss, key, ',') && iss >> value) {
                string_to_index[key] = value;
            }
        }
        ifs.close();
    } else {
        std::cerr << "failed to open file for reading: " << filepath << std::endl;
    }
    return string_to_index;
}

// 填充one-hot编码矩阵
void fill_one_hot_matrix(std::vector<std::vector<float>>& one_hot_matrix,
        const std::vector<std::vector<std::string>>& data_lists,
        const std::unordered_map<std::string, std::size_t>& string_to_index) noexcept 
{
    std::size_t other_index = string_to_index.at("other");
    for (size_t i = 0; i < data_lists.size(); ++i) 
    {
        const auto& data = data_lists[i];
        for (std::size_t j = 0; j < data.size(); j += 2)
        {
            const std::string& str = data[j];
            double weight = std::stod(data[j + 1]);
            auto it = string_to_index.find(str);
            if (it != string_to_index.end()) 
            {
                one_hot_matrix[i][it->second] += weight;
            } else 
            {
                one_hot_matrix[i][other_index] += weight;
            }
        }
    }
}

// 进行one-hot编码处理
std::vector<std::vector<float>> build_23_maps_and_encode(const std::vector<std::vector<std::string>>& data_lists, std::size_t max_features, const std::string& filepath, bool create_new_mapping)
{
    std::unordered_map<std::string, std::size_t> string_to_index;

    if (create_new_mapping) 
    {
        std::cout << "Cols of PVSK cols: create new maps\n";
        // 提取字符串出现次数
        std::unordered_map<std::string, std::size_t> string_counts = extract_string_counts(data_lists);
        // 创建字符串到索引的映射
        string_to_index = create_string_to_index_mapping(string_counts, max_features);
        // 保存映射表
        save_string_to_index_mapping(string_to_index, filepath);
    } else 
    {
        // 从文件中读取映射表
        std::cout << "Cols of PVSK cols: using existed maps\n";
        string_to_index = load_string_to_index_mapping(filepath);
    }

    // 初始化one-hot编码矩阵
    std::size_t num_samples = data_lists.size();
    std::size_t num_features = string_to_index.size();
    std::vector<std::vector<float>> one_hot_matrix(num_samples, std::vector<float>(num_features, 0));

    // 填充one-hot编码矩阵
    fill_one_hot_matrix(one_hot_matrix, data_lists, string_to_index);

    // 返回onehot结果
    return one_hot_matrix;
}

