#include "mapping_code.h"
#include "../utils/Utils_2.h"
#include <cstddef>
#include <iostream>
#include <stdexcept>

// Function to flatten a list of lists potential FATAL : total size of results exceed size_t
std::vector<std::string> flatten_list(const std::vector<std::vector<std::string>> &words_list) {
    size_t total_size = std::accumulate(words_list.begin(), words_list.end(), 0,
            [](size_t sum, const std::vector<std::string>& sublist) {
            return sum + sublist.size();
            });
    std::vector<std::string> result;
    result.reserve(total_size);
    for (const auto &sublist : words_list)
        result.insert(result.end(), sublist.begin(), sublist.end());
    return result;
}

// Function to build a word frequency mapping
std::unordered_map<std::string, int> build_mapping(const std::vector<std::string> &words, int max_onehot) {
    std::unordered_map<std::string, int> word_count;
    word_count.reserve(words.size());
    for (const auto &word : words)
        word_count[word]++;

    std::vector<std::pair<std::string, int>> sorted_word_count(word_count.begin(), word_count.end());
    std::sort(sorted_word_count.begin(), sorted_word_count.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });

    std::unordered_map<std::string, int> mapping;
    mapping.reserve(max_onehot+1);
    int i = 0;
    for (const auto &pair : sorted_word_count) {
        if (i >= max_onehot) break;
        mapping[pair.first] = i++;
    }
    mapping["other"] = i;
    return mapping;
}

// Function to save mapping to a file
void save_mapping(const std::unordered_map<std::string, int> &mapping, const std::string &file_path) {
    std::ofstream file(file_path);
    for (const auto &pair : mapping)
        file << pair.first << "," << pair.second << "\n";
    file.close();
}

// Function to load mapping from a file
std::unordered_map<std::string, int> load_mapping(const std::string &file_path) {
    std::unordered_map<std::string, int> mapping;
    std::ifstream file(file_path);
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string word;
        int index;
        std::getline(ss, word, ',');
        ss >> index;
        mapping[word] = index;
    }
    file.close();
    return mapping;
}

void build_maps(const std::vector<std::string>& input_strings, const std::string& mapping_path, int max_onehot)
{
    // Function to split and strip input strings
    auto split_and_strip = [](const std::vector<std::string> &input_strings) 
    {
        static const std::vector<std::string> delimiter = {"|", ">>"};
        std::vector<std::vector<std::string>> result(input_strings.size());
        for(size_t i = 0; i < input_strings.size(); i+=1)
            result[i] = split_str_multi_sep_trim(input_strings[i], delimiter);
        return result;
    };

    auto parallel_groups = split_and_strip(input_strings);
    auto all_words = flatten_list(parallel_groups);
    std::unordered_map<std::string, int> mapping = build_mapping(all_words, max_onehot);
    save_mapping(mapping, mapping_path);
}

// Function for One-Hot encoding, padding, and counting
std::vector<std::vector<int>> build_str_onehot(const std::vector<std::string> &input_strings, const std::string &map_path, const size_t constraint_block_size)
{
    // Function to handle sequence words
    auto split_2d_vec = [](const std::vector<std::string> &input_strings)
    {
        const static std::vector<std::string> delimiter = {"|", ">>"};
        std::vector<std::vector<std::string>> result(input_strings.size());
        for(size_t i = 0; i < input_strings.size(); i+=1)
            result[i] = split_str_multi_sep(input_strings[i], delimiter);
        return result;
    };

    // 用于映射的maps
    std::unordered_map<std::string, int> mapping = load_mapping(map_path);
    // 按block切分数据
    std::vector<std::vector<std::string>> splited_vec = split_2d_vec(input_strings);
    // 构建实际block尺寸
    size_t actual_block_size = constraint_block_size;
    if(constraint_block_size == 0)
        for(auto &line : splited_vec)
            if(line.size() > actual_block_size)
                actual_block_size = line.size();
    // 固定尺寸决定于 1. maps大小 2. 定义的最大block尺寸
    if(actual_block_size > std::numeric_limits<size_t>::max() / mapping.size())
        throw std::runtime_error("Num overflow in method build_str_onehot");
    std::vector<std::vector<int>> onehot_code(splited_vec.size(), std::vector<int>(actual_block_size * mapping.size()));
    std::vector<std::string> parallel_groups;
    int index;
    auto it = mapping.end();
    for (size_t i = 0; i < splited_vec.size(); i+=1)
    {
        // 限制最大block尺寸
        for (size_t j = 0; j < std::min(splited_vec[i].size(), actual_block_size); j+=1)
        {
            for (const auto ele : split_string_trim(splited_vec[i][j], ";"))
            {
                it = mapping.find(ele);
                if (it != mapping.end())
                    index = it->second;
                else
                    index = mapping.at("other");
                onehot_code[i][j * mapping.size() + index] += 1;
            }
        }
    }
    std::cout << ", blocks: " << actual_block_size << ", maps: " << mapping.size();
    return onehot_code;
}
