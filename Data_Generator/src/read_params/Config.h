#ifndef CONFIG_H
#define CONFIG_H

#include <map>
#include <string>
#include <vector>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../utils/Utils_2.h"

namespace Config
{
    extern const std::map<std::string, std::vector<std::string>> *config;

    // 转为任意类型
    template <typename T>
        inline T convert(const std::string &value)
        {
            std::istringstream iss(value);
            T result;
            if (!(iss >> result))
                throw std::runtime_error("Conversion failed for value: " + value);
            return result;
        }

    void initialize_config(const std::string &filename);

    template <typename T>
        T get_config_value(const std::string &key, size_t index = 0);

    template <typename T>
        std::vector<T> get_config_values(const std::string &key);

    template <typename T>
        std::vector<std::vector<T>> get_config_matrix(const std::string &key);

    template <typename T>
        T get_config_value(const std::string &key, size_t index)
        {
            if (config->find(key) != config->end())
            {
                const auto &values = config->at(key);
                if (index < values.size())
                    return convert<T>(values[index]);
                else
                    throw std::runtime_error("Index out of range for key: " + key);
            }
            else
                throw std::runtime_error("Key not found: " + key);
        }

    template <typename T>
        std::vector<T> get_config_values(const std::string &key)
        {
            if (config->find(key) != config->end())
            {
                const auto &values = config->at(key);
                std::vector<T> result;
                for (const auto &val : values)
                    result.emplace_back(convert<T>(val));
                return result;
            }
            else
            {
                throw std::runtime_error("Key not found: " + key);
            }
        }

    // 可返回多值函数所有变量
    template <typename T>
        std::vector<T> get_config_values_vec(const std::string &key)
        {
            if (config->find(key) != config->end())
            {
                const auto &values = config->at(key);
                std::vector<T> results(values.size());
                size_t ptr = 0;
                while (ptr < values.size())
                {
                    results[ptr] = convert<T>(values[ptr]);
                    ptr += 1;
                }
                return results;
            }
            else
                throw std::runtime_error("Key not found: " + key);
        }

    // 读取二维数组
    template <typename T>
        std::vector<std::vector<T>> get_config_matrix(const std::string &key)
        {
            static auto reconstruct_str = [](const std::vector<std::string> &values) -> std::string {
                auto merged_values = insert_separator_to_vec(values, ",");
                std::string res;
                for (auto ele : merged_values)
                    res += ele;
                return res;
            };

            // 去掉字符串中的空格
            static auto remove_spaces = [](const std::string &input) -> std::string {
                std::string output;
                std::remove_copy_if(input.begin(), input.end(), std::back_inserter(output), [](char c) { return std::isspace(c); });
                return output;
            };

            if (config->find(key) != config->end())
            {
                const auto &values = config->at(key);
                // 重组数组项
                std::string remerged_str = reconstruct_str(values);
                // 去除空格，最外侧中括号
                remerged_str = remove_spaces(remerged_str).substr(1, remerged_str.size() - 2);
                std::vector<std::vector<T>> result;
                std::stringstream ss(remerged_str);
                std::string segment;
                while (std::getline(ss, segment, ']'))
                {
                    if (segment.empty() || segment == ",")
                        continue;
                    if (segment[0] == ',')
                        segment = segment.substr(1); // 去掉前面的逗号

                    std::stringstream rowStream(segment.substr(1)); // 去掉前面的中括号
                    std::vector<T> row;
                    std::string item;

                    while (std::getline(rowStream, item, ','))
                        // row.emplace_back(convert_specific_type<T>(item));
                        row.emplace_back(convert<T>(item));

                    result.emplace_back(row);
                }
                return result;
            }
            else
                throw std::runtime_error("Key not found: " + key);
        }
} // namespace Config
#endif // CONFIG_H
