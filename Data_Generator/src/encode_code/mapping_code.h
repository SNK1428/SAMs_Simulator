#ifndef MAPPING_CODE_H
#define MAPPING_CODE_H

#include <iostream>
#include <vector>
#include <string>
#include <regex>
#include <unordered_map>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <sstream>
#include <cstddef>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <algorithm>
#include <limits>

// 构建maps
void build_maps(const std::vector<std::string>& input_strings, const std::string& mapping_path, int max_onehot = 1);

// 构建onehot
std::vector<std::vector<int>> build_str_onehot(const std::vector<std::string> &input_strings, const std::string& mapping_file_path, const size_t constraint_block_size = 0);

#endif
