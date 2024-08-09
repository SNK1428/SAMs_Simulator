#ifndef _REGEX_23_H
#define _REGEX_23_H
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip> // 用于 std::fixed 和 std::setprecision
#include <iostream>
#include <map>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <iomanip>
#include <algorithm>

std::vector<std::vector<std::string>> build_23_raw_results(const std::string& defined_symbols_path, const std::vector<std::string>& unprocessed_formulas);

std::vector<std::vector<float>> build_23_maps_and_encode(const std::vector<std::vector<std::string>>& data_lists, std::size_t max_features, const std::string& filepath, bool create_new_mapping);
#endif
