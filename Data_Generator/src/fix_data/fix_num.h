#ifndef FIX_NUM_H
#define FIX_NUM_H

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

std::vector<std::string> replace_invalid_values(const std::vector<std::string> &column, const std::string &replacement_strategy = "mean", double custom_value = 0.0, const std::vector<std::string> &delimiters = {"|", ">>", ";"}) noexcept;

#endif
