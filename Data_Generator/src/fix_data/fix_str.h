#ifndef FIX_STR_H
#define FIX_STR_H
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

tuple<vector<string>, unordered_map<string, int>, unordered_map<string, int>> replace_empty_values_with_count(vector<string> &arr, const string &replacement = "",
                                                                                                              const vector<string> &replace_list = {},
                                                                                                              const vector<string> &delimiters = {"|", ";", ">>"}) noexcept;

#endif
