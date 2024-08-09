#include "fix_str.h"
#include "../utils/Utils_2.h"
#include <cstddef>
#include <utility>

using namespace std;

inline vector<string> split_by_multiple_delimiters(const string &item, const vector<string> &delimiters)
{
    vector<string> parts = {item};
    for (const auto &delimiter : delimiters)
    {
        vector<string> temp_list;
        for (const auto &part : parts)
        {
            vector<string> split_parts = split_string_v2(part, delimiter);
            temp_list.insert(temp_list.end(), split_parts.begin(), split_parts.end());
        }
        parts = std::move(temp_list);
    }
    return parts;
}

unordered_map<string, int> count_string_occurrences(const vector<string> &arr, const vector<string> &delimiters) noexcept
{
    auto clean_and_split_array = [&](const vector<string> &arr)
    {
        vector<string> clean_list;
        for (const auto &item : arr)
        {
            vector<string> parts = split_by_multiple_delimiters(item, delimiters);
            clean_list.insert(clean_list.end(), parts.begin(), parts.end());
        }
        return std::move(clean_list);
    };

    vector<string> clean_list = clean_and_split_array(arr);
    unordered_map<string, int> counter;
    for (const auto &item : clean_list)
        counter[item]++;
    return counter;
}

string find_most_common_string(const vector<string> &clean_list) noexcept
{
    unordered_map<string, int> counter;
    for (const auto &item : clean_list)
        if (!item.empty())
            counter[item]++;
    string most_common_str;
    int max_count = 0;
    for (auto it = counter.begin(); it != counter.end(); ++it)
        if (it->second > max_count)
        {
            most_common_str = it->first;
            max_count = it->second;
        }
    return most_common_str;
}

vector<string> replace_in_array(const vector<string> &arr, const unordered_set<string> &replace_set, const string &replacement, const vector<string> &delimiters) noexcept
{
    auto replace_in_item = [&](const string &item) noexcept 
    {
        vector<string> parts = split_by_multiple_delimiters(item, delimiters);

        auto replace_parts = [&](const vector<string> &parts) 
        {
            vector<string> replaced_parts;
            for (const auto &part : parts)
                if (replace_set.find(part) != replace_set.end())
                    replaced_parts.push_back(replacement);
                else
                    replaced_parts.push_back(part);
            return replaced_parts;
        };

        vector<string> replaced_parts = replace_parts(parts);
        for (const auto &delimiter : delimiters)
        {
            if (item.find(delimiter) != string::npos)
            {
                stringstream ss;
                for (size_t i = 0; i < replaced_parts.size(); ++i)
                {
                    if (i > 0)
                        ss << delimiter;
                    ss << replaced_parts[i];
                }
                return ss.str();
            }
        }
        return replaced_parts[0];
    };

    vector<string> result;
    result.reserve(arr.size());
    for (const auto &item : arr)
        result.emplace_back(replace_in_item(item));
    return result;
}

tuple<vector<string>, unordered_map<string, int>, unordered_map<string, int>> replace_empty_values_with_count(vector<string> &arr, const string &replacement,
        const vector<string> &replace_list,
        const vector<string> &delimiters) noexcept
{
    if (arr.empty())
        return make_tuple(vector<string>{}, unordered_map<string, int>{}, unordered_map<string, int>{});

    unordered_map<string, int> count_before = count_string_occurrences(arr, delimiters);

    auto clean_and_split_array = [&](const vector<string> &arr) 
    {
        vector<string> clean_list;
        for (const auto &item : arr)
        {
            vector<string> parts = split_by_multiple_delimiters(item, delimiters);
            clean_list.insert(clean_list.end(), parts.begin(), parts.end());
        }
        return clean_list;
    };

    vector<string> clean_list = clean_and_split_array(arr);

    string most_common_str = replacement.empty() ? find_most_common_string(clean_list) : replacement;

    unordered_set<string> replace_set(replace_list.begin(), replace_list.end());
    replace_set.insert("");

    arr = replace_in_array(arr, replace_set, most_common_str, delimiters);

    unordered_map<string, int> count_after = count_string_occurrences(arr, delimiters);

    return make_tuple(arr, count_before, count_after);
}

