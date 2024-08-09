#include "fix_years.h"

// 处理日期字符串数组
std::vector<std::string> convert_to_days_since_custom(const std::vector<std::string> &date_strings, const std::string &start_date_string)
{

    // 解析日期字符串
    auto parse_date_string = [](const std::string &date_string, std::chrono::system_clock::time_point &date_time_point) {
        std::tm tm = {};
        if (sscanf(date_string.c_str(), "%d/%d/%d", &tm.tm_year, &tm.tm_mon, &tm.tm_mday) != 3)
        {
            std::cerr << "Error parsing date string '" << date_string << "'\n";
            return false;
        }
        tm.tm_year -= 1900; // Adjust years since 1900
        tm.tm_mon -= 1;     // Adjust months to be zero-based
        // 字符串转换为时间
        date_time_point = std::chrono::system_clock::from_time_t(std::mktime(&tm));
        return true;
    };

    std::vector<std::string> days_list;     // 已经转换的时间类型
    days_list.reserve(date_strings.size()); // 预分配内存，提高性能
    std::chrono::system_clock::time_point date_obj, start_date;

    if (!parse_date_string(start_date_string, start_date))
        return {"Invalid start date"};

    for (const auto &date_string : date_strings)
        if (parse_date_string(date_string, date_obj))
            days_list.emplace_back(std::to_string(std::chrono::duration_cast<std::chrono::hours>(date_obj - start_date).count() / 24));
        else
            days_list.emplace_back("Invalid date");
    return days_list;
}
