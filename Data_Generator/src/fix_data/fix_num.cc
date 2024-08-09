#include "fix_num.h"

// Splits a string into values and delimiters
inline static std::pair<std::vector<std::string>, std::vector<std::string>> split_values_and_delimiters(const std::string &item,
        const std::vector<std::string> &delimiters) noexcept
{
    std::vector<std::string> values;
    std::vector<std::string> delims;
    std::string current_value;
    size_t i = 0;
    while (i < item.size())
    {
        bool matched = false;
        for (const auto &delim : delimiters)
        {
            if (item.substr(i, delim.size()) == delim)
            {
                values.emplace_back(std::move(current_value));
                delims.emplace_back(delim);
                current_value.clear();
                i += delim.size();
                matched = true;
                break;
            }
        }
        if (!matched)
        {
            current_value += item[i];
            ++i;
        }
    }
    values.emplace_back(std::move(current_value));
    return {std::move(values), std::move(delims)};
}

// Splits a column into values and delimiters
inline static std::pair<std::vector<std::vector<std::string>>, std::vector<std::vector<std::string>>> split_column(const std::vector<std::string> &column,
        const std::vector<std::string> &delimiters) noexcept
{
    std::vector<std::vector<std::string>> split_columns;
    std::vector<std::vector<std::string>> split_delimiters;

    split_columns.reserve(column.size());
    split_delimiters.reserve(column.size());

    for (const auto &item : column)
    {
        auto split_result = split_values_and_delimiters(item, delimiters);
        split_columns.emplace_back(std::move(split_result.first));
        split_delimiters.emplace_back(std::move(split_result.second));
    }

    return {std::move(split_columns), std::move(split_delimiters)};
}

// Flattens a nested list of doubles
inline static std::vector<double> flatten_list(const std::vector<std::vector<double>> &nested_list) noexcept
{
    // Checks if a value is NaN
    static auto is_nan = [](double value) -> bool
    {
        return std::isnan(value);
    };

    std::vector<double> flat_list;
    flat_list.reserve(nested_list.size() * 2); // Rough estimate for initial size
    for (const auto &sublist : nested_list)
    {
        for (const auto &item : sublist)
        {
            if (!is_nan(item))
            {
                flat_list.emplace_back(item);
            }
        }
    }
    return flat_list;
}

// Calculates the mean of a list of doubles
inline static double calculate_mean(const std::vector<double> &values) noexcept
{
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

// Calculates the mode of a list of doubles
inline static double calculate_mode(const std::vector<double> &values) noexcept
{
    std::unordered_map<double, int> frequency;
    for (const auto &value : values)
    {
        frequency[value]++;
    }
    return std::max_element(frequency.begin(), frequency.end(), [](const auto &a, const auto &b) { return a.second < b.second; })->first;
}

// Recombines values and delimiters back into original format
inline static std::string recombine_row(const std::vector<double> &values, const std::vector<std::string> &delims) noexcept
{
    std::ostringstream oss;
    for (size_t i = 0; i < delims.size(); ++i)
    {
        oss << values[i] << delims[i];
    }
    oss << values.back();
    return oss.str();
}

// Recombines numeric columns and delimiters back into original format
inline static std::vector<std::string> recombine_columns(const std::vector<std::vector<double>> &numeric_columns,
        const std::vector<std::vector<std::string>> &split_delimiters) noexcept
{
    std::vector<std::string> recombined_columns;
    recombined_columns.reserve(numeric_columns.size());
    for (size_t i = 0; i < numeric_columns.size(); ++i)
    {
        recombined_columns.emplace_back(recombine_row(numeric_columns[i], split_delimiters[i]));
    }
    return recombined_columns;
}

std::vector<std::string> replace_invalid_values(const std::vector<std::string> &column, const std::string &replacement_strategy, double custom_value,
        const std::vector<std::string> &delimiters) noexcept
{
    auto split_result = split_column(column, delimiters);
    auto split_columns = std::move(split_result.first);
    auto split_delimiters = std::move(split_result.second);

    auto convert_row_to_numeric = [](const std::vector<std::string> &row) noexcept {
        auto process_value = [](const std::string &value) noexcept {
            if (value.empty())
            {
                return std::nan("");
            }
            char *end;
            double result = std::strtod(value.c_str(), &end);
            if (*end != '\0')
            {
                return std::nan("");
            }
            return result;
        };

        std::vector<double> numeric_row;
        numeric_row.reserve(row.size());
        for (const auto &value : row)
        {
            numeric_row.emplace_back(process_value(value));
        }
        return numeric_row;
    };

    auto convert_to_numeric = [&](const std::vector<std::vector<std::string>> &columns) noexcept {
        std::vector<std::vector<double>> numeric_columns;
        numeric_columns.reserve(columns.size());
        for (const auto &row : columns)
        {
            numeric_columns.emplace_back(convert_row_to_numeric(row));
        }
        return numeric_columns;
    };

    auto numeric_columns = convert_to_numeric(split_columns);

    auto calculate_replacement_value = [&](const std::vector<std::vector<double>> &columns) noexcept {
        auto flat_list = flatten_list(columns);
        if (replacement_strategy == "mean")
        {
            return calculate_mean(flat_list);
        }
        else if (replacement_strategy == "mode")
        {
            return calculate_mode(flat_list);
        }
        else if (replacement_strategy == "custom")
        {
            return custom_value;
        }
        else
        {
            return std::nan(""); // Invalid strategy, return NaN
        }
    };

    double replacement_value = calculate_replacement_value(numeric_columns);

    auto replace_invalid_values_in_row = [&](const std::vector<double> &row, double value) noexcept
    {


        // Checks if a value is NaN
        static auto is_nan = [](double value) -> bool
        {
            return std::isnan(value);
        };


        std::vector<double> replaced_row;
        replaced_row.reserve(row.size());
        for (const auto &item : row)
        {
            if (is_nan(item))
            {
                replaced_row.emplace_back(value);
            }
            else
            {
                replaced_row.emplace_back(item);
            }
        }
        return replaced_row;
    };

    auto replace_invalid_values_in_numeric_columns = [&](const std::vector<std::vector<double>> &columns, double value) noexcept {
        std::vector<std::vector<double>> replaced_columns;
        replaced_columns.reserve(columns.size());
        for (const auto &row : columns)
        {
            replaced_columns.emplace_back(replace_invalid_values_in_row(row, value));
        }
        return replaced_columns;
    };

    numeric_columns = replace_invalid_values_in_numeric_columns(numeric_columns, replacement_value);
    return recombine_columns(numeric_columns, split_delimiters);
}
