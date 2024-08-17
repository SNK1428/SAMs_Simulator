#ifndef Utils_2_H
#define Utils_2_H

#include "../include/IMPORT_LIB.h"
#include <algorithm>
#include <cstddef>
#include <vector>

bool is_double(const std::string& str);

// 判断配置文件路径
std::string get_config_file_path(int argc,  char * argv[]);

// Templated function to generate and return a random series
template <typename T>
std::vector<T> random_vec_generator(size_t length, T min_value, T max_value) {
    static_assert(std::is_integral<T>::value, "T must be an integral type");            // static_assert is a check happend in compile stage
    assert(min_value <= max_value);
    assert(length != 0);

    std::random_device rd; // Hardware random number generator
    std::uniform_int_distribution<T> distrib(min_value, max_value);

    std::vector<T> series;
    series.reserve(length); // Reserve memory to avoid reallocations

    for (size_t i = 0; i < length; ++i)
        series.emplace_back(distrib(rd)); // Generate and store random number

    return series;
}

// trim from start (in place)
inline void ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
}

// trim from end (in place)
inline void rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

// trim from both ends (in place)
inline void trim(std::string &s)
{
    ltrim(s);
    rtrim(s);
}

// trim from start (copying)
inline std::string ltrim_copy(std::string s)
{
    ltrim(s);
    return s;
}

// trim from end (copying)
inline std::string rtrim_copy(std::string s)
{
    rtrim(s);
    return s;
}

// trim from both ends (copying)
inline std::string trim_copy(std::string s)
{
    trim(s);
    return s;
}

// 专用转换函数，用于将std::string转换为特定类型
template <typename T>
T stringToType(const std::string &str);

template <>
int stringToType<int>(const std::string &str);

template <>
float stringToType<float>(const std::string &str);

template <>
double stringToType<double>(const std::string &str);

// 通用转换函数模板，用于将一种类型的二维vector转换为另一种类型
    template <typename T, typename U>
std::vector<std::vector<U>> convert_2D_vector(const std::vector<std::vector<T>> &input)
{
    std::vector<std::vector<U>> result;
    result.reserve(input.size());

    for (const auto &row : input)
    {
        std::vector<U> convertedRow;
        convertedRow.reserve(row.size());
        for (const auto &element : row)
        {
            if constexpr (std::is_same<T, std::string>::value && (std::is_same<U, int>::value || std::is_same<U, float>::value || std::is_same<U, double>::value))
            {
                convertedRow.emplace_back(stringToType<U>(element));
            }
            else
            {
                convertedRow.emplace_back(static_cast<U>(element));
            }
        }
        result.emplace_back(std::move(convertedRow));
    }

    return result;
}

// 打印一维vector，但是不会主动换行
    template <typename T>
void print_vector_no_blank(const std::vector<T> &vec)
{
    for (const T &val : vec)
        std::cout << "\"" << val << "\"\t";
}

// 打印一维vector的模板函数
    template <typename T>
void print_vector(const std::vector<T> &vec)
{
    std::cout << "{\t";
    for (const T &val : vec)
        std::cout << "\"" << val << "\"" << "\t";
    std::cout << "}" << std::endl;
}

// 打印二维vector的模板函数
    template <typename T>
void print_vector(const std::vector<std::vector<T>> &vec)
{
    std::cout << "\n";
    for (const std::vector<T> &row : vec)
        print_vector(row);
}

// 将 one-hot 编码字符串转换为一维向量

// 模板函数，将 one-hot 编码字符串转换为一维向量，元素类型由模板参数决定
    template <typename T>
std::vector<T> split_one_hot_to_vector(const std::string &onehot_str)
{
    std::vector<T> vector;
    vector.reserve(onehot_str.size()); // 预分配内存
    for (char ch : onehot_str)
        // 将字符转换为整数并添加到向量中
        vector.emplace_back(static_cast<T>(ch - '0'));
    return vector;
}

// 模板函数，将向量中的每个元素与一个常数相乘
template <typename T, typename U>
auto vector_multiply(const std::vector<T> &vector, U constant) -> std::vector<decltype(T() * U())>
{
    using ResultType = decltype(T() * U());
    std::vector<ResultType> result;
    result.reserve(vector.size()); // 预分配内存
    for (const auto &num : vector)
    {
        result.emplace_back(num * constant);
    }
    return result;
}

// 模板函数，将两个不同类型的等长向量按位相加，返回类型由模板参数决定
template <typename T, typename U, typename R = decltype(T() + U())>
auto vector_add(const std::vector<T> &vector1, const std::vector<U> &vector2) -> std::vector<decltype(T() + U())>
{
    if (vector1.size() != vector2.size())
        throw std::invalid_argument("Vectors must be of the same length.");
    std::vector<R> result;
    result.reserve(vector1.size()); // 预分配内存
    for (size_t i = 0; i < vector1.size(); ++i)
        result.emplace_back(vector1[i] + vector2[i]);
    return result;
}

// 函数：按指定分隔符（字符）分割字符串
static std::vector<std::string> split_string_ch(const std::string &s, char delimiter = '\t')
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

/**
 * @brief 1D内容写入文件
 *
 * @tparam T 任意能转换为字符串的类
 * @param matrix
 * @param path_file
 * @param delim 分隔符（默认制表符 "\\t"）
 */
template <typename T>
static inline void write_content_to_file(const std::vector<T> &vector, const std::string &path_file, const std::string &delim = "\t")
{
    std::ofstream file_obj(path_file.c_str());
    for (auto it_line = vector.begin(); it_line != vector.end(); it_line++)
    {
        file_obj << *it_line;
        if (it_line != vector.end() - 1)
            file_obj << "\n";
    }
    file_obj.close();
}

/**
 * @brief 内容写入文件
 *
 * @tparam T 任意能转换为字符串的类
 * @param matrix
 * @param path_file
 * @param delim 分隔符（默认制表符 "\\t"）
 */
template <typename T>
static inline void write_content_to_file(const std::vector<std::vector<T>> &matrix, const std::string &path_file, const std::string &delim = "\t")
{
    std::ofstream file_obj(path_file.c_str());

    // 使用 stringstream 作为缓冲区
    std::stringstream buffer;
    for (size_t i = 0; i < matrix.size(); ++i)
    {
        for (size_t j = 0; j < matrix[i].size(); ++j)
        {
            buffer << matrix[i][j];
            if (j != matrix[i].size() - 1)
                buffer << delim;
        }
        if (i != matrix.size() - 1)
            buffer << '\n';
    }

    // 一次性将缓冲区内容写入文件
    file_obj << buffer.str();
    file_obj.close();
}

// 转置二维vector矩阵
    template <class T>
static inline std::vector<std::vector<T>> transpose_matrix(const std::vector<std::vector<T>> &matrix)
{
    // 矩阵转置
    std::vector<std::vector<T>> array;
    std::vector<T> temparay;

    for (unsigned int i = 0; i < matrix.at(0).size(); ++i) // m*n 维数组
    {
        // std::cout << i << std ::endl;
        for (unsigned int j = 0; j < matrix.size(); ++j)
            temparay.push_back(matrix[j][i]);
        array.push_back(temparay);
        temparay.erase(temparay.begin(), temparay.end());
    }
    return array;
}

// 复制文件
bool copy_file(const std::string &sourceFile, const std::string &destinationFile);

// // 函数模板，用于将字符串转换为指定类型
template <typename T>
T convert_specific_type(const std::string &str);

template <>
int convert_specific_type<int>(const std::string &str);

template <>
float convert_specific_type<float>(const std::string &str);

template <>
double convert_specific_type<double>(const std::string &str);

template<>
long convert_specific_type<long>(const std::string &str);

template <>
unsigned long convert_specific_type<unsigned long>(const std::string& str);


// 函数：将std::vector<std::string>转换为std::vector<double>
inline std::vector<double> convert_to_double_vector(const std::vector<std::string>& string_vec) {
    std::vector<double> double_vec;
    double_vec.reserve(string_vec.size());
    for (const std::string& str : string_vec)
        double_vec.emplace_back(std::stod(str));
    return double_vec;
}

// 函数，将二维std::string数组转换为二维其他类型的数组
    template <typename T>
std::vector<std::vector<T>> convert_2D_string_array(const std::vector<std::vector<std::string>> &stringArray)
{
    std::vector<std::vector<T>> result;
    result.reserve(stringArray.size());

    for (const auto &row : stringArray)
    {
        std::vector<T> convertedRow;
        convertedRow.reserve(row.size());
        for (const auto &element : row)
        {
            convertedRow.emplace_back(convert_specific_type<T>(element));
        }
        result.emplace_back(std::move(convertedRow));
    }

    return result;
}

// 通用模板，将任意类型转换为std::string
    template <typename T>
std::string toString(const T &value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

// 特化模板用于处理一维std::vector类型
    template <typename T>
std::vector<std::string> convert_vector_to_string(const std::vector<T> &vec)
{
    std::vector<std::string> result;
    result.reserve(vec.size());
    for (const auto &item : vec)
    {
        result.push_back(toString(item));
    }
    return result;
}

// 特化模板用于处理二维std::vector类型
    template <typename T>
std::vector<std::vector<std::string>> convert_vector_to_string(const std::vector<std::vector<T>> &vec)
{
    std::vector<std::vector<std::string>> result;
    result.reserve(vec.size());
    for (const auto &row : vec)
    {
        result.push_back(convert_vector_to_string(row));
    }
    return result;
}

// 模板函数，将向量转换为字符串，不预估长度
    template <typename T>
inline std::string vector_to_string(const std::vector<T> &vec)
{
    std::ostringstream oss;
    // 遍历向量中的每个元素，并将其转换为字符串并追加到结果字符串中
    for (const auto &item : vec)
    {
        oss << item;
    }
    return oss.str();
}

// 特化版本，用于处理字符串类型的向量
    template <>
inline std::string vector_to_string<std::string>(const std::vector<std::string> &vec)
{
    std::string results;
    size_t total_length = 0;
    for (auto ele : vec)
        total_length += ele.size();
    results.reserve(total_length);
    for (auto ele : vec)
        results.append(ele);
    return results;
}

// 检查矩阵维度
    template <typename T>
void check_dimensions(const std::vector<std::vector<T>> &mat_1, const std::vector<std::vector<T>> &mat_2, size_t axis)
{
    if (axis == 0)
    {
        for (const auto &row : mat_1)
            if (!row.empty() && row.size() != mat_2[0].size())
                throw std::runtime_error("Matrices have different number of columns, or size is zero");
    }
    else if (axis == 1)
    {
        if (mat_1.size() != mat_2.size())
            throw std::runtime_error("Matrices have different number of rows");
    }
    else
        throw std::invalid_argument("Invalid axis value");
}

// 拼接矩阵，，将mat_2直接拼接到mat_1上
    template <typename T>
std::vector<std::vector<T>> concatenate_matrices(const std::vector<std::vector<T>> &mat_1, const std::vector<std::vector<T>> &mat_2, size_t axis = 0)
{
    check_dimensions(mat_1, mat_2, axis);

    std::vector<std::vector<T>> result = mat_1;

    if (axis == 0)
    {
        result.reserve(mat_1.size() + mat_2.size());
        result.insert(result.end(), mat_2.begin(), mat_2.end());
    }
    else if (axis == 1)
    {
        for (size_t i = 0; i < mat_1.size(); ++i)
        {
            result[i].reserve(mat_1[i].size() + mat_2[i].size());
            result[i].insert(result[i].end(), mat_2[i].begin(), mat_2[i].end());
        }
    }

    return result;
}

// 方法定义
inline static std::vector<std::string> split_string_v2(const std::string &str, const std::string &delimiter)
{
    std::vector<std::string> tokens;
    if (delimiter.empty())
    {
        tokens.push_back(str);
        return tokens;
    }

    size_t start = 0;
    size_t end;

    while ((end = str.find(delimiter, start)) != std::string::npos)
    {
        tokens.emplace_back(str, start, end - start);
        start = end + delimiter.length();
    }

    // 最后一个子字符串
    tokens.emplace_back(str, start, str.length() - start);

    // 处理空字符串特殊情况
    if (str.empty() || (start == 0 && str.find(delimiter) == std::string::npos))
    {
        tokens.clear();
        tokens.push_back(str);
    }

    return tokens;
}

// 对于一维vector，在每一个元素中间插入一个字符串，如 {"a","b","c"} 中间插入 "i"，则变为{"a","i","b","i","c"}
std::vector<std::string> insert_separator_to_vec(const std::vector<std::string> &vec, const std::string &separator);

    template <typename T>
void remove_element_by_index(std::vector<T> &vec, size_t index)
{
    if (index < vec.size())
    {
        vec.erase(vec.begin() + index);
    }
    else
    {
        throw std::out_of_range("Index out of range");
    }
}

// 新的转置矩阵的方法（创建新矩阵）
    template <typename T>
std::vector<std::vector<T>> new_transpose_matrix(const std::vector<std::vector<T>> &vec)
{
    // Assert that the input vector is not empty (for debugging purposes)
    assert(!vec.empty() && "Input vector should not be empty");

    // Determine the maximum row size (number of columns in the transposed matrix)
    size_t max_col_size = 0;
    for (const auto &row : vec) 
        max_col_size = std::max(max_col_size, row.size());

    std::vector<std::vector<T>> transposed(max_col_size, std::vector<T>(vec.size()));

    // Fill the transposed matrix
    for(size_t i = 0; i < vec.size(); i+=1)
        for(size_t j = 0; j < vec[i].size(); j+=1)
            transposed[j][i] = vec[i][j];

    return transposed;
}

/**
 * @brief Splits the input text into tokens based on multiple delimiters.
 *
 * This function splits the input text into tokens, considering multiple delimiters.
 * It also handles consecutive delimiters by inserting empty tokens.
 *
 * @param text The input text to be split.
 * @param delimiters A vector of delimiter strings.
 * @return A vector of tokens obtained by splitting the input text.
 */
std::vector<std::string> manual_split(const std::string &text, const std::vector<std::string> &delimiters);

// Helper function to trim whitespace from both ends of a string

// 左右修剪
inline std::string trim(const std::string &s) noexcept
{
    auto ltrim = [](const std::string &str) noexcept -> std::string {
        auto start = std::find_if(str.begin(), str.end(), [](unsigned char ch) { return !std::isspace(ch); });
        return std::string(start, str.end());
    };

    auto rtrim = [](const std::string &str) noexcept -> std::string {
        auto end = std::find_if(str.rbegin(), str.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base();
        return std::string(str.begin(), end);
    };

    return ltrim(rtrim(s));
}

// 基于正则表达式的分隔符切分字符串

inline std::vector<std::string> split_string_reg(const std::string &input, const std::string &reg_str)
{
    std::vector<std::string> result;
    std::regex re(reg_str);
    std::sregex_token_iterator it(input.begin(), input.end(), re, -1);
    std::sregex_token_iterator reg_end;
    for (; it != reg_end; ++it)
        result.emplace_back(trim(it->str()));
    return result;
}

// 用指定分隔符分割字符串, 并对每个子字符串进行trim
inline std::vector<std::string> split_string_trim(const std::string &input, const std::string &delimiter)
{
    std::vector<std::string> result;
    size_t start = 0;
    size_t end = input.find(delimiter);
    while (end != std::string::npos)
    {
        result.emplace_back(trim(input.substr(start, end - start)));
        start = end + delimiter.length();
        end = input.find(delimiter, start);
    }
    result.emplace_back(trim(input.substr(start)));
    return result;
}

std::vector<std::string> split_str_multi_sep(const std::string &str, const std::vector<std::string> &delimiters);

std::vector<std::string> split_str_multi_sep_trim(const std::string &str, const std::vector<std::string> &delimiters);

// Template function to read and split file into a 2D vector of specified type
template <typename T>
std::vector<std::vector<T>> read_2d_datas(const std::string &filename, const std::string &delimiter = "\t")
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return {};
    }

    std::vector<std::vector<T>> data;
    std::string line;
    while (std::getline(file, line))
    {
        data.emplace_back(split_string_v2(line, "\t"));
    }

    file.close();
    return data;
}
class Timer
{
    public:
        void start();
        double stop(const std::string &unit = "seconds");

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time_point;
        std::chrono::time_point<std::chrono::high_resolution_clock> end_time_point;
};

#endif // Utils_2_H
