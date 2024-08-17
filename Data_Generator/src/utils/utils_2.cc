#include "Utils_2.h"

bool is_double(const std::string& str) {
    if (str.empty()) return false;

    char* end;
    std::strtod(str.c_str(), &end);

    // Return true if the entire string was successfully converted
    return end != str.c_str() && *end == '\0';
}

// 判断配置文件路径
std::string get_config_file_path(int argc,  char * argv[])
{
    if (argc > 1)
    {
        std::string config_file_path_str = std::string(argv[1]);
        if(std::filesystem::exists(config_file_path_str))
            return config_file_path_str;
    }
    return std::string("config.txt");
}

template <>
int stringToType<int>(const std::string &str)
{
  return std::stoi(str);
}

template <>
float stringToType<float>(const std::string &str)
{
  return std::stof(str);
}

template <>
double stringToType<double>(const std::string &str)
{
  return std::stod(str);
}

template <>
int convert_specific_type<int>(const std::string &str)
{
  return std::stoi(str);
}

template <>
float convert_specific_type<float>(const std::string &str)
{
  return std::stof(str);
}

template <>
double convert_specific_type<double>(const std::string &str)
{
  return std::stod(str);
}

template<>
long convert_specific_type<long>(const std::string &str)
{
    return std::stol(str);
}


template <>
unsigned long convert_specific_type<unsigned long>(const std::string& str)
{
    return std::stoul(str);
}



bool copy_file(const std::string &sourceFile, const std::string &destinationFile)
{
  std::ifstream src(sourceFile, std::ios::binary);
  if (!src)
  {
    std::cerr << "Error opening source file: " << sourceFile << std::endl;
    return false;
  }

  std::ofstream dst(destinationFile, std::ios::binary);
  if (!dst)
  {
    std::cerr << "Error opening destination file: " << destinationFile << std::endl;
    return false;
  }

  dst << src.rdbuf();

  if (!dst)
  {
    std::cerr << "Error writing to destination file: " << destinationFile << std::endl;
    return false;
  }

  return true;
}

std::vector<std::string> insert_separator_to_vec(const std::vector<std::string> &vec, const std::string &separator)
{
  if (vec.empty() || vec.size() == 1)
    return vec;

  std::vector<std::string> result;
  result.reserve(vec.size() + vec.size() - 1); // 预留足够空间

  for (size_t i = 0; i < vec.size(); ++i)
  {
    result.emplace_back(vec[i]);
    if (i != vec.size() - 1)
      result.emplace_back(separator);
  }

  return result;
}

std::vector<std::string> manual_split(const std::string &text, const std::vector<std::string> &delimiters)
{
  auto find_next_delimiter = [&](size_t start) -> std::pair<size_t, size_t> {
    size_t min_pos = std::string::npos;
    size_t delim_len = 0;

    for (const auto &delim : delimiters)
    {
      size_t pos = text.find(delim, start);
      if (pos != std::string::npos && (min_pos == std::string::npos || pos < min_pos))
      {
        min_pos = pos;
        delim_len = delim.size();
      }
    }

    return {min_pos, delim_len};
  };

  std::vector<std::string> tokens;
  size_t start = 0;

  while (start < text.size())
  {
    std::pair<size_t, size_t> next_delim = find_next_delimiter(start);
    size_t min_pos = next_delim.first;
    size_t delim_len = next_delim.second;

    if (min_pos == std::string::npos)
    {
      tokens.push_back(text.substr(start));
      break;
    }
    else
    {
      tokens.push_back(text.substr(start, min_pos - start));
      start = min_pos + delim_len;
    }
  }

  // 处理文本末尾是分隔符的情况
  if (start == text.size())
    tokens.push_back("");

  return tokens;
}

std::vector<std::string> split_str_multi_sep(const std::string &str, const std::vector<std::string> &delimiters)
{
  std::vector<std::string> result;
  size_t start = 0;
  size_t end = 0;
  size_t str_size = str.size();
  result.reserve(str_size / 2); // Reserve space to reduce reallocation

  while (start < str_size)
  {
    size_t min_pos = std::string::npos;
    std::string current_delim;

    // Find the first occurrence of any of the delimiters
    for (const auto &delim : delimiters)
    {
      size_t pos = str.find(delim, start);
      if (pos != std::string::npos && (min_pos == std::string::npos || pos < min_pos))
      {
        min_pos = pos;
        current_delim = delim;
      }
    }

    if (min_pos != std::string::npos)
    {
      // Extract substring before the delimiter (including empty parts)
      result.emplace_back(str.substr(start, min_pos - start));
      // Move start position to after the delimiter
      start = min_pos + current_delim.size();
    }
    else
    {
      // Add the last part if there's any remaining string
      result.emplace_back(str.substr(start));
      break;
    }
  }

  return result;
}

std::vector<std::string> split_str_multi_sep_trim(const std::string &str, const std::vector<std::string> &delimiters)
{
  std::vector<std::string> result;
  size_t start = 0;
  size_t end = 0;
  size_t str_size = str.size();
  result.reserve(str_size / 2); // Reserve space to reduce reallocation

  while (start < str_size)
  {
    size_t min_pos = std::string::npos;
    std::string current_delim;

    // Find the first occurrence of any of the delimiters
    for (const auto &delim : delimiters)
    {
      size_t pos = str.find(delim, start);
      if (pos != std::string::npos && (min_pos == std::string::npos || pos < min_pos))
      {
        min_pos = pos;
        current_delim = delim;
      }
    }

    if (min_pos != std::string::npos)
    {
      // Extract substring before the delimiter (including empty parts)
      result.emplace_back(trim(str.substr(start, min_pos - start)));
      // Move start position to after the delimiter
      start = min_pos + current_delim.size();
    }
    else
    {
      // Add the last part if there's any remaining string
      result.emplace_back(trim(str.substr(start)));
      break;
    }
  }

  return result;
}

void Timer::start()
{
  start_time_point = std::chrono::high_resolution_clock::now();
}

double Timer::stop(const std::string &unit)
{
  end_time_point = std::chrono::high_resolution_clock::now();
  auto duration = end_time_point - start_time_point;

  if (unit == "seconds")
    return std::chrono::duration_cast<std::chrono::seconds>(duration).count();
  else if (unit == "milliseconds")
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
  else if (unit == "microseconds")
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  else if (unit == "nanoseconds")
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
  else
  {
    std::cerr << "Unknown time unit: " << unit << std::endl;
    return 0.0;
  }
}
