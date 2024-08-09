#include "Config.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace Config
{
    const std::map<std::string, std::vector<std::string>> *config = nullptr;

    std::map<std::string, std::vector<std::string>> readConfig(const std::string &filename)
    {
        std::map<std::string, std::vector<std::string>> tempConfig;
        std::ifstream file(filename);
        std::string line;

        if (file.is_open())
        {
            while (std::getline(file, line))
            {
                // 去除首尾空格
                line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
                line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

                // 跳过空行和以#号开头的注释行
                if (line.empty() || line[0] == '#')
                {
                    continue;
                }

                // 处理行内注释
                size_t commentPos = line.find('#');
                if (commentPos != std::string::npos)
                {
                    line = line.substr(0, commentPos);
                    // 去除行内注释后的首尾空格
                    line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
                    line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);
                }

                std::istringstream iss(line);
                std::string key, value;
                if (std::getline(iss, key, '=') && std::getline(iss, value))
                {
                    // 去除键和值的首尾空格
                    key.erase(0, key.find_first_not_of(" \t\n\r\f\v"));
                    key.erase(key.find_last_not_of(" \t\n\r\f\v") + 1);
                    value.erase(0, value.find_first_not_of(" \t\n\r\f\v"));
                    value.erase(value.find_last_not_of(" \t\n\r\f\v") + 1);

                    std::vector<std::string> values;
                    std::istringstream valueStream(value);
                    std::string singleValue;
                    while (std::getline(valueStream, singleValue, ','))
                    {
                        // 去除每个值的首尾空格
                        singleValue.erase(0, singleValue.find_first_not_of(" \t\n\r\f\v"));
                        singleValue.erase(singleValue.find_last_not_of(" \t\n\r\f\v") + 1);
                        values.push_back(singleValue);
                    }

                    tempConfig[key] = values;
                }
            }
            file.close();
        }
        else
        {
            std::cerr << "Unable to open file: " << filename << std::endl;
        }

        return tempConfig;
    }

    void initialize_config(const std::string &filename)
    {
        static std::map<std::string, std::vector<std::string>> tempConfig = readConfig(filename);
        config = &tempConfig;
    }
} // namespace Config
