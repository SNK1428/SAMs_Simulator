#ifndef FIX_YEARS_H
#define FIX_YEARS_H

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::vector<std::string> convert_to_days_since_custom(const std::vector<std::string> &date_strings, const std::string &start_date_string);

#endif
