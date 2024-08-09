#ifndef NUM_ENCODE_H
#define NUM_ENCODE_H
#include <algorithm>
#include <asm-generic/errno.h>
#include <cstddef>
#include <iterator>
#include <new>
#include <ratio>
#include <regex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

std::vector<std::vector<std::string>> build_num_onehot(const std::vector<std::string>& str_array, const size_t padding_length = 0, const size_t block_size = 0);
#endif
