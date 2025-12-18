#include "detail/utils.h"
#include <arrow/compute/api.h>
#include <arrow/status.h>
#include <cctype>
#include <iterator>
#include <mutex>
#include <ranges>

std::string utl::to_lower(std::string word) {
  auto rr = std::views::transform(
      word, [](char w) -> char { return static_cast<char>(std::tolower(w)); });

  return std::string{std::begin(rr), std::end(rr)};
}

std::string utl::to_upper(std::string word) {
  auto rr = std::views::transform(
      word, [](char w) -> char { return static_cast<char>(std::toupper(w)); });

  return std::string{std::begin(rr), std::end(rr)};
}

void utl::initialize_arrow_compute() {
  static std::once_flag once;

  /// 'once' is marked done if lambda does not throw exception
  /// then all subsequent calls are ignored
  std::call_once(once, [] {
    auto st = arrow::compute::Initialize();
    if (!st.ok())
      throw std::runtime_error(st.ToString());
  });
}
