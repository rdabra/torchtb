#include "detail/utils.h"
#include <arrow/compute/api.h>
#include <arrow/status.h>
#include <cctype>
#include <iterator>
#include <mutex>
#include <ranges>

utl::ReturnCode utl::map_status(const arrow::Status &status) {
  if (status.ok())
    return utl::ReturnCode::Ok;
  switch (status.code()) {
  case arrow::StatusCode::OutOfMemory:
    return utl::ReturnCode::OutOfMemory;
  case arrow::StatusCode::IOError:
    return utl::ReturnCode::IOError;
  case arrow::StatusCode::CapacityError:
    return utl::ReturnCode::CapacityError;
  case arrow::StatusCode::Invalid:
    return utl::ReturnCode::Invalid;
  case arrow::StatusCode::KeyError:
    return utl::ReturnCode::KeyError;
  case arrow::StatusCode::IndexError:
    return utl::ReturnCode::IndexError;
  case arrow::StatusCode::Cancelled:
    return utl::ReturnCode::Cancelled;
  case arrow::StatusCode::TypeError:
    return utl::ReturnCode::TypeError;
  case arrow::StatusCode::SerializationError:
    return utl::ReturnCode::SerializationError;
  case arrow::StatusCode::NotImplemented:
    return utl::ReturnCode::NotImplemented;
  default:
    return utl::ReturnCode::Unknown;
  }
}

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
    return ReturnCode::Ok;
  });
}
