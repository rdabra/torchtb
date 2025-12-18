#ifndef UTILS_H
#define UTILS_H
#pragma once

#include <arrow/status.h>

namespace utl {

static const std::string LIBRARY_NAME{"torchtb"};

template <typename T>
concept NumericType = std::same_as<T, int> || std::same_as<T, int64_t> || std::same_as<T, float> ||
                      std::same_as<T, double>;

template <utl::NumericType T>
using ArrowType = typename arrow::CTypeTraits<T>::ArrowType;

template <utl::NumericType T>
constexpr std::shared_ptr<arrow::DataType> arrow_dtype() {
  return arrow::CTypeTraits<T>::type_singleton();
}

template <typename T>
constexpr torch::ScalarType torch_type() {
  return c10::CppTypeToScalarType<T>::value;
}

template <utl::NumericType T>
using ArrowArrayType = typename arrow::TypeTraits<ArrowType<T>>::ArrayType;

template <utl::NumericType T>
using ArrowBuilderType = typename arrow::TypeTraits<ArrowType<T>>::BuilderType;

std::string to_lower(std::string word);
std::string to_upper(std::string word);

template <typename T>
using shp = std::shared_ptr<T>;

template <typename T>
using wkp = std::weak_ptr<T>;

template <typename T>
using unp = std::unique_ptr<T>;

template <typename T, typename... Args>
auto new_unp(Args &&...args) -> unp<T> {
  return std::make_unique<T>(std::forward<Args>(args)...);
}

template <typename T, typename... Args>
auto new_shp(Args &&...args) -> unp<T> {
  return std::make_shared<T>(std::forward<Args>(args)...);
}

template <typename T>
bool is_zero(T value) {
  if constexpr (std::is_floating_point_v<T>)
    return std::abs(value) < std::numeric_limits<T>::epsilon();

  return value == 0;
}

void initialize_arrow_compute();

} // namespace utl
#endif
