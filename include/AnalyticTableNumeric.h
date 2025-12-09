#ifndef ANALYTICTALENUMERIC_H
#define ANALYTICTALENUMERIC_H
#pragma once

#include "AnalyticTable.h"
#include "detail/utils.h"

namespace ttb {

template <utl::NumericType T>
class AnalyticTableNumeric : public ttb::AnalyticTable {
  public:
    AnalyticTableNumeric(const AnalyticTableNumeric &) = delete;
    AnalyticTableNumeric(AnalyticTableNumeric &&) = default;
    AnalyticTableNumeric &operator=(const AnalyticTableNumeric &) = delete;
    AnalyticTableNumeric &operator=(AnalyticTableNumeric &&) = default;
    ~AnalyticTableNumeric() = default;

    AnalyticTableNumeric(utl::shp<arrow::Table> &&arrow_table)
        : ttb::AnalyticTable{std::move(arrow_table)}, _arrow_dtype(utl::arrow_dtype<T>()) {
      this->to_dtype();
    };

    AnalyticTableNumeric(ttb::AnalyticTable &&data_table)
        : ttb::AnalyticTable{std::move(data_table)}, _arrow_dtype(utl::arrow_dtype<T>()) {
      this->to_dtype();
    }

    AnalyticTableNumeric(std::unordered_map<std::string, std::vector<T>> &&field_and_data);

    utl::ReturnCode one_hot_expand(int col_index) override;

    /**
     * @brief Finds the index of the max value in specified axis
     *
     * @param axis Direction to search for the max value
     * @return std::vector<int64_t> collection of indices
     */
    [[nodiscard]] std::vector<int64_t> argmax(ttb::Axis axis) const;

    static utl::shp<arrow::Table>
    make_numeric_table(std::unordered_map<std::string, std::vector<T>> &&field_and_data);

    [[nodiscard]] std::shared_ptr<arrow::DataType> arrow_dtype() const {
      return this->_arrow_dtype;
    }

  private:
    AnalyticTableNumeric() = default;
    utl::ReturnCode to_dtype();
    std::shared_ptr<arrow::DataType> _arrow_dtype;
};

class DataTableNumericError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

template <utl::NumericType T>
using TbNumeric = AnalyticTableNumeric<T>;

using TbInt = TbNumeric<int>;
using TbLong = TbNumeric<int64_t>;
using TbFloat = TbNumeric<float>;
using TbDouble = TbNumeric<double>;

} // namespace ttb
#endif