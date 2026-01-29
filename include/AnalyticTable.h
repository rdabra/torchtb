#ifndef ANALYTICTABLE_H
#define ANALYTICTABLE_H
#pragma once

#include "detail/ttbutils.h"

#include <arrow/type.h>
#include <cstdint>
#include <optional>
#include <utility>

namespace ttb {

enum class Axis { ROW = 0, COLUMN = 1 };

enum class SortOrder { ASC = 0, DESC = 1 };

enum class LogicOp { AND = 0, OR = 1 };

struct Criterion {
    enum class Condition {
      LESS_EQUAL = 0,
      LESS = 1,
      EQUAL = 2,
      GREATER = 3,
      GREATER_EQUAL = 4,
      NOT_EQUAL = 5
    };

    std::string field_name{""};
    Condition condition{Condition::EQUAL};
    std::string value{""};
};

/**
 * @brief Analytics Base Table (ABT), in the sense defined by Kelleher et al. in
 * "Fundamentals of Machine Learning for Predictive Data Analytics".
 *
 */
class AnalyticTable {
  public:
    AnalyticTable(const AnalyticTable &) = delete;
    AnalyticTable(AnalyticTable &&) = default;
    AnalyticTable &operator=(const AnalyticTable &) = delete;
    AnalyticTable &operator=(AnalyticTable &&) = default;
    virtual ~AnalyticTable() = default;

    AnalyticTable(ttb::utl::shp<arrow::Table> &&arrow_table) : _arrow_tb{std::move(arrow_table)} {};

    [[nodiscard]] int64_t n_rows() const;
    [[nodiscard]] int n_cols() const;
    [[nodiscard]] std::vector<std::string> col_names() const;
    [[nodiscard]] std::string col_name(int index) const;
    [[nodiscard]] std::vector<std::string> col_dtypes() const;
    [[nodiscard]] std::optional<int> col_index(std::string name) const;

    void remove_col(int index);
    void keep_cols(std::vector<int> indices);
    void append(const AnalyticTable &table, const ttb::Axis &axis);
    void rename_cols(const std::vector<std::string> &names);
    void slice(int64_t row_offset, int64_t row_length);
    void reorder_cols(const std::vector<int> &indices);
    void move_column(int from_index, int to_index);
    void sort(int col_index, ttb::SortOrder mode = ttb::SortOrder::ASC);

    void keep_rows(const std::vector<ttb::Criterion> &criteria,
                   const ttb::LogicOp &op = ttb::LogicOp::AND);
    void drop_nulls(const std::vector<std::string> &field_names,
                    const ttb::LogicOp &op = ttb::LogicOp::AND);
    void drop_nulls(const ttb::LogicOp &op = ttb::LogicOp::AND);

    /**
     * @brief Removes duplicate rows from the table. Order is not preserved.
     *
     */
    void drop_duplicates();

    /**
     * @brief Moves the specified column to the rightmost postion and one-hot encode it with
     * int values
     *
     * @param col_index Column to be one-hot encoded
     * @return ttb::utl::ReturnCode
     */
    virtual void one_hot_expand(int col_index);

    /**
     * @brief Extracts the specified column from this table
     *
     * @param col_index Index of the column to be removed
     * @return std::expected<ttb::DataTable, ttb::utl::ReturnCode>
     */
    ttb::AnalyticTable extract_column(int col_index);

    /**
     * @brief Extracts the columns to the right of the specified index
     *
     * @param col_index Column the right of which other columns are extracted
     * @return std::expected<ttb::DataTable, ttb::utl::ReturnCode>
     */
    ttb::AnalyticTable right_extract_of(int col_index);

    /**
     * @brief Returns a row-wise portion of this table
     *
     * @param row_offset Starting row index (inclusive)
     * @param row_length Number of rows to slice (final_index=row_offset + row_length)
     * @return std::expected<ttb::DataTable, ttb::utl::ReturnCode>
     */
    [[nodiscard]] ttb::AnalyticTable sliced(int64_t row_offset, int64_t row_length) const;
    [[nodiscard]] ttb::AnalyticTable copy_cols(std::vector<int> indices) const;

    [[nodiscard]] ttb::AnalyticTable clone() const;

    void print_head(int64_t n_rows = 20) const;
    void print_tail(int64_t n_rows = 20) const;
    void reset();

    [[nodiscard]] const ttb::utl::shp<arrow::Table> &arrow_table() const { return _arrow_tb; }

  protected:
    AnalyticTable() = default;

    ttb::utl::shp<arrow::Table> _arrow_tb{nullptr};

    void bottom_append(const AnalyticTable &table);
    void right_append(const AnalyticTable &table);
    std::shared_ptr<arrow::Scalar>
    to_arrow_scalar(const std::shared_ptr<arrow::DataType> &field_type, std::string value);
    arrow::compute::Expression
    apply_filter_xx(const std::vector<arrow::compute::Expression> expressions);
    ttb::utl::shp<arrow::Table>
    submit_expressions(const std::vector<arrow::compute::Expression> expressions, ttb::LogicOp op);
};

class AnalyticTableError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace ttb
#endif