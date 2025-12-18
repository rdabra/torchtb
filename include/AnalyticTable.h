#ifndef ANALYTICTABLE_H
#define ANALYTICTABLE_H
#pragma once

#include "detail/utils.h"

#include <arrow/type.h>
#include <cstdint>
#include <optional>
#include <utility>

namespace ttb {

enum class Axis { ROW = 0, COLUMN = 1 };

enum class SortOrder { ASC = 0, DESC = 1 };
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

    AnalyticTable(utl::shp<arrow::Table> &&arrow_table) : _arrow_tb{std::move(arrow_table)} {};

    [[nodiscard]] int64_t n_rows() const;
    [[nodiscard]] int n_cols() const;
    [[nodiscard]] std::vector<std::string> col_names() const;
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

    /**
     * @brief Moves the specified column to the rightmost postion and one-hot encode it with
     * int values
     *
     * @param col_index Column to be one-hot encoded
     * @return utl::ReturnCode
     */
    virtual void one_hot_expand(int col_index);

    /**
     * @brief Extracts the specified column from this table
     *
     * @param col_index Index of the column to be removed
     * @return std::expected<ttb::DataTable, utl::ReturnCode>
     */
    ttb::AnalyticTable extract_column(int col_index);

    /**
     * @brief Extracts the columns to the right of the specified index
     *
     * @param col_index Column the right of which other columns are extracted
     * @return std::expected<ttb::DataTable, utl::ReturnCode>
     */
    ttb::AnalyticTable right_extract_of(int col_index);

    /**
     * @brief Returns a row-wise portion of this table
     *
     * @param row_offset Starting row index (inclusive)
     * @param row_length Number of rows to slice (final_index=row_offset + row_length)
     * @return std::expected<ttb::DataTable, utl::ReturnCode>
     */
    [[nodiscard]] ttb::AnalyticTable sliced(int64_t row_offset, int64_t row_length) const;
    [[nodiscard]] ttb::AnalyticTable copy_cols(std::vector<int> indices) const;

    [[nodiscard]] ttb::AnalyticTable clone() const;

    void print_head(int64_t n_rows = 20) const;
    void print_tail(int64_t n_rows = 20) const;
    void reset();

    [[nodiscard]] const utl::shp<arrow::Table> &arrow_table() const { return _arrow_tb; }

  protected:
    AnalyticTable() = default;

    utl::shp<arrow::Table> _arrow_tb{nullptr};

    void bottom_append(const AnalyticTable &table);
    void right_append(const AnalyticTable &table);
};

class AnalyticTableError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace ttb
#endif