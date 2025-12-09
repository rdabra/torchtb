#ifndef PARQUET_IO_H
#define PARQUET_IO_H
#include "XYMatrix.h"
#pragma once

#include "AnalyticTable.h"
#include "AnalyticTableNumeric.h"

#include "detail/utils.h"
#include <ATen/core/TensorBody.h>
#include <expected>
#include <filesystem>

#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/platform.h>
#include <parquet/properties.h>
#include <parquet/type_fwd.h>

namespace ttb {

class Parquet_IO {
  public:
    Parquet_IO(std::filesystem::path path) : _path{std::move(path)} {};

    [[nodiscard]] std::expected<ttb::AnalyticTable, utl::ReturnCode> read() const;

    template <utl::NumericType T>
    [[nodiscard]] std::expected<ttb::AnalyticTableNumeric<T>, utl::ReturnCode> read_numeric() const;

    [[nodiscard]] utl::ReturnCode write(const ttb::AnalyticTable &table) const;

    template <utl::NumericType T>
    utl::ReturnCode write(torch::Tensor &&tensor) const;

    template <utl::NumericType T>
    utl::ReturnCode write(ttb::XYMatrix &&xy_matrix) const;

  private:
    std::filesystem::path _path;
};

} // namespace ttb
#endif