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

    [[nodiscard]] ttb::AnalyticTable read() const;

    template <utl::NumericType T>
    [[nodiscard]] ttb::AnalyticTableNumeric<T> read_numeric() const;

    void write(const ttb::AnalyticTable &table) const;

    template <utl::NumericType T>
    void write(torch::Tensor &&tensor) const;

    template <utl::NumericType T>
    void write(ttb::XYMatrix &&xy_matrix) const;

  private:
    std::filesystem::path _path;
};

class Parquet_IOError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace ttb
#endif