#ifndef IO_FILE_H
#define IO_FILE_H
#pragma once

#include "AnalyticTable.h"
#include "AnalyticTableNumeric.h"
#include "XYMatrix.h"

namespace ttb {

class IO_File {
  public:
    IO_File(const IO_File &) = default;
    IO_File(IO_File &&) = delete;
    IO_File &operator=(const IO_File &) = default;
    IO_File &operator=(IO_File &&) = delete;
    virtual ~IO_File() = default;

    IO_File(std::filesystem::path path) : _path{std::move(path)} {};

    [[nodiscard]] virtual ttb::AnalyticTable read() const = 0;

    virtual void write(const ttb::AnalyticTable &table) const = 0;

    template <utl::NumericType T>
    [[nodiscard]] ttb::AnalyticTableNumeric<T> read_numeric() const;

    template <utl::NumericType T>
    void write_tensor(torch::Tensor &&tensor) const;

    template <utl::NumericType T>
    void write_matrix(ttb::XYMatrix &&xy_matrix) const;

  protected:
    std::filesystem::path _path;
};

} // namespace ttb
#endif