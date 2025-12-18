#ifndef CSVREADER_H
#define CSVREADER_H

#include "AnalyticTable.h"
#include "AnalyticTableNumeric.h"
#include "detail/utils.h"

#include <arrow/table.h>
#include <filesystem>
#include <utility>

#pragma once

namespace ttb {

class CSV_IO {
  public:
    CSV_IO(std::filesystem::path path, bool has_header = true)
        : _path{std::move(path)}, _has_header{has_header} {};

    [[nodiscard]] ttb::AnalyticTable read(char separator = ',') const;

    template <utl::NumericType T>
    ttb::AnalyticTableNumeric<T> read_numeric(char separator = ',') const;

    void write(const ttb::AnalyticTable &table, char separator = ',') const;

  private:
    std::filesystem::path _path;
    bool _has_header;
};

class CSV_IOError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace ttb
#endif