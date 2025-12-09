#ifndef CSVREADER_H
#define CSVREADER_H

#include "AnalyticTable.h"
#include "AnalyticTableNumeric.h"
#include "detail/utils.h"

#include <arrow/table.h>
#include <expected>
#include <filesystem>
#include <utility>

#pragma once

namespace ttb {

class CSV_IO {
  public:
    CSV_IO(std::filesystem::path path, bool has_header = true)
        : _path{std::move(path)}, _has_header{has_header} {};

    std::expected<ttb::AnalyticTable, utl::ReturnCode> read(char separator = ',') const;

    template <utl::NumericType T>
    std::expected<ttb::AnalyticTableNumeric<T>, utl::ReturnCode>
    read_numeric(char separator = ',') const;

    [[nodiscard]] utl::ReturnCode write(const ttb::AnalyticTable &table,
                                        char separator = ',') const;

  private:
    std::filesystem::path _path;
    bool _has_header;
};

} // namespace ttb
#endif