#ifndef IO_FILECSV_H
#define IO_FILECSV_H

#include "AnalyticTable.h"
#include "IO_File.h"

#include <arrow/table.h>
#include <filesystem>
#include <utility>

#pragma once

namespace ttb {

class IO_FileCSV : public ttb::IO_File {
  public:
    IO_FileCSV(std::filesystem::path path, bool has_header = true, char separator = ',')
        : ttb::IO_File{std::move(path)}, _has_header{has_header}, _separator{separator} {};

    [[nodiscard]] ttb::AnalyticTable read() const override;

    void write(const ttb::AnalyticTable &table) const override;

    [[nodiscard]] char separator() const { return this->_separator; }
    void set_separator(char separator) { this->_separator = separator; }

  private:
    bool _has_header;
    char _separator;
};

class IO_FileCSVError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace ttb
#endif
