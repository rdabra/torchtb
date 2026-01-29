#ifndef IO_FILEPARQUET_H
#define IO_FILEPARQUET_H
#pragma once

#include "AnalyticTable.h"
#include "IO_File.h"

#include <ATen/core/TensorBody.h>
#include <filesystem>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/platform.h>
#include <parquet/properties.h>
#include <parquet/type_fwd.h>

namespace ttb {

class IO_FileParquet : public ttb::IO_File {
  public:
    IO_FileParquet(std::filesystem::path path) : ttb::IO_File{std::move(path)} {};

    [[nodiscard]] ttb::AnalyticTable read() const override;

    void write(const ttb::AnalyticTable &table) const override;
};

class IO_FileParquetError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace ttb
#endif
