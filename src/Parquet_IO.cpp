#include "Parquet_IO.h"
#include "AnalyticTable.h"
#include "AnalyticTableNumeric.h"
#include "Converter.h"
#include "detail/utils.h"

#include <arrow/io/api.h>
#include <arrow/type_fwd.h>
#include <expected>
#include <memory>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/platform.h>
#include <parquet/properties.h>
#include <parquet/type_fwd.h>

std::expected<ttb::AnalyticTable, utl::ReturnCode> ttb::Parquet_IO::read() const {
  auto r_infile = arrow::io::ReadableFile::Open(_path);
  if (!r_infile.ok())
    return std::unexpected(utl::map_status(r_infile.status()));

  auto r_reader =
      parquet::arrow::OpenFile(r_infile.MoveValueUnsafe(), arrow::default_memory_pool());
  if (!r_reader.ok())
    return std::unexpected(utl::map_status(r_reader.status()));

  auto reader = r_reader.MoveValueUnsafe();
  reader->set_use_threads(true);

  utl::shp<arrow::Table> table;
  auto status = reader->ReadTable(&table);
  if (!status.ok())
    return std::unexpected(utl::map_status(status));

  return ttb::AnalyticTable{std::move(table)};
}

template <utl::NumericType T>
std::expected<ttb::AnalyticTableNumeric<T>, utl::ReturnCode> ttb::Parquet_IO::read_numeric() const {
  auto r_table = this->read();
  if (!r_table)
    return std::unexpected(r_table.error());

  return ttb::AnalyticTableNumeric<T>{std::move(r_table.value())};
}

utl::ReturnCode ttb::Parquet_IO::write(const ttb::AnalyticTable &table) const {
  auto r_outfile = arrow::io::FileOutputStream::Open(_path);
  if (!r_outfile.ok())
    return utl::map_status(r_outfile.status());

  auto parquet_props = parquet::WriterProperties::Builder()
                           .compression(parquet::Compression::ZSTD)
                           ->created_by(utl::LIBRARY_NAME)
                           ->build();

  auto arrow_props = parquet::ArrowWriterProperties::Builder().store_schema()->build();

  auto status =
      parquet::arrow::WriteTable(*table.arrow_table(), arrow::default_memory_pool(),
                                 r_outfile.MoveValueUnsafe(), 1 << 20, parquet_props, arrow_props);

  std::cout << status << std::endl;
  if (!status.ok())
    return utl::map_status(status);

  return utl::ReturnCode::Ok;
};

template <utl::NumericType T>
utl::ReturnCode ttb::Parquet_IO::write(torch::Tensor &&tensor) const {
  auto r_table = ttb::Converter::analytic_table<T>(std::move(tensor));
  if (!r_table)
    return r_table.error();

  return this->write(r_table.value());
};

template <utl::NumericType T>
utl::ReturnCode ttb::Parquet_IO::write(ttb::XYMatrix &&xy_matrix) const {
  auto my_xy_matrix = std::move(xy_matrix);

  auto X = my_xy_matrix.X().clone();
  auto Y = my_xy_matrix.Y().clone();
  auto XY = torch::cat({std::move(X), std::move(Y)}, 1);

  return this->write<T>(std::move(XY));
}

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define INSTANTIATE_PARQUET_IO_TEMPLATES(T)                                                        \
  template std::expected<ttb::AnalyticTableNumeric<T>, utl::ReturnCode>                            \
  ttb::Parquet_IO::read_numeric<T>() const;                                                        \
  template utl::ReturnCode ttb::Parquet_IO::write<T>(torch::Tensor && tensor) const;               \
  template utl::ReturnCode ttb::Parquet_IO::write<T>(ttb::XYMatrix &&) const;

INSTANTIATE_PARQUET_IO_TEMPLATES(int);
INSTANTIATE_PARQUET_IO_TEMPLATES(int64_t)
INSTANTIATE_PARQUET_IO_TEMPLATES(float)
INSTANTIATE_PARQUET_IO_TEMPLATES(double)

#undef INSTANTIATE_PARQUET_IO_TEMPLATES
