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

ttb::AnalyticTable ttb::Parquet_IO::read() const {
  auto r_infile = arrow::io::ReadableFile::Open(_path);
  if (!r_infile.ok())
    throw ttb::Parquet_IOError(r_infile.status().ToString());

  auto r_reader =
      parquet::arrow::OpenFile(r_infile.MoveValueUnsafe(), arrow::default_memory_pool());
  if (!r_reader.ok())
    throw ttb::Parquet_IOError(r_reader.status().ToString());

  auto reader = r_reader.MoveValueUnsafe();
  reader->set_use_threads(true);

  utl::shp<arrow::Table> table;
  auto status = reader->ReadTable(&table);
  if (!status.ok())
    throw ttb::Parquet_IOError(status.ToString());

  return ttb::AnalyticTable{std::move(table)};
}

template <utl::NumericType T>
ttb::AnalyticTableNumeric<T> ttb::Parquet_IO::read_numeric() const {
  auto table = this->read();

  return ttb::AnalyticTableNumeric<T>{std::move(table)};
}

void ttb::Parquet_IO::write(const ttb::AnalyticTable &table) const {
  auto r_outfile = arrow::io::FileOutputStream::Open(_path);
  if (!r_outfile.ok())
    throw ttb::Parquet_IOError(r_outfile.status().ToString());

  auto parquet_props = parquet::WriterProperties::Builder()
                           .compression(parquet::Compression::ZSTD)
                           ->created_by(utl::LIBRARY_NAME)
                           ->build();

  auto arrow_props = parquet::ArrowWriterProperties::Builder().store_schema()->build();

  auto status =
      parquet::arrow::WriteTable(*table.arrow_table(), arrow::default_memory_pool(),
                                 r_outfile.MoveValueUnsafe(), 1 << 20, parquet_props, arrow_props);

  if (!status.ok())
    throw ttb::Parquet_IOError(status.ToString());
};

template <utl::NumericType T>
void ttb::Parquet_IO::write(torch::Tensor &&tensor) const {
  auto table = ttb::Converter::analytic_table<T>(std::move(tensor));

  this->write(table);
};

template <utl::NumericType T>
void ttb::Parquet_IO::write(ttb::XYMatrix &&xy_matrix) const {
  auto my_xy_matrix = std::move(xy_matrix);

  auto X = my_xy_matrix.X().clone();
  auto Y = my_xy_matrix.Y().clone();
  auto XY = torch::cat({std::move(X), std::move(Y)}, 1);

  this->write<T>(std::move(XY));
}

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define INSTANTIATE_PARQUET_IO_TEMPLATES(T)                                                        \
  template ttb::AnalyticTableNumeric<T> ttb::Parquet_IO::read_numeric<T>() const;                  \
  template void ttb::Parquet_IO::write<T>(torch::Tensor && tensor) const;                          \
  template void ttb::Parquet_IO::write<T>(ttb::XYMatrix &&) const;

INSTANTIATE_PARQUET_IO_TEMPLATES(int);
INSTANTIATE_PARQUET_IO_TEMPLATES(int64_t)
INSTANTIATE_PARQUET_IO_TEMPLATES(float)
INSTANTIATE_PARQUET_IO_TEMPLATES(double)

#undef INSTANTIATE_PARQUET_IO_TEMPLATES
