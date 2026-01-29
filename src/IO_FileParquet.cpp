#include "IO_FileParquet.h"
#include "AnalyticTable.h"
#include "AnalyticTableNumeric.h"
#include "Converter.h"
#include "detail/utils.h"

#include <arrow/io/api.h>
#include <arrow/type_fwd.h>
#include <memory>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/platform.h>
#include <parquet/properties.h>
#include <parquet/type_fwd.h>

ttb::AnalyticTable ttb::IO_FileParquet::read() const {
  auto r_infile = arrow::io::ReadableFile::Open(_path);
  if (!r_infile.ok())
    throw ttb::IO_FileParquetError(r_infile.status().ToString());

  auto r_reader =
      parquet::arrow::OpenFile(r_infile.MoveValueUnsafe(), arrow::default_memory_pool());
  if (!r_reader.ok())
    throw ttb::IO_FileParquetError(r_reader.status().ToString());

  auto reader = r_reader.MoveValueUnsafe();
  reader->set_use_threads(true);

  utl::shp<arrow::Table> table;
  auto status = reader->ReadTable(&table);
  if (!status.ok())
    throw ttb::IO_FileParquetError(status.ToString());

  return ttb::AnalyticTable{std::move(table)};
}

void ttb::IO_FileParquet::write(const ttb::AnalyticTable &table) const {
  auto r_outfile = arrow::io::FileOutputStream::Open(_path);
  if (!r_outfile.ok())
    throw ttb::IO_FileParquetError(r_outfile.status().ToString());

  auto parquet_props = parquet::WriterProperties::Builder()
                           .compression(parquet::Compression::ZSTD)
                           ->created_by(utl::LIBRARY_NAME)
                           ->build();

  auto arrow_props = parquet::ArrowWriterProperties::Builder().store_schema()->build();

  auto status =
      parquet::arrow::WriteTable(*table.arrow_table(), arrow::default_memory_pool(),
                                 r_outfile.MoveValueUnsafe(), 1 << 20, parquet_props, arrow_props);

  if (!status.ok())
    throw ttb::IO_FileParquetError(status.ToString());
};
