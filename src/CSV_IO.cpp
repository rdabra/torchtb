#include "CSV_IO.h"
#include "AnalyticTable.h"
#include "AnalyticTableNumeric.h"
#include "detail/utils.h"

#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/csv/options.h>
#include <arrow/csv/reader.h>
#include <arrow/csv/writer.h>
#include <arrow/io/api.h>
#include <arrow/io/file.h>
#include <arrow/io/type_fwd.h>
#include <arrow/table.h>
#include <expected>
#include <memory>

namespace rread {

std::expected<utl::shp<arrow::Table>, utl::ReturnCode> read_file(const std::filesystem::path &path,
                                                                 bool has_header, char separator) {
  auto infile = arrow::io::ReadableFile::Open(path);
  if (!infile.ok())
    return std::unexpected(utl::map_status(infile.status()));

  auto read_opts = arrow::csv::ReadOptions::Defaults();
  auto parse_opts = arrow::csv::ParseOptions::Defaults();
  auto convert_opts = arrow::csv::ConvertOptions::Defaults();

  parse_opts.delimiter = separator;
  read_opts.autogenerate_column_names = !has_header;
  read_opts.use_threads = true;

  auto reader =
      arrow::csv::TableReader::Make(arrow::io::default_io_context(), infile.MoveValueUnsafe(),
                                    read_opts, parse_opts, convert_opts);
  if (!reader.ok())
    return std::unexpected(utl::map_status(reader.status()));

  auto table = reader.MoveValueUnsafe()->Read();
  if (!table.ok())
    return std::unexpected(utl::map_status(table.status()));

  return table.ValueUnsafe();
}

} // namespace rread

std::expected<ttb::AnalyticTable, utl::ReturnCode> ttb::CSV_IO::read(char separator) const {
  auto resp = rread::read_file(this->_path, _has_header, separator);
  if (!resp)
    return std::unexpected(resp.error());

  return ttb::AnalyticTable{std::move(resp.value())};
}

utl::ReturnCode ttb::CSV_IO::write(const ttb::AnalyticTable &table, char separator) const {
  auto r_outfile = arrow::io::FileOutputStream::Open(_path);
  if (!r_outfile.ok())
    return utl::map_status(r_outfile.status());

  auto opts = arrow::csv::WriteOptions::Defaults();
  opts.include_header = _has_header;
  opts.batch_size = 1024;
  opts.delimiter = separator;
  auto outfile = r_outfile.MoveValueUnsafe();

  auto resp = arrow::csv::WriteCSV(*table.arrow_table(), opts, outfile.get());
  return utl::map_status(resp);
}

template <utl::NumericType T>
std::expected<ttb::AnalyticTableNumeric<T>, utl::ReturnCode>
ttb::CSV_IO::read_numeric(char separator) const {
  auto r_table = this->read(separator);
  if (!r_table)
    return std::unexpected(r_table.error());

  return ttb::AnalyticTableNumeric<T>{std::move(r_table.value())};
}

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define INSTANTIATE_CSV_IO_TEMPLATES(T)                                                            \
  template std::expected<ttb::AnalyticTableNumeric<T>, utl::ReturnCode>                            \
  ttb::CSV_IO::read_numeric<T>(char) const;

INSTANTIATE_CSV_IO_TEMPLATES(int);
INSTANTIATE_CSV_IO_TEMPLATES(int64_t)
INSTANTIATE_CSV_IO_TEMPLATES(float)
INSTANTIATE_CSV_IO_TEMPLATES(double)

#undef INSTANTIATE_CSV_IO_TEMPLATES