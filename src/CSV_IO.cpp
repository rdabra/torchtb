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

utl::shp<arrow::Table> read_file(const std::filesystem::path &path, bool has_header,
                                 char separator) {
  auto infile = arrow::io::ReadableFile::Open(path);
  if (!infile.ok())
    throw ttb::CSV_IOError(infile.status().ToString());

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
    throw ttb::CSV_IOError(reader.status().ToString());

  auto table = reader.MoveValueUnsafe()->Read();
  if (!table.ok())
    throw ttb::CSV_IOError(table.status().ToString());

  return table.ValueUnsafe();
}

} // namespace rread

ttb::AnalyticTable ttb::CSV_IO::read(char separator) const {
  auto resp = rread::read_file(this->_path, _has_header, separator);

  return ttb::AnalyticTable{std::move(resp)};
}

void ttb::CSV_IO::write(const ttb::AnalyticTable &table, char separator) const {
  auto r_outfile = arrow::io::FileOutputStream::Open(_path);
  if (!r_outfile.ok())
    throw CSV_IOError(r_outfile.status().ToString());

  auto opts = arrow::csv::WriteOptions::Defaults();
  opts.include_header = _has_header;
  opts.batch_size = 1024;
  opts.delimiter = separator;
  auto outfile = r_outfile.MoveValueUnsafe();

  auto st = arrow::csv::WriteCSV(*table.arrow_table(), opts, outfile.get());
  if (!st.ok())
    throw CSV_IOError(st.ToString());
}

template <utl::NumericType T>
ttb::AnalyticTableNumeric<T> ttb::CSV_IO::read_numeric(char separator) const {
  auto r_table = this->read(separator);

  return ttb::AnalyticTableNumeric<T>{std::move(r_table)};
}

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define INSTANTIATE_CSV_IO_TEMPLATES(T)                                                            \
  template ttb::AnalyticTableNumeric<T> ttb::CSV_IO::read_numeric<T>(char) const;

INSTANTIATE_CSV_IO_TEMPLATES(int);
INSTANTIATE_CSV_IO_TEMPLATES(int64_t)
INSTANTIATE_CSV_IO_TEMPLATES(float)
INSTANTIATE_CSV_IO_TEMPLATES(double)

#undef INSTANTIATE_CSV_IO_TEMPLATES