#include "IO_FileCSV.h"
#include "AnalyticTable.h"
#include "detail/ttbutils.h"

#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/csv/options.h>
#include <arrow/csv/reader.h>
#include <arrow/csv/writer.h>
#include <arrow/io/api.h>
#include <arrow/io/file.h>
#include <arrow/io/type_fwd.h>
#include <arrow/table.h>
#include <memory>

namespace rread {

ttb::utl::shp<arrow::Table> read_file(const std::filesystem::path &path, bool has_header,
                                      char separator) {
  auto infile = arrow::io::ReadableFile::Open(path);
  if (!infile.ok())
    throw ttb::IO_FileCSVError(infile.status().ToString());

  auto read_opts = arrow::csv::ReadOptions::Defaults();
  auto parse_opts = arrow::csv::ParseOptions::Defaults();
  auto convert_opts = arrow::csv::ConvertOptions::Defaults();

  parse_opts.delimiter = separator;
  read_opts.autogenerate_column_names = !has_header;
  read_opts.use_threads = true;
  convert_opts.null_values.emplace_back("NULL");
  convert_opts.null_values.emplace_back("Null");
  convert_opts.null_values.emplace_back("null");

  auto reader =
      arrow::csv::TableReader::Make(arrow::io::default_io_context(), infile.MoveValueUnsafe(),
                                    read_opts, parse_opts, convert_opts);
  if (!reader.ok())
    throw ttb::IO_FileCSVError(reader.status().ToString());

  auto table = reader.MoveValueUnsafe()->Read();
  if (!table.ok())
    throw ttb::IO_FileCSVError(table.status().ToString());

  return table.ValueUnsafe();
}

} // namespace rread

ttb::AnalyticTable ttb::IO_FileCSV::read() const {
  auto resp = rread::read_file(this->_path, _has_header, _separator);

  return ttb::AnalyticTable{std::move(resp)};
}

void ttb::IO_FileCSV::write(const ttb::AnalyticTable &table) const {
  auto r_outfile = arrow::io::FileOutputStream::Open(_path);
  if (!r_outfile.ok())
    throw IO_FileCSVError(r_outfile.status().ToString());

  auto opts = arrow::csv::WriteOptions::Defaults();
  opts.include_header = _has_header;
  opts.batch_size = 1024;
  opts.delimiter = _separator;
  auto outfile = r_outfile.MoveValueUnsafe();

  auto st = arrow::csv::WriteCSV(*table.arrow_table(), opts, outfile.get());
  if (!st.ok())
    throw IO_FileCSVError(st.ToString());
}
