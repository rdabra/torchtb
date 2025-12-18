#include <gtest/gtest.h>

#include "AnalyticTable.h"
#include "CSV_IO.h"

#include <arrow/api.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace tcsv_io {
fs::path temp_dir() {
  auto base = fs::temp_directory_path() / "torchtb_csvio_tests";
  fs::create_directories(base);
  return base;
}

fs::path unique_path(const std::string &stem, const std::string &ext = ".csv") {
  return temp_dir() / fs::path(stem + "_" + std::to_string(::getpid()) + "_" +
                               std::to_string(std::rand()) + ext);
}

void write_text(const fs::path &p, const std::string &content) {
  std::ofstream ofs(p, std::ios::binary);
  ASSERT_TRUE(ofs.is_open()) << "Failed to open file for writing: " << p;
  ofs << content;
  ofs.close();
  ASSERT_TRUE(fs::exists(p));
}

ttb::AnalyticTable make_simple_table() {
  // Two columns: float32 and int64, 3 rows
  arrow::FloatBuilder fb;
  arrow::Int64Builder ib;

  EXPECT_TRUE(fb.AppendValues({1.5f, 2.5f, 3.5f}).ok());
  EXPECT_TRUE(ib.AppendValues({10, 20, 30}).ok());

  utl::shp<arrow::Array> fcol;
  utl::shp<arrow::Array> icol;
  EXPECT_TRUE(fb.Finish(&fcol).ok());
  EXPECT_TRUE(ib.Finish(&icol).ok());

  auto schema =
      arrow::schema({arrow::field("f32", arrow::float32()), arrow::field("i64", arrow::int64())});
  auto table = arrow::Table::Make(schema, {fcol, icol});
  return ttb::AnalyticTable{std::move(table)};
}
} // namespace tcsv_io

TEST(CSV_IO_Test, WithHeader_Succeeds) {
  auto path = tcsv_io::unique_path("read_with_header");
  tcsv_io::write_text(path, "a,b\n1,2\n3,4\n");

  ttb::CSV_IO reader(path, /*has_header=*/true);
  auto res = reader.read();
  EXPECT_EQ(res.n_rows(), 2);
  EXPECT_EQ(res.n_cols(), 2);

  auto names = res.col_names();
  ASSERT_EQ(names.size(), 2u);
  EXPECT_EQ(names[0], "a");
  EXPECT_EQ(names[1], "b");

  fs::remove(path);
}

TEST(CSV_IO_Test, WithoutHeader_Succeeds) {
  auto path = tcsv_io::unique_path("read_no_header");
  tcsv_io::write_text(path, "1,2\n3,4\n");

  ttb::CSV_IO reader(path, /*has_header=*/false);
  auto res = reader.read();
  EXPECT_EQ(res.n_rows(), 2);
  EXPECT_EQ(res.n_cols(), 2);

  fs::remove(path);
}

TEST(CSV_IO_Test, MissingFile_Fails) {
  auto path = tcsv_io::unique_path("missing_file");
  // Do not create the file

  ttb::CSV_IO reader(path, true);
  EXPECT_THROW(auto x = reader.read(), ttb::CSV_IOError);
}

TEST(CSV_IO_Test, WritesFileAndIsReadable) {
  auto path = tcsv_io::unique_path("write_roundtrip");

  auto table = tcsv_io::make_simple_table();
  ttb::CSV_IO writer(path, /*has_header=*/true);
  writer.write(table, ',');
  ASSERT_TRUE(fs::exists(path));

  // Optional round-trip check
  ttb::CSV_IO reader(path, /*has_header=*/true);
  auto res = reader.read();
  EXPECT_EQ(res.n_rows(), 3);
  EXPECT_EQ(res.n_cols(), 2);

  auto names = res.col_names();
  ASSERT_EQ(names.size(), 2u);
  EXPECT_EQ(names[0], "f32");
  EXPECT_EQ(names[1], "i64");

  fs::remove(path);
}
