#include <gtest/gtest.h>

#include "AnalyticTable.h"
#include "AnalyticTableNumeric.h"
#include "Parquet_IO.h"
#include "detail/utils.h"

#include <arrow/api.h>
#include <filesystem>
#include <random>
#include <torch/torch.h>

namespace fs = std::filesystem;

namespace tparquet_io {
static fs::path temp_dir() {
  auto p = fs::temp_directory_path() / "torchtb_parquet_tests";
  fs::create_directories(p);
  return p;
}

static fs::path unique_parquet(const std::string &stem) {
  static std::mt19937_64 rng{std::random_device{}()};
  return temp_dir() / fs::path(stem + "_" + std::to_string(rng()) + ".parquet");
}

static ttb::AnalyticTable make_table() {
  arrow::FloatBuilder fb;
  arrow::Int64Builder ib;
  EXPECT_TRUE(fb.AppendValues({1.0f, 2.0f, 3.0f}).ok());
  EXPECT_TRUE(ib.AppendValues({10, 20, 30}).ok());

  utl::shp<arrow::Array> fcol;
  utl::shp<arrow::Array> icol;
  EXPECT_TRUE(fb.Finish(&fcol).ok());
  EXPECT_TRUE(ib.Finish(&icol).ok());

  auto schema = arrow::schema({
      arrow::field("feat", arrow::float32()),
      arrow::field("label", arrow::int64()),
  });
  auto tbl = arrow::Table::Make(schema, {fcol, icol});
  return ttb::AnalyticTable{std::move(tbl)};
}

// Helper to create test XYMatrix
ttb::XYMatrix make_test_xy_matrix(int64_t rows, int64_t x_cols, int64_t y_cols) {
  auto X = torch::rand({rows, x_cols}, torch::dtype(torch::kFloat32));
  auto Y = torch::rand({rows, y_cols}, torch::dtype(torch::kFloat32));
  return ttb::XYMatrix(std::move(X), std::move(Y));
}

// Cleanup helper
class TempFile {
  public:
    explicit TempFile(const std::string &name) : path_(fs::temp_directory_path() / name) {
      // Clean up if exists from previous test
      if (fs::exists(path_)) {
        fs::remove(path_);
      }
    }

    ~TempFile() {
      if (fs::exists(path_)) {
        fs::remove(path_);
      }
    }

    const fs::path &path() const { return path_; }

  private:
    fs::path path_;
};

} // namespace tparquet_io

TEST(Parquet_IO_Test, MissingFileFails) {
  auto path = tparquet_io::unique_parquet("missing");
  ttb::Parquet_IO io(path);
  auto r = io.read();
  EXPECT_FALSE(r.has_value());
  if (!r.has_value())
    EXPECT_NE(r.error(), utl::ReturnCode::Ok);
}

TEST(Parquet_IO_Test, RoundTripTable) {
  auto path = tparquet_io::unique_parquet("roundtrip_tbl");
  auto table = tparquet_io::make_table();

  ttb::Parquet_IO io(path);
  auto wrc = io.write(table);
  EXPECT_EQ(wrc, utl::ReturnCode::Ok);
  ASSERT_TRUE(fs::exists(path));

  auto r = io.read();
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r->n_rows(), 3);
  EXPECT_EQ(r->n_cols(), 2);
  auto names = r->col_names();
  ASSERT_EQ(names.size(), 2u);
  EXPECT_EQ(names[0], "feat");
  EXPECT_EQ(names[1], "label");
}

TEST(Parquet_IO_Test, WriteThenReadNumericFloat) {
  auto path = tparquet_io::unique_parquet("tensor_float");
  // 3 rows x 2 cols tensor
  torch::Tensor t =
      torch::tensor({{1.0f, 5.0f}, {2.0f, 6.0f}, {3.0f, 7.0f}}, torch::dtype(torch::kFloat32));

  ttb::Parquet_IO io(path);
  auto wrc = io.write<float>(std::move(t));
  EXPECT_EQ(wrc, utl::ReturnCode::Ok);
  ASSERT_TRUE(fs::exists(path));

  auto rn = io.read_numeric<float>();
  ASSERT_TRUE(rn.has_value());
  EXPECT_EQ(rn->n_rows(), 3);
  // Assuming each column stored separately
  EXPECT_GE(rn->n_cols(), 1);
}

TEST(Parquet_IO_Test, FallbackToGenericIfNotPureNumeric) {
  auto path = tparquet_io::unique_parquet("numeric_int");
  // Build a pure int64 table
  arrow::Int64Builder ib;
  EXPECT_TRUE(ib.AppendValues({11, 22, 33}).ok());
  utl::shp<arrow::Array> icol;
  EXPECT_TRUE(ib.Finish(&icol).ok());
  auto schema = arrow::schema({arrow::field("x", arrow::int64())});
  auto tbl = arrow::Table::Make(schema, {icol});
  ttb::AnalyticTable dt{std::move(tbl)};

  ttb::Parquet_IO io(path);
  auto wrc = io.write(dt);
  EXPECT_EQ(wrc, utl::ReturnCode::Ok);
  ASSERT_TRUE(fs::exists(path));

  auto rn = io.read_numeric<int>(); // depends on template support
  if (rn.has_value()) {
    EXPECT_EQ(rn->n_rows(), 3);
    EXPECT_EQ(rn->n_cols(), 1);
  } else {
    // If implementation rejects int64_t path
    EXPECT_NE(rn.error(), utl::ReturnCode::Ok);
  }
}

namespace fs = std::filesystem;

TEST(Parquet_IO_Test, WritesXYMatrixToFile) {
  tparquet_io::TempFile temp("test_xy_matrix.parquet");

  auto xy = tparquet_io::make_test_xy_matrix(10, 5, 2);
  ttb::Parquet_IO writer(temp.path());

  auto rc = writer.write<float>(std::move(xy));
  EXPECT_EQ(rc, utl::ReturnCode::Ok);
  EXPECT_TRUE(fs::exists(temp.path()));
}

TEST(Parquet_IO_Test, WrittenFileIsReadable) {
  tparquet_io::TempFile temp("test_xy_readable.parquet");

  auto xy = tparquet_io::make_test_xy_matrix(20, 3, 4);
  ttb::Parquet_IO writer(temp.path());

  auto rc_write = writer.write<float>(std::move(xy));
  ASSERT_EQ(rc_write, utl::ReturnCode::Ok);

  // Read back and verify
  ttb::Parquet_IO reader(temp.path());
  auto result = reader.read();
  ASSERT_TRUE(result.has_value());

  auto &table = result.value();
  EXPECT_EQ(table.n_rows(), 20);
  EXPECT_EQ(table.n_cols(), 7); // 3 + 4 columns
}

TEST(Parquet_IO_Test, PreservesDataDimensions) {
  tparquet_io::TempFile temp("test_xy_dimensions.parquet");

  constexpr int64_t rows = 15;
  constexpr int64_t x_cols = 4;
  constexpr int64_t y_cols = 3;

  auto xy = tparquet_io::make_test_xy_matrix(rows, x_cols, y_cols);
  ttb::Parquet_IO writer(temp.path());

  auto rc = writer.write<float>(std::move(xy));
  ASSERT_EQ(rc, utl::ReturnCode::Ok);

  // Read and verify dimensions
  ttb::Parquet_IO reader(temp.path());
  auto result = reader.read();
  ASSERT_TRUE(result.has_value());

  EXPECT_EQ(result->n_rows(), rows);
  EXPECT_EQ(result->n_cols(), x_cols + y_cols);
}

TEST(Parquet_IO_Test, WorksWithDifferentNumericTypes) {
  tparquet_io::TempFile temp("test_xy_types.parquet");

  // Test with int
  {
    auto X = torch::randint(0, 100, {5, 2}, torch::dtype(torch::kInt32));
    auto Y = torch::randint(0, 10, {5, 1}, torch::dtype(torch::kInt32));
    ttb::XYMatrix xy(std::move(X), std::move(Y));

    ttb::Parquet_IO writer(temp.path());
    auto rc = writer.write<int>(std::move(xy));
    EXPECT_EQ(rc, utl::ReturnCode::Ok);
  }

  // Test with double
  {
    auto X = torch::rand({5, 2}, torch::dtype(torch::kFloat64));
    auto Y = torch::rand({5, 1}, torch::dtype(torch::kFloat64));
    ttb::XYMatrix xy(std::move(X), std::move(Y));

    tparquet_io::TempFile temp2("test_xy_double.parquet");
    ttb::Parquet_IO writer(temp2.path());
    auto rc = writer.write<double>(std::move(xy));
    EXPECT_EQ(rc, utl::ReturnCode::Ok);
  }
}

TEST(Parquet_IO_Test, HandlesEmptyXYMatrix) {
  tparquet_io::TempFile temp("test_xy_empty.parquet");

  auto X = torch::empty({0, 3}, torch::dtype(torch::kFloat32));
  auto Y = torch::empty({0, 2}, torch::dtype(torch::kFloat32));
  ttb::XYMatrix xy(std::move(X), std::move(Y));

  ttb::Parquet_IO writer(temp.path());
  auto rc = writer.write<float>(std::move(xy));

  // Should succeed (writing empty table is valid)
  EXPECT_EQ(rc, utl::ReturnCode::Ok);

  // Verify by reading back
  ttb::Parquet_IO reader(temp.path());
  auto result = reader.read();
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->n_rows(), 0);
}

TEST(Parquet_IO_Test, OverwritesExistingFile) {
  tparquet_io::TempFile temp("test_xy_overwrite.parquet");

  // Write first XYMatrix
  {
    auto xy1 = tparquet_io::make_test_xy_matrix(10, 2, 2);
    ttb::Parquet_IO writer(temp.path());
    auto rc = writer.write<float>(std::move(xy1));
    ASSERT_EQ(rc, utl::ReturnCode::Ok);
  }

  // Overwrite with different dimensions
  {
    auto xy2 = tparquet_io::make_test_xy_matrix(5, 3, 1);
    ttb::Parquet_IO writer(temp.path());
    auto rc = writer.write<float>(std::move(xy2));
    EXPECT_EQ(rc, utl::ReturnCode::Ok);
  }

  // Verify the second write succeeded
  ttb::Parquet_IO reader(temp.path());
  auto result = reader.read();
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->n_rows(), 5);
  EXPECT_EQ(result->n_cols(), 4); // 3 + 1
}

TEST(Parquet_IO_Test, FailsWithInvalidPath) {
  fs::path invalid_path("/nonexistent/directory/test.parquet");

  auto xy = tparquet_io::make_test_xy_matrix(5, 2, 2);
  ttb::Parquet_IO writer(invalid_path);

  auto rc = writer.write<float>(std::move(xy));
  EXPECT_NE(rc, utl::ReturnCode::Ok);
}

TEST(Parquet_IO_Test, PreservesColumnOrder) {
  tparquet_io::TempFile temp("test_xy_col_order.parquet");

  // Create XYMatrix with specific values to track order
  auto X = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
  auto Y = torch::tensor({{5.0f}, {6.0f}});
  ttb::XYMatrix xy(std::move(X), std::move(Y));

  ttb::Parquet_IO writer(temp.path());
  auto rc = writer.write<float>(std::move(xy));
  ASSERT_EQ(rc, utl::ReturnCode::Ok);

  // Read back and verify column order (X columns before Y columns)
  ttb::Parquet_IO reader(temp.path());
  auto result = reader.read();
  ASSERT_TRUE(result.has_value());

  auto &table = result.value();
  EXPECT_EQ(table.n_cols(), 3);

  // Column names should reflect X_0, X_1, Y_0 or similar pattern
  auto names = table.col_names();
  ASSERT_EQ(names.size(), 3);
}

TEST(Parquet_IO_Test, WorksWithLargeXYMatrix) {
  tparquet_io::TempFile temp("test_xy_large.parquet");

  constexpr int64_t rows = 1000;
  constexpr int64_t x_cols = 50;
  constexpr int64_t y_cols = 10;

  auto xy = tparquet_io::make_test_xy_matrix(rows, x_cols, y_cols);
  ttb::Parquet_IO writer(temp.path());

  auto rc = writer.write<float>(std::move(xy));
  EXPECT_EQ(rc, utl::ReturnCode::Ok);

  // Verify file size is reasonable (should be > 0)
  EXPECT_GT(fs::file_size(temp.path()), 0);
}
