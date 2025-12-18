#include <gtest/gtest.h>

#include "AnalyticTableNumeric.h"
#include "CSV_IO.h"
#include "Converter.h"
#include "Parquet_IO.h"
#include "detail/utils.h"

#include <arrow/api.h>
#include <filesystem>
#include <fstream>
#include <random>
#include <torch/torch.h>

namespace fs = std::filesystem;

namespace tconverter {
static fs::path temp_dir() {
  auto p = fs::temp_directory_path() / "torchtb_converter_tests";
  fs::create_directories(p);
  return p;
}

static fs::path unique_file(const std::string &stem, const std::string &ext) {
  static std::mt19937_64 rng{std::random_device{}()};
  return temp_dir() / fs::path(stem + "_" + std::to_string(rng()) + ext);
}

static void write_csv(const fs::path &p, const std::string &content) {
  std::ofstream ofs(p);
  ofs << content;
  ofs.close();
}

static ttb::AnalyticTableNumeric<float> make_numeric_table_float(int64_t rows = 3, int cols = 2) {
  std::vector<utl::shp<arrow::Array>> arrays;

  for (int c = 0; c < cols; ++c) {
    arrow::FloatBuilder fb;
    for (int64_t r = 0; r < rows; ++r) {
      EXPECT_TRUE(fb.Append(static_cast<float>(r * 10 + c)).ok());
    }
    utl::shp<arrow::Array> arr;
    EXPECT_TRUE(fb.Finish(&arr).ok());
    arrays.push_back(arr);
  }

  std::vector<utl::shp<arrow::Field>> fields;
  for (int c = 0; c < cols; ++c) {
    fields.push_back(arrow::field("col_" + std::to_string(c), arrow::float32()));
  }

  auto schema = arrow::schema(fields);
  auto table = arrow::Table::Make(schema, arrays);
  return ttb::AnalyticTableNumeric<float>{std::move(table)};
}
} // namespace tconverter

TEST(Converter_Test, ConvertsDataTableNumericFloat) {
  auto table = tconverter::make_numeric_table_float(5, 3);
  auto tensor = ttb::Converter::torch_tensor(std::move(table));

  EXPECT_EQ(tensor.dim(), 2);
  EXPECT_EQ(tensor.size(0), 5);
  EXPECT_EQ(tensor.size(1), 3);
  EXPECT_EQ(tensor.scalar_type(), torch::kFloat32);
}

TEST(Converter_Test, ConvertsDataTableNumericDouble) {
  std::vector<utl::shp<arrow::Array>> arrays;
  arrow::DoubleBuilder db;

  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(db.Append(static_cast<double>(i) * 2.5).ok());
  }

  utl::shp<arrow::Array> arr;
  EXPECT_TRUE(db.Finish(&arr).ok());
  arrays.push_back(arr);

  auto schema = arrow::schema({arrow::field("x", arrow::float64())});
  auto table = arrow::Table::Make(schema, arrays);
  ttb::AnalyticTableNumeric<double> dt{std::move(table)};

  auto tensor = ttb::Converter::torch_tensor(std::move(dt));
  EXPECT_EQ(tensor.dim(), 2);
  EXPECT_EQ(tensor.size(0), 4);
  EXPECT_EQ(tensor.size(1), 1);
  EXPECT_EQ(tensor.scalar_type(), torch::kFloat64);
}

TEST(Converter_Test, ReadsCSVAndConvertToTensor) {
  auto path = tconverter::unique_file("csv_to_tensor", ".csv");
  tconverter::write_csv(path, "a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n");

  ttb::CSV_IO reader(path, true);
  auto tensor = ttb::Converter::torch_tensor<float>(std::move(reader));
  EXPECT_EQ(tensor.dim(), 2);
  EXPECT_EQ(tensor.size(0), 3);
  EXPECT_EQ(tensor.size(1), 2);

  fs::remove(path);
}

TEST(Converter_Test, FailsOnMissingFile) {
  auto path = tconverter::unique_file("missing_csv", ".csv");

  ttb::CSV_IO reader(path, true);
  EXPECT_THROW(ttb::Converter::torch_tensor<float>(std::move(reader)), ttb::CSV_IOError);
}

TEST(Converter_Test, ReadsParquetAndConvertsToTensor) {
  auto path = tconverter::unique_file("pq_to_tensor", ".parquet");

  // Write a simple parquet file first
  arrow::FloatBuilder fb;
  EXPECT_TRUE(fb.AppendValues({1.0f, 2.0f, 3.0f}).ok());
  utl::shp<arrow::Array> arr;
  EXPECT_TRUE(fb.Finish(&arr).ok());

  auto schema = arrow::schema({arrow::field("value", arrow::float32())});
  auto table = arrow::Table::Make(schema, {arr});

  ttb::AnalyticTable dt{std::move(table)};
  ttb::Parquet_IO writer(path);
  writer.write(dt);

  // Now read it back
  ttb::Parquet_IO reader(path);
  auto tensor = ttb::Converter::torch_tensor<float>(std::move(reader));
  EXPECT_EQ(tensor.dim(), 2);
  EXPECT_EQ(tensor.size(0), 3);

  fs::remove(path);
}

TEST(Converter_Test, FailsOnMissingFileFromParquet) {
  auto path = tconverter::unique_file("missing_pq", ".parquet");

  ttb::Parquet_IO reader(path);
  EXPECT_THROW(ttb::Converter::torch_tensor<float>(std::move(reader)), ttb::Parquet_IOError);
}

TEST(Converter_Test, ConvertsTensorToDataTableFloat) {
  auto tensor =
      torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}}, torch::dtype(torch::kFloat32));

  auto table = ttb::Converter::analytic_table<float>(std::move(tensor));
  EXPECT_EQ(table.n_rows(), 3);
  EXPECT_EQ(table.n_cols(), 2);
}

TEST(Converter_Test, FailsOnNon2DTensor) {
  auto tensor = torch::tensor({1.0f, 2.0f, 3.0f}, torch::dtype(torch::kFloat32));

  EXPECT_THROW(auto result = ttb::Converter::analytic_table<float>(std::move(tensor)),
               ttb::ConverterError);
}

TEST(Converter_Test, FailsOnNonFloat32Tensor) {
  auto tensor = torch::tensor({{1, 2}, {3, 4}}, torch::dtype(torch::kInt64));
  auto table = ttb::Converter::analytic_table<float>(std::move(tensor));
  EXPECT_EQ(table.arrow_dtype(), arrow::float32());
}

TEST(Converter_Test, RoundTripPreservesData) {
  auto original_table = tconverter::make_numeric_table_float(4, 3);
  auto tensor = ttb::Converter::torch_tensor(std::move(original_table));

  auto recovered_table = ttb::Converter::analytic_table<float>(std::move(tensor));
  EXPECT_EQ(recovered_table.n_rows(), 4);
  EXPECT_EQ(recovered_table.n_cols(), 3);
}
