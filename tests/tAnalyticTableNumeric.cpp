#include <arrow/api.h>
#include <arrow/type_traits.h>
#include <gtest/gtest.h>

#include "AnalyticTable.h"
#include "AnalyticTableNumeric.h"
#include "detail/utils.h"

using ttb::AnalyticTableNumeric;
using ttb::TbNumeric;

namespace {

// Helper to create a simple Arrow table with numeric data
template <typename T>
std::shared_ptr<arrow::Table> make_arrow_table(int64_t rows, int cols) {
  std::vector<std::shared_ptr<arrow::Array>> columns;
  std::vector<std::shared_ptr<arrow::Field>> fields;

  using BuilderType = typename arrow::CTypeTraits<T>::BuilderType;

  for (int c = 0; c < cols; ++c) {
    BuilderType builder;
    for (int64_t r = 0; r < rows; ++r) {
      EXPECT_TRUE(builder.Append(static_cast<T>(r * 10 + c)).ok());
    }
    std::shared_ptr<arrow::Array> array;
    EXPECT_TRUE(builder.Finish(&array).ok());
    columns.push_back(array);
    fields.push_back(
        arrow::field("col" + std::to_string(c), arrow::CTypeTraits<T>::type_singleton()));
  }

  auto schema = arrow::schema(fields);
  return arrow::Table::Make(schema, columns);
}

// Helper to create mixed-type Arrow table (for to_dtype testing)
template <typename T>
std::shared_ptr<arrow::Table> make_mixed_type_table() {
  arrow::Int32Builder int_builder;
  arrow::FloatBuilder float_builder;

  EXPECT_TRUE(int_builder.AppendValues({1, 2, 3}).ok());
  EXPECT_TRUE(float_builder.AppendValues({1.5f, 2.5f, 3.5f}).ok());

  std::shared_ptr<arrow::Array> int_array, float_array;
  EXPECT_TRUE(int_builder.Finish(&int_array).ok());
  EXPECT_TRUE(float_builder.Finish(&float_array).ok());

  auto schema = arrow::schema(
      {arrow::field("int_col", arrow::int32()), arrow::field("float_col", arrow::float32())});

  return arrow::Table::Make(schema, {int_array, float_array});
}

} // namespace

// ------------ Constructor tests ------------

TEST(AnalyticTableNumeric_Test, AcceptsArrowTableAndConvertsDtype) {
  auto table = make_arrow_table<int>(5, 3);
  TbNumeric<int> tb(std::move(table));

  EXPECT_EQ(tb.n_rows(), 5);
  EXPECT_EQ(tb.n_cols(), 3);

  // Verify all columns have int32 type
  auto dtypes = tb.col_dtypes();
  for (const auto &dtype : dtypes) {
    EXPECT_EQ(dtype, "int32");
  }
}

TEST(AnalyticTableNumeric_Test, AcceptsAnalyticTableAndConvertsDtype) {
  auto table = make_arrow_table<float>(4, 2);
  ttb::AnalyticTable base_table{std::move(table)};

  TbNumeric<float> tb(std::move(base_table));

  EXPECT_EQ(tb.n_rows(), 4);
  EXPECT_EQ(tb.n_cols(), 2);

  auto dtypes = tb.col_dtypes();
  for (const auto &dtype : dtypes) {
    EXPECT_EQ(dtype, "float");
  }
}

TEST(AnalyticTableNumeric_Test, ConvertsMixedTypesToTargetType) {
  auto mixed_table = make_mixed_type_table<float>();

  TbNumeric<float> tb(std::move(mixed_table));

  EXPECT_EQ(tb.n_rows(), 3);
  EXPECT_EQ(tb.n_cols(), 2);

  // All columns should be converted to float
  auto dtypes = tb.col_dtypes();
  for (const auto &dtype : dtypes) {
    EXPECT_EQ(dtype, "float");
  }
}

TEST(AnalyticTableNumeric_Test, WorksWithAllNumericTypes) {
  {
    auto t = make_arrow_table<int>(2, 1);
    TbNumeric<int> tb(std::move(t));
    EXPECT_EQ(tb.col_dtypes()[0], "int32");
  }
  {
    auto t = make_arrow_table<int64_t>(2, 1);
    TbNumeric<int64_t> tb(std::move(t));
    EXPECT_EQ(tb.col_dtypes()[0], "int64");
  }
  {
    auto t = make_arrow_table<float>(2, 1);
    TbNumeric<float> tb(std::move(t));
    EXPECT_EQ(tb.col_dtypes()[0], "float");
  }
  {
    auto t = make_arrow_table<double>(2, 1);
    TbNumeric<double> tb(std::move(t));
    EXPECT_EQ(tb.col_dtypes()[0], "double");
  }
}

// ------------ make_numeric_table static factory ------------

TEST(AnalyticTableNumeric_Test, CreatesTableFromMap) {
  std::unordered_map<std::string, std::vector<float>> data = {{"feature1", {1.0f, 2.0f, 3.0f}},
                                                              {"feature2", {4.0f, 5.0f, 6.0f}},
                                                              {"feature3", {7.0f, 8.0f, 9.0f}}};

  auto aux = TbNumeric<float>::make_numeric_table(std::move(data));
  TbNumeric<float> tb{std::move(aux)};

  EXPECT_EQ(tb.n_rows(), 3);
  EXPECT_EQ(tb.n_cols(), 3);

  auto names = tb.col_names();
  EXPECT_EQ(names.size(), 3);
  EXPECT_TRUE(std::find(names.begin(), names.end(), "feature1") != names.end());
  EXPECT_TRUE(std::find(names.begin(), names.end(), "feature2") != names.end());
  EXPECT_TRUE(std::find(names.begin(), names.end(), "feature3") != names.end());
}

TEST(AnalyticTableNumeric_Test, HandlesEmptyVectors) {
  std::unordered_map<std::string, std::vector<int>> data = {{"col1", {}}, {"col2", {}}};

  auto aux = TbNumeric<int>::make_numeric_table(std::move(data));
  TbNumeric<float> tb{std::move(aux)};

  EXPECT_EQ(tb.n_rows(), 0);
  EXPECT_EQ(tb.n_cols(), 2);
}

TEST(AnalyticTableNumeric_Test, WorksWithAllTypes) {
  {
    std::unordered_map<std::string, std::vector<int>> data = {{"a", {1, 2}}};
    auto aux = TbNumeric<int>::make_numeric_table(std::move(data));
    TbNumeric<int> tb{std::move(aux)};
    EXPECT_EQ(tb.n_rows(), 2);
  }
  {
    std::unordered_map<std::string, std::vector<int64_t>> data = {{"a", {1, 2}}};
    auto aux = TbNumeric<int64_t>::make_numeric_table(std::move(data));
    TbNumeric<int64_t> tb{std::move(aux)};
    EXPECT_EQ(tb.n_rows(), 2);
  }
  {
    std::unordered_map<std::string, std::vector<float>> data = {{"a", {1.0f, 2.0f}}};
    auto aux = TbNumeric<float>::make_numeric_table(std::move(data));
    TbNumeric<float> tb{std::move(aux)};
    EXPECT_EQ(tb.n_rows(), 2);
  }
  {
    std::unordered_map<std::string, std::vector<double>> data = {{"a", {1.0, 2.0}}};
    auto aux = TbNumeric<double>::make_numeric_table(std::move(data));
    TbNumeric<double> tb{std::move(aux)};
    EXPECT_EQ(tb.n_rows(), 2);
  }
}

// ------------ argmax tests ------------

TEST(AnalyticTableNumeric_Test, ReturnsMaxIndicesAlongRows) {
  std::unordered_map<std::string, std::vector<float>> data = {
      {"c0", {1.0f, 5.0f, 2.0f}}, {"c1", {3.0f, 1.0f, 4.0f}}, {"c2", {2.0f, 3.0f, 6.0f}}};

  auto aux = TbNumeric<float>::make_numeric_table(std::move(data));
  TbNumeric<float> tb{std::move(aux)};
  auto result = tb.argmax(ttb::Axis::ROW);

  // For each row, find the column index with max value
  // Row 0: max is c1 (3.0)
  // Row 1: max is c0 (5.0)
  // Row 2: max is c2 (6.0)
  EXPECT_EQ(result.size(), 3);
}

TEST(AnalyticTableNumeric_Test, ReturnsMaxIndicesAlongColumns) {
  std::unordered_map<std::string, std::vector<float>> data = {
      {"c0", {1.0f, 5.0f, 2.0f}}, {"c1", {3.0f, 1.0f, 4.0f}}, {"c2", {2.0f, 3.0f, 6.0f}}};

  auto aux = TbNumeric<float>::make_numeric_table(std::move(data));
  TbNumeric<float> tb{std::move(aux)};
  auto result = tb.argmax(ttb::Axis::COLUMN);

  // For each column, find the row index with max value
  // c0: max at row 1 (5.0)
  // c1: max at row 2 (4.0)
  // c2: max at row 2 (6.0)
  EXPECT_EQ(result.size(), 3);
}

TEST(AnalyticTableNumeric_Test, WorksWithSingleRow) {
  std::unordered_map<std::string, std::vector<int>> data = {{"a", {1}}, {"b", {5}}, {"c", {3}}};

  auto aux = TbNumeric<int>::make_numeric_table(std::move(data));
  TbNumeric<int> tb{std::move(aux)};
  auto result = tb.argmax(ttb::Axis::COLUMN);

  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], 1); // Column "b" has max value (5)
}

TEST(AnalyticTableNumeric_Test, WorksWithSingleColumn) {
  std::unordered_map<std::string, std::vector<double>> data = {{"only_col", {1.5, 3.7, 2.1}}};

  auto aux = TbNumeric<double>::make_numeric_table(std::move(data));
  TbNumeric<double> tb{std::move(aux)};
  auto result = tb.argmax(ttb::Axis::ROW);

  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], 1); // Row 1 has max value (3.7)
}

// ------------ one_hot_expand override ------------

TEST(AnalyticTableNumeric_Test, ExpandsIntegerColumn) {
  std::unordered_map<std::string, std::vector<int>> data = {{"category", {0, 1, 0, 2}},
                                                            {"value", {10, 20, 30, 40}}};

  auto aux = TbNumeric<int>::make_numeric_table(std::move(data));
  TbNumeric<int> tb{std::move(aux)};
  int original_cols = tb.n_cols();

  auto rc = tb.one_hot_expand(0); // Expand "category" column
  EXPECT_EQ(rc, utl::ReturnCode::Ok);

  // Should add one-hot columns and remove original
  EXPECT_GT(tb.n_cols(), original_cols);
}

TEST(AnalyticTableNumeric_Test, FailsOnInvalidIndex) {
  std::unordered_map<std::string, std::vector<float>> data = {{"col", {1.0f, 2.0f}}};

  auto aux = TbNumeric<float>::make_numeric_table(std::move(data));
  TbNumeric<float> tb{std::move(aux)};

  EXPECT_THROW(tb.one_hot_expand(-1), std::runtime_error);
  EXPECT_THROW(tb.one_hot_expand(99), std::runtime_error);
}

// ------------ Type alias tests ------------

TEST(AnalyticTableNumeric_Test, UsesShortAliases) {
  std::unordered_map<std::string, std::vector<int>> data = {{"a", {1, 2}}};

  ttb::TbInt tb_int = ttb::TbInt::make_numeric_table(std::move(data));
  EXPECT_EQ(tb_int.n_rows(), 2);

  std::unordered_map<std::string, std::vector<int64_t>> data2 = {{"a", {1, 2}}};
  ttb::TbLong tb_long = ttb::TbLong::make_numeric_table(std::move(data2));
  EXPECT_EQ(tb_long.n_rows(), 2);

  std::unordered_map<std::string, std::vector<float>> data3 = {{"a", {1.0f, 2.0f}}};
  ttb::TbFloat tb_float = ttb::TbFloat::make_numeric_table(std::move(data3));
  EXPECT_EQ(tb_float.n_rows(), 2);

  std::unordered_map<std::string, std::vector<double>> data4 = {{"a", {1.0, 2.0}}};
  ttb::TbDouble tb_double = ttb::TbDouble::make_numeric_table(std::move(data4));
  EXPECT_EQ(tb_double.n_rows(), 2);
}

// -------- Constructor from unordered_map --------

TEST(AnalyticTableNumeric_Test, CreatesTableFromMap1) {
  std::unordered_map<std::string, std::vector<int>> data{
      {"col1", {1, 2, 3}},
      {"col2", {4, 5, 6}},
  };

  TbNumeric<int> tb(std::move(data));

  EXPECT_EQ(tb.n_rows(), 3);
  EXPECT_EQ(tb.n_cols(), 2);
}

TEST(AnalyticTableNumeric_Test, PreservesColumnNames) {
  std::unordered_map<std::string, std::vector<float>> data{
      {"feature_a", {1.0f, 2.0f}},
      {"feature_b", {3.0f, 4.0f}},
      {"feature_c", {5.0f, 6.0f}},
  };

  TbNumeric<float> tb(std::move(data));

  auto names = tb.col_names();
  ASSERT_EQ(names.size(), 3);
  EXPECT_NE(std::find(names.begin(), names.end(), "feature_a"), names.end());
  EXPECT_NE(std::find(names.begin(), names.end(), "feature_b"), names.end());
  EXPECT_NE(std::find(names.begin(), names.end(), "feature_c"), names.end());
}

TEST(AnalyticTableNumeric_Test, ConvertsToCorrectDtype) {
  {
    std::unordered_map<std::string, std::vector<int>> data{{"a", {1, 2}}};
    TbNumeric<int> tb(std::move(data));
    auto dtypes = tb.col_dtypes();
    ASSERT_EQ(dtypes.size(), 1);
    EXPECT_EQ(dtypes[0], "int32");
  }

  {
    std::unordered_map<std::string, std::vector<int64_t>> data{{"a", {1, 2}}};
    TbNumeric<int64_t> tb(std::move(data));
    auto dtypes = tb.col_dtypes();
    ASSERT_EQ(dtypes.size(), 1);
    EXPECT_EQ(dtypes[0], "int64");
  }

  {
    std::unordered_map<std::string, std::vector<float>> data{{"a", {1.0f, 2.0f}}};
    TbNumeric<float> tb(std::move(data));
    auto dtypes = tb.col_dtypes();
    ASSERT_EQ(dtypes.size(), 1);
    EXPECT_EQ(dtypes[0], "float");
  }

  {
    std::unordered_map<std::string, std::vector<double>> data{{"a", {1.0, 2.0}}};
    TbNumeric<double> tb(std::move(data));
    auto dtypes = tb.col_dtypes();
    ASSERT_EQ(dtypes.size(), 1);
    EXPECT_EQ(dtypes[0], "double");
  }
}

TEST(AnalyticTableNumeric_Test, HandlesEmptyVectors1) {
  std::unordered_map<std::string, std::vector<int>> data{
      {"empty1", {}},
      {"empty2", {}},
  };

  TbNumeric<int> tb(std::move(data));

  EXPECT_EQ(tb.n_rows(), 0);
  EXPECT_EQ(tb.n_cols(), 2);
}

TEST(AnalyticTableNumeric_Test, HandlesSingleColumn) {
  std::unordered_map<std::string, std::vector<double>> data{
      {"only_col", {1.5, 2.5, 3.5}},
  };

  TbNumeric<double> tb(std::move(data));

  EXPECT_EQ(tb.n_rows(), 3);
  EXPECT_EQ(tb.n_cols(), 1);

  auto names = tb.col_names();
  ASSERT_EQ(names.size(), 1);
  EXPECT_EQ(names[0], "only_col");
}

TEST(AnalyticTableNumeric_Test, HandlesSingleRow) {
  std::unordered_map<std::string, std::vector<float>> data{
      {"x", {1.0f}},
      {"y", {2.0f}},
      {"z", {3.0f}},
  };

  TbNumeric<float> tb(std::move(data));

  EXPECT_EQ(tb.n_rows(), 1);
  EXPECT_EQ(tb.n_cols(), 3);
}

TEST(AnalyticTableNumeric_Test, HandlesLargeDataset) {
  std::unordered_map<std::string, std::vector<int64_t>> data;
  constexpr int num_cols = 10;
  constexpr int num_rows = 1000;

  for (int c = 0; c < num_cols; ++c) {
    std::vector<int64_t> col_data(num_rows);
    for (int r = 0; r < num_rows; ++r) {
      col_data[r] = r * num_cols + c;
    }
    data["col_" + std::to_string(c)] = std::move(col_data);
  }

  TbNumeric<int64_t> tb(std::move(data));

  EXPECT_EQ(tb.n_rows(), num_rows);
  EXPECT_EQ(tb.n_cols(), num_cols);
}

TEST(AnalyticTableNumeric_Test, PreservesDataValues) {
  std::unordered_map<std::string, std::vector<int>> data{
      {"values", {10, 20, 30, 40}},
  };

  TbNumeric<int> tb(std::move(data));

  // Access the underlying Arrow table to verify values
  auto arrow_tb = tb.arrow_table();
  auto col = arrow_tb->column(0)->chunk(0);
  auto int_array = std::static_pointer_cast<arrow::Int32Array>(col);

  ASSERT_EQ(int_array->length(), 4);
  EXPECT_EQ(int_array->Value(0), 10);
  EXPECT_EQ(int_array->Value(1), 20);
  EXPECT_EQ(int_array->Value(2), 30);
  EXPECT_EQ(int_array->Value(3), 40);
}

TEST(AnalyticTableNumeric_Test, WorksWithTypeAliases) {
  {
    std::unordered_map<std::string, std::vector<int>> data{{"a", {1, 2}}};
    ttb::TbInt tb(std::move(data));
    EXPECT_EQ(tb.n_rows(), 2);
  }

  {
    std::unordered_map<std::string, std::vector<int64_t>> data{{"a", {1, 2}}};
    ttb::TbLong tb(std::move(data));
    EXPECT_EQ(tb.n_rows(), 2);
  }

  {
    std::unordered_map<std::string, std::vector<float>> data{{"a", {1.0f, 2.0f}}};
    ttb::TbFloat tb(std::move(data));
    EXPECT_EQ(tb.n_rows(), 2);
  }

  {
    std::unordered_map<std::string, std::vector<double>> data{{"a", {1.0, 2.0}}};
    ttb::TbDouble tb(std::move(data));
    EXPECT_EQ(tb.n_rows(), 2);
  }
}