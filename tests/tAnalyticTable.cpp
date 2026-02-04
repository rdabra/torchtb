#include <gtest/gtest.h>

#include "AnalyticTable.h"
#include "detail/ttbutils.h"

#include <arrow/api.h>
#include <memory>
#include <string>
#include <vector>

static ttb::AnalyticTable make_simple_table(int64_t nrows = 3) {
  arrow::Int64Builder ib;
  arrow::FloatBuilder fb;

  std::vector<int64_t> ivals;
  std::vector<float> fvals;
  for (int64_t i = 0; i < nrows; ++i) {
    ivals.push_back(i * 10);
    fvals.push_back(static_cast<float>(i) * 1.5f);
  }

  EXPECT_TRUE(ib.AppendValues(ivals).ok());
  EXPECT_TRUE(fb.AppendValues(fvals).ok());

  ttb::utl::shp<arrow::Array> icol, fcol;
  EXPECT_TRUE(ib.Finish(&icol).ok());
  EXPECT_TRUE(fb.Finish(&fcol).ok());

  auto schema = arrow::schema(
      {arrow::field("col_int", arrow::int64()), arrow::field("col_float", arrow::float32())});

  auto table = arrow::Table::Make(schema, {icol, fcol});
  return ttb::AnalyticTable{std::move(table)};
}

static ttb::AnalyticTable make_cat_table() {
  arrow::Int64Builder catb;
  arrow::FloatBuilder valb;
  EXPECT_TRUE(catb.AppendValues({1, 2, 1, 3}).ok());
  EXPECT_TRUE(valb.AppendValues({10.5f, 11.5f, 12.5f, 13.5f}).ok());

  std::shared_ptr<arrow::Array> cat_col, val_col;
  EXPECT_TRUE(catb.Finish(&cat_col).ok());
  EXPECT_TRUE(valb.Finish(&val_col).ok());
  auto schema = arrow::schema(
      {arrow::field("category", arrow::int64()), arrow::field("value", arrow::float32())});
  auto tbl = arrow::Table::Make(schema, {cat_col, val_col});
  return ttb::AnalyticTable{std::move(tbl)};
}

TEST(AnalyticTable_Test, MovesArrowTable) {
  auto table = make_simple_table();
  EXPECT_NE(table.arrow_table(), nullptr);
  EXPECT_EQ(table.n_rows(), 3);
  EXPECT_EQ(table.n_cols(), 2);
}

TEST(AnalyticTable_Test, ReturnsCorrectDimensions) {
  auto table = make_simple_table(5);
  EXPECT_EQ(table.n_rows(), 5);
  EXPECT_EQ(table.n_cols(), 2);
}

TEST(AnalyticTable_Test, ReturnsColumnNames) {
  auto table = make_simple_table();
  auto names = table.col_names();
  ASSERT_EQ(names.size(), 2u);
  EXPECT_EQ(names[0], "col_int");
  EXPECT_EQ(names[1], "col_float");
}

TEST(AnalyticTable_Test, ReturnsColumnNameByIndex) {
  auto table = make_simple_table();
  EXPECT_EQ(table.col_name(0), "col_int");
  EXPECT_EQ(table.col_name(1), "col_float");
}

TEST(AnalyticTable_Test, ReturnsColumnTypes) {
  auto table = make_simple_table();
  auto types = table.col_dtypes();
  ASSERT_EQ(types.size(), 2u);
  EXPECT_EQ(types[0], "int64");
  EXPECT_EQ(types[1], "float");
}

TEST(AnalyticTable_Test, FindsExistingColumn) {
  auto table = make_simple_table();
  auto idx = table.col_index("col_float");
  ASSERT_TRUE(idx.has_value());
  EXPECT_EQ(*idx, 1);
}

TEST(AnalyticTable_Test, ReturnsNulloptForMissing) {
  auto table = make_simple_table();
  auto idx = table.col_index("nonexistent");
  EXPECT_FALSE(idx.has_value());
}

TEST(AnalyticTable_Test, RemovesColumnByIndex) {
  auto table = make_simple_table();
  table.remove_col(0);
  EXPECT_EQ(table.n_cols(), 1);
  auto names = table.col_names();
  EXPECT_EQ(names[0], "col_float");
}

TEST(AnalyticTable_Test, FailsOnInvalidIndex) {
  auto table = make_simple_table();
  EXPECT_THROW(table.remove_col(99), ttb::AnalyticTableError);
  EXPECT_EQ(table.n_cols(), 2); // unchanged
}

TEST(AnalyticTable_Test, KeepsSpecifiedColumns) {
  auto table = make_simple_table();
  table.keep_cols({1});
  EXPECT_EQ(table.n_cols(), 1);
  auto names = table.col_names();
  EXPECT_EQ(names[0], "col_float");
}

TEST(AnalyticTable_Test, KeepNumericColsRemovesAllNonNumericColumns) {
  arrow::Int64Builder id_builder;
  arrow::StringBuilder label_builder;
  arrow::FloatBuilder score_builder;
  arrow::StringBuilder note_builder;

  ASSERT_TRUE(id_builder.AppendValues({1, 2, 3}).ok());
  ASSERT_TRUE(label_builder.AppendValues({"a", "b", "c"}).ok());
  ASSERT_TRUE(score_builder.AppendValues({10.0f, 20.0f, 30.0f}).ok());
  ASSERT_TRUE(note_builder.AppendValues({"x", "y", "z"}).ok());

  std::shared_ptr<arrow::Array> id_array, label_array, score_array, note_array;
  ASSERT_TRUE(id_builder.Finish(&id_array).ok());
  ASSERT_TRUE(label_builder.Finish(&label_array).ok());
  ASSERT_TRUE(score_builder.Finish(&score_array).ok());
  ASSERT_TRUE(note_builder.Finish(&note_array).ok());

  auto schema = arrow::schema({arrow::field("id", arrow::int64()),
                               arrow::field("label", arrow::utf8()),
                               arrow::field("score", arrow::float32()),
                               arrow::field("note", arrow::utf8())});
  auto arrow_table = arrow::Table::Make(schema, {id_array, label_array, score_array, note_array});
  ttb::AnalyticTable table{ttb::utl::shp<arrow::Table>(arrow_table)};

  table.keep_numeric_cols();

  EXPECT_EQ(table.n_rows(), 3);
  EXPECT_EQ(table.n_cols(), 2);

  auto names = table.col_names();
  ASSERT_EQ(names.size(), 2);
  EXPECT_EQ(names[0], "id");
  EXPECT_EQ(names[1], "score");

  auto dtypes = table.col_dtypes();
  ASSERT_EQ(dtypes.size(), 2);
  EXPECT_EQ(dtypes[0], "int64");
  EXPECT_EQ(dtypes[1], "float");
}

TEST(AnalyticTable_Test, FailsOnInvalidIndices) {
  auto table = make_simple_table();
  EXPECT_THROW(table.keep_cols({0, 99}), ttb::AnalyticTableError);
}

TEST(AnalyticTable_Test, RenamesAllColumns) {
  auto table = make_simple_table();
  table.rename_cols({"new_a", "new_b"});
  auto names = table.col_names();
  ASSERT_EQ(names.size(), 2u);
  EXPECT_EQ(names[0], "new_a");
  EXPECT_EQ(names[1], "new_b");
}

TEST(AnalyticTable_Test, FailsOnSizeMismatch) {
  auto table = make_simple_table();
  EXPECT_THROW(table.rename_cols({"only_one"}), ttb::AnalyticTableError);
}

TEST(AnalyticTable_Test, SlicesRows) {
  auto table = make_simple_table(10);
  table.slice(2, 5);
  EXPECT_EQ(table.n_rows(), 5);
  EXPECT_EQ(table.n_cols(), 2);
}

TEST(AnalyticTable_Test, FailsOnInvalidRange) {
  auto table = make_simple_table(5);
  EXPECT_THROW(table.slice(0, 100), ttb::AnalyticTableError);
}

TEST(AnalyticTable_Test, ReturnsNewSlicedTable) {
  auto table = make_simple_table(10);
  auto res = table.sliced(1, 3);
  EXPECT_EQ(res.n_rows(), 3);
  EXPECT_EQ(res.n_cols(), 2);
  // Original unchanged
  EXPECT_EQ(table.n_rows(), 10);
}

TEST(AnalyticTable_Test, ReordersColumns) {
  auto table = make_simple_table();
  table.reorder_cols({1, 0});
  auto names = table.col_names();
  ASSERT_EQ(names.size(), 2u);
  EXPECT_EQ(names[0], "col_float");
  EXPECT_EQ(names[1], "col_int");
}

TEST(AnalyticTable_Test, FailsOnInvalidIndicesReorder) {
  auto table = make_simple_table();
  EXPECT_THROW(table.reorder_cols({1}), ttb::AnalyticTableError);
}

TEST(AnalyticTable_Test, CopiesSpecifiedColumns) {
  auto table = make_simple_table();
  auto res = table.copy_cols({0});
  EXPECT_EQ(res.n_cols(), 1);
  EXPECT_EQ(res.n_rows(), 3);
  auto names = res.col_names();
  EXPECT_EQ(names[0], "col_int");
}

TEST(AnalyticTable_Test, CreatesIndependentCopy) {
  auto table = make_simple_table();
  auto res = table.clone();
  EXPECT_EQ(res.n_rows(), table.n_rows());
  EXPECT_EQ(res.n_cols(), table.n_cols());

  // Modify clone
  res.remove_col(0);
  EXPECT_EQ(res.n_cols(), 1);
  // Original unchanged
  EXPECT_EQ(table.n_cols(), 2);
}

TEST(AnalyticTable_Test, ExtractsColumnFromIndex) {
  auto table = make_simple_table();
  auto res = table.right_extract_of(0);
  EXPECT_EQ(res.n_cols(), 1);
  EXPECT_THROW(table.right_extract_of(1), ttb::AnalyticTableError);

  auto names = res.col_names();
  EXPECT_EQ(names[0], "col_float");
  // Original modified
  EXPECT_EQ(table.n_cols(), 1);
}

TEST(AnalyticTable_Test, AppendsRowsFromAnotherTable) {
  auto table1 = make_simple_table(2);
  auto table2 = make_simple_table(3);

  table1.append(table2, ttb::Axis::ROW);
  EXPECT_EQ(table1.n_rows(), 5);
  EXPECT_EQ(table1.n_cols(), 2);
}

TEST(AnalyticTable_Test, AppendsColumnsFromAnotherTable) {
  auto table1 = make_simple_table(3);
  auto table2 = make_simple_table(3);

  table1.append(table2, ttb::Axis::COLUMN);
  EXPECT_EQ(table1.n_rows(), 3);
  EXPECT_EQ(table1.n_cols(), 4);
}

TEST(AnalyticTable_Test, ClearsTableReference) {
  auto table = make_simple_table();
  EXPECT_NE(table.arrow_table(), nullptr);
  table.reset();
  EXPECT_EQ(table.arrow_table(), nullptr);
}

TEST(AnalyticTable_Test, DoesNotCrash) {
  auto table = make_simple_table(100);
  EXPECT_NO_THROW(table.print_head(10));
}

TEST(AnalyticTable_Test, DoesNotCrashPrintTail) {
  auto table = make_simple_table(100);
  EXPECT_NO_THROW(table.print_tail(10));
}

TEST(AnalyticTable_Test, MovesColumnToNewPosition) {
  auto table = make_simple_table(); // has col_int at 0, col_float at 1

  // Move col_float (index 1) to position 0
  table.move_column(1, 0);

  auto names = table.col_names();
  ASSERT_EQ(names.size(), 2u);
  EXPECT_EQ(names[0], "col_float");
  EXPECT_EQ(names[1], "col_int");
}

TEST(AnalyticTable_Test, MovesColumnToEnd) {
  // Create table with 3 columns
  arrow::Int64Builder ib;
  arrow::FloatBuilder fb;
  arrow::DoubleBuilder db;

  EXPECT_TRUE(ib.AppendValues({1, 2, 3}).ok());
  EXPECT_TRUE(fb.AppendValues({1.5f, 2.5f, 3.5f}).ok());
  EXPECT_TRUE(db.AppendValues({10.0, 20.0, 30.0}).ok());

  std::shared_ptr<arrow::Array> icol, fcol, dcol;
  EXPECT_TRUE(ib.Finish(&icol).ok());
  EXPECT_TRUE(fb.Finish(&fcol).ok());
  EXPECT_TRUE(db.Finish(&dcol).ok());

  auto schema =
      arrow::schema({arrow::field("a", arrow::int64()), arrow::field("b", arrow::float32()),
                     arrow::field("c", arrow::float64())});
  auto tbl = arrow::Table::Make(schema, {icol, fcol, dcol});
  ttb::AnalyticTable table{std::move(tbl)};

  // Move first column (a) to last position
  table.move_column(0, 2);

  auto names = table.col_names();
  ASSERT_EQ(names.size(), 3u);
  EXPECT_EQ(names[0], "b");
  EXPECT_EQ(names[1], "c");
  EXPECT_EQ(names[2], "a");
}

TEST(AnalyticTable_Test, SamePositionSucceeds) {
  auto table = make_simple_table();

  table.move_column(0, 0);

  auto names = table.col_names();
  EXPECT_EQ(names[0], "col_int");
  EXPECT_EQ(names[1], "col_float");
}

TEST(AnalyticTable_Test, FailsOnInvalidFromIndex) {
  auto table = make_simple_table();

  EXPECT_THROW(table.move_column(99, 0), ttb::AnalyticTableError);
}

TEST(AnalyticTable_Test, FailsOnInvalidToIndex) {
  auto table = make_simple_table();

  EXPECT_THROW(table.move_column(0, 99), ttb::AnalyticTableError);
}

TEST(AnalyticTable_Test, FailsOnNegativeIndices) {
  auto table = make_simple_table();

  EXPECT_THROW(table.move_column(-1, 0), ttb::AnalyticTableError);
  EXPECT_THROW(table.move_column(0, -1), ttb::AnalyticTableError);
}

TEST(AnalyticTable_Test, ExtractsAndRemovesColumn) {
  auto table = make_simple_table();
  EXPECT_EQ(table.n_cols(), 2);

  auto res = table.extract_column(1); // Extract col_float

  // Original table should have one less column
  EXPECT_EQ(table.n_cols(), 1);
  auto names = table.col_names();
  EXPECT_EQ(names[0], "col_int");

  // Extracted table should have only the extracted column
  EXPECT_EQ(res.n_cols(), 1);
  auto extracted_names = res.col_names();
  EXPECT_EQ(extracted_names[0], "col_float");
  EXPECT_EQ(res.n_rows(), table.n_rows());
}

TEST(AnalyticTable_Test, ExtractsFirstColumn) {
  auto table = make_simple_table();

  auto res = table.extract_column(0);

  EXPECT_EQ(table.n_cols(), 1);
  auto names = table.col_names();
  EXPECT_EQ(names[0], "col_float");

  EXPECT_EQ(res.n_cols(), 1);
  auto extracted_names = res.col_names();
  EXPECT_EQ(extracted_names[0], "col_int");
}

TEST(AnalyticTable_Test, ExtractsLastColumn) {
  // Create table with 3 columns
  arrow::Int64Builder ib;
  arrow::FloatBuilder fb;
  arrow::DoubleBuilder db;

  EXPECT_TRUE(ib.AppendValues({1, 2}).ok());
  EXPECT_TRUE(fb.AppendValues({1.5f, 2.5f}).ok());
  EXPECT_TRUE(db.AppendValues({10.0, 20.0}).ok());

  std::shared_ptr<arrow::Array> icol, fcol, dcol;
  EXPECT_TRUE(ib.Finish(&icol).ok());
  EXPECT_TRUE(fb.Finish(&fcol).ok());
  EXPECT_TRUE(db.Finish(&dcol).ok());

  auto schema =
      arrow::schema({arrow::field("a", arrow::int64()), arrow::field("b", arrow::float32()),
                     arrow::field("c", arrow::float64())});
  auto tbl = arrow::Table::Make(schema, {icol, fcol, dcol});
  ttb::AnalyticTable table{std::move(tbl)};

  auto res = table.extract_column(2);

  EXPECT_EQ(table.n_cols(), 2);
  auto names = table.col_names();
  EXPECT_EQ(names[0], "a");
  EXPECT_EQ(names[1], "b");

  EXPECT_EQ(res.n_cols(), 1);
  auto extracted_names = res.col_names();
  EXPECT_EQ(extracted_names[0], "c");
}

TEST(AnalyticTable_Test, ExtractCol_FailsOnInvalidIndex) {
  auto table = make_simple_table();

  EXPECT_THROW(table.extract_column(99), ttb::AnalyticTableError);

  // Table should remain unchanged
  EXPECT_EQ(table.n_cols(), 2);
}

TEST(AnalyticTable_Test, FailsOnNegativeIndex) {
  auto table = make_simple_table();

  EXPECT_THROW(table.extract_column(-1), ttb::AnalyticTableError);

  EXPECT_EQ(table.n_cols(), 2);
}

TEST(AnalyticTable_Test, FailsOnEmptyTable) {
  arrow::Int64Builder ib;
  std::shared_ptr<arrow::Array> icol;
  EXPECT_TRUE(ib.Finish(&icol).ok());

  auto schema = arrow::schema({arrow::field("empty", arrow::int64())});
  auto tbl = arrow::Table::Make(schema, {icol});
  ttb::AnalyticTable table{std::move(tbl)};

  // Extract the only column
  EXPECT_THROW(table.extract_column(0), ttb::AnalyticTableError);
}

TEST(AnalyticTable_Test, PreservesRowCount) {
  auto table = make_simple_table(100);
  int64_t original_rows = table.n_rows();

  auto res = table.extract_column(0);

  EXPECT_EQ(table.n_rows(), original_rows);
  EXPECT_EQ(res.n_rows(), original_rows);
}

TEST(AnalyticTable_Test, ExpandsCategoricalColumn) {
  auto t = make_cat_table();
  int64_t rows_before = t.n_rows();
  int cols_before = t.n_cols();

  t.one_hot_expand(0);

  EXPECT_EQ(t.n_rows(), rows_before);
  EXPECT_GT(t.n_cols(), cols_before); // columns increased

  // Collect unique values from original data
  std::vector<int64_t> original_vals = {1, 2, 1, 3};
  std::map<int64_t, int> counts;
  for (auto v : original_vals)
    counts[v]++;

  // Assume one-hot columns are now the rightmost |unique| columns
  int unique_ct = static_cast<int>(counts.size());
  int cols_after = t.n_cols();
  ASSERT_GE(cols_after, cols_before + unique_ct - 1);

  // Try interpretation A: original removed (rightmost unique_ct columns)
  int start_idx = cols_after - unique_ct;

  // Validate one-hot property
  for (int c = start_idx; c < cols_after; ++c) {
    auto arr = t.arrow_table()->column(c)->chunk(0);
    ASSERT_EQ(arr->length(), rows_before);
    // Expect integer type
    ASSERT_TRUE(arr->type_id() == arrow::Type::INT32 || arr->type_id() == arrow::Type::INT64);
  }

  // Row-wise: each row should sum to 1 across these one-hot columns
  for (int64_t r = 0; r < rows_before; ++r) {
    int row_sum = 0;
    for (int c = start_idx; c < cols_after; ++c) {
      auto arr = t.arrow_table()->column(c)->chunk(0);
      int64_t v = (arr->type_id() == arrow::Type::INT32)
                      ? static_cast<int64_t>(static_cast<const arrow::Int32Array &>(*arr).Value(r))
                      : static_cast<const arrow::Int64Array &>(*arr).Value(r);
      ASSERT_TRUE(v == 0 || v == 1);
      row_sum += static_cast<int>(v);
    }
    EXPECT_EQ(row_sum, 1) << "Row " << r << " does not have exactly one '1'";
  }

  // Column sums should match counts of each category (order dependent on implementation).
  // We can't assert order without name info; just assert sums match multiset.
  std::multiset<int> observed_counts;
  for (int c = start_idx; c < cols_after; ++c) {
    auto arr = t.arrow_table()->column(c)->chunk(0);
    int sum = 0;
    for (int64_t r = 0; r < rows_before; ++r) {
      int64_t v = (arr->type_id() == arrow::Type::INT32)
                      ? static_cast<int64_t>(static_cast<const arrow::Int32Array &>(*arr).Value(r))
                      : static_cast<const arrow::Int64Array &>(*arr).Value(r);
      sum += static_cast<int>(v);
    }
    observed_counts.insert(sum);
  }
  std::multiset<int> expected_counts;
  for (auto &kv : counts)
    expected_counts.insert(kv.second);
  EXPECT_EQ(observed_counts, expected_counts);
}

TEST(AnalyticTable_Test, InvalidNegativeIndexFails) {
  auto t = make_cat_table();
  EXPECT_THROW(t.one_hot_expand(-1), ttb::AnalyticTableError);
}

TEST(AnalyticTable_Test, OutOfRangeIndexFails) {
  auto t = make_cat_table();
  EXPECT_THROW(t.one_hot_expand(99), ttb::AnalyticTableError);
}

TEST(AnalyticTable_Test, SortsTableAscendingByIntColumn) {
  auto table = make_simple_table(5);
  table.sort(0, ttb::SortOrder::ASC);
  EXPECT_EQ(table.n_rows(), 5);
  EXPECT_EQ(table.n_cols(), 2);
}

TEST(AnalyticTable_Test, SortsTableDescendingByIntColumn) {
  auto table = make_simple_table(5);
  table.sort(0, ttb::SortOrder::DESC);
  EXPECT_EQ(table.n_rows(), 5);
  EXPECT_EQ(table.n_cols(), 2);
}

TEST(AnalyticTable_Test, SortsTableByFloatColumn) {
  auto table = make_simple_table(5);
  table.sort(1, ttb::SortOrder::ASC);
  EXPECT_EQ(table.n_rows(), 5);
}

TEST(AnalyticTable_Test, SortDefaultsToAscending) {
  auto table = make_simple_table(3);
  table.sort(0); // No mode specified, defaults to ASC
}

TEST(AnalyticTable_Test, SortFailsOnInvalidNegativeIndex) {
  auto table = make_simple_table(3);
  EXPECT_THROW(table.sort(-1), ttb::AnalyticTableError);
}

TEST(AnalyticTable_Test, SortFailsOnInvalidOutOfBoundsIndex) {
  auto table = make_simple_table(3);
  EXPECT_THROW(table.sort(99), ttb::AnalyticTableError);
}

TEST(AnalyticTable_Test, SortPreservesCategoryValues) {
  auto table = make_cat_table();
  // Original values: {1, 2, 1, 3}

  table.sort(0, ttb::SortOrder::ASC);

  auto tbl = table.arrow_table();
  auto cat_col = tbl->column(0)->chunk(0);
  ASSERT_EQ(cat_col->length(), 4);

  // After sorting by category column (ascending), should be: 1, 1, 2, 3
  auto cat_array = std::static_pointer_cast<arrow::Int64Array>(cat_col);
  EXPECT_EQ(cat_array->Value(0), 1);
  EXPECT_EQ(cat_array->Value(1), 1);
  EXPECT_EQ(cat_array->Value(2), 2);
  EXPECT_EQ(cat_array->Value(3), 3);
}

TEST(AnalyticTable_Test, SortDescendingOrder) {
  auto table = make_cat_table();
  // Original values: {1, 2, 1, 3}

  table.sort(0, ttb::SortOrder::DESC);

  auto tbl = table.arrow_table();
  auto cat_col = tbl->column(0)->chunk(0);
  ASSERT_EQ(cat_col->length(), 4);

  // After sorting by category column (descending), should be: 3, 2, 1, 1
  auto cat_array = std::static_pointer_cast<arrow::Int64Array>(cat_col);
  EXPECT_EQ(cat_array->Value(0), 3);
  EXPECT_EQ(cat_array->Value(1), 2);
  EXPECT_EQ(cat_array->Value(2), 1);
  EXPECT_EQ(cat_array->Value(3), 1);
}

TEST(AnalyticTable_Test, SortMaintainsRowIntegrity) {
  // Create table with paired values to verify row integrity
  arrow::Int64Builder catb;
  arrow::FloatBuilder valb;
  EXPECT_TRUE(catb.AppendValues({3, 1, 2, 1}).ok());
  EXPECT_TRUE(valb.AppendValues({30.0f, 10.0f, 20.0f, 10.5f}).ok());

  std::shared_ptr<arrow::Array> cat_col, val_col;
  EXPECT_TRUE(catb.Finish(&cat_col).ok());
  EXPECT_TRUE(valb.Finish(&val_col).ok());
  auto schema = arrow::schema(
      {arrow::field("category", arrow::int64()), arrow::field("value", arrow::float32())});
  auto tbl = arrow::Table::Make(schema, {cat_col, val_col});
  ttb::AnalyticTable table{std::move(tbl)};

  table.sort(0, ttb::SortOrder::ASC);

  auto sorted_tbl = table.arrow_table();
  auto sorted_cat = std::static_pointer_cast<arrow::Int64Array>(sorted_tbl->column(0)->chunk(0));
  auto sorted_val = std::static_pointer_cast<arrow::FloatArray>(sorted_tbl->column(1)->chunk(0));

  // Verify sorted order and row pairing integrity
  EXPECT_EQ(sorted_cat->Value(0), 1);
  EXPECT_EQ(sorted_val->Value(0), 10.0f);

  EXPECT_EQ(sorted_cat->Value(1), 1);
  EXPECT_EQ(sorted_val->Value(1), 10.5f);

  EXPECT_EQ(sorted_cat->Value(2), 2);
  EXPECT_EQ(sorted_val->Value(2), 20.0f);

  EXPECT_EQ(sorted_cat->Value(3), 3);
  EXPECT_EQ(sorted_val->Value(3), 30.0f);
}

TEST(AnalyticTable_Test, FilterRemovesRowsNotMatchingCriterion) {
  arrow::Int64Builder id_builder;
  ASSERT_TRUE(id_builder.AppendValues({1, 2, 3, 4, 5}).ok());
  std::shared_ptr<arrow::Array> id_array;
  ASSERT_TRUE(id_builder.Finish(&id_array).ok());

  arrow::StringBuilder cat_builder;
  ASSERT_TRUE(cat_builder.AppendValues({"a", "b", "b", "a", "c"}).ok());
  std::shared_ptr<arrow::Array> cat_array;
  ASSERT_TRUE(cat_builder.Finish(&cat_array).ok());

  auto schema =
      arrow::schema({arrow::field("id", arrow::int64()), arrow::field("cat", arrow::utf8())});
  auto table = arrow::Table::Make(schema, {id_array, cat_array});

  ttb::AnalyticTable analytic_table{ttb::utl::shp<arrow::Table>(table)};
  auto or_table = analytic_table.clone();
  ASSERT_EQ(analytic_table.n_rows(), 5);

  std::vector<ttb::Criterion> criteria{{"id", ttb::Criterion::Condition::GREATER_EQUAL, "2"},
                                       {"cat", ttb::Criterion::Condition::EQUAL, "b"}};
  analytic_table.keep_rows(criteria);

  EXPECT_EQ(analytic_table.n_rows(), 2);

  auto filtered = analytic_table.arrow_table();
  auto id_chunked = filtered->GetColumnByName("id");
  ASSERT_NE(id_chunked, nullptr);
  auto id_array_filtered = std::static_pointer_cast<arrow::Int64Array>(id_chunked->chunk(0));
  ASSERT_EQ(id_array_filtered->length(), 2);
  EXPECT_EQ(id_array_filtered->Value(0), 2);
  EXPECT_EQ(id_array_filtered->Value(1), 3);

  auto cat_chunked = filtered->GetColumnByName("cat");
  ASSERT_NE(cat_chunked, nullptr);
  auto cat_array_filtered = std::static_pointer_cast<arrow::StringArray>(cat_chunked->chunk(0));
  ASSERT_EQ(cat_array_filtered->length(), 2);
  EXPECT_EQ(cat_array_filtered->GetString(0), "b");
  EXPECT_EQ(cat_array_filtered->GetString(1), "b");

  or_table.keep_rows(criteria, ttb::LogicOp::OR);

  EXPECT_EQ(or_table.n_rows(), 4);

  auto or_filtered = or_table.arrow_table();
  auto or_id_chunked = or_filtered->GetColumnByName("id");
  ASSERT_NE(or_id_chunked, nullptr);
  auto or_id_array = std::static_pointer_cast<arrow::Int64Array>(or_id_chunked->chunk(0));
  ASSERT_EQ(or_id_array->length(), 4);
  EXPECT_EQ(or_id_array->Value(0), 2);
  EXPECT_EQ(or_id_array->Value(1), 3);
  EXPECT_EQ(or_id_array->Value(2), 4);
  EXPECT_EQ(or_id_array->Value(3), 5);

  auto or_cat_chunked = or_filtered->GetColumnByName("cat");
  ASSERT_NE(or_cat_chunked, nullptr);
  auto or_cat_array = std::static_pointer_cast<arrow::StringArray>(or_cat_chunked->chunk(0));
  ASSERT_EQ(or_cat_array->length(), 4);
  EXPECT_EQ(or_cat_array->GetString(0), "b");
  EXPECT_EQ(or_cat_array->GetString(1), "b");
  EXPECT_EQ(or_cat_array->GetString(2), "a");
  EXPECT_EQ(or_cat_array->GetString(3), "c");
}

TEST(AnalyticTable_Test, DropNullsRemovesRowsWithNulls) {
  arrow::Int64Builder id_builder;
  arrow::StringBuilder note_builder;
  arrow::FloatBuilder score_builder;

  ASSERT_TRUE(id_builder.AppendValues({1, 2, 3, 4, 5}).ok());
  ASSERT_TRUE(note_builder.Append("a").ok());
  ASSERT_TRUE(note_builder.AppendNull().ok());
  ASSERT_TRUE(note_builder.Append("c").ok());
  ASSERT_TRUE(note_builder.Append("d").ok());
  ASSERT_TRUE(note_builder.AppendNull().ok());

  ASSERT_TRUE(score_builder.Append(10.0f).ok());
  ASSERT_TRUE(score_builder.Append(11.0f).ok());
  ASSERT_TRUE(score_builder.AppendNull().ok());
  ASSERT_TRUE(score_builder.Append(13.0f).ok());
  ASSERT_TRUE(score_builder.AppendNull().ok());

  std::shared_ptr<arrow::Array> id_array, note_array, score_array;
  ASSERT_TRUE(id_builder.Finish(&id_array).ok());
  ASSERT_TRUE(note_builder.Finish(&note_array).ok());
  ASSERT_TRUE(score_builder.Finish(&score_array).ok());

  auto schema =
      arrow::schema({arrow::field("id", arrow::int64()), arrow::field("note", arrow::utf8()),
                     arrow::field("score", arrow::float32())});
  auto table = arrow::Table::Make(schema, {id_array, note_array, score_array});
  ttb::AnalyticTable analytic_table{ttb::utl::shp<arrow::Table>(table)};
  auto or_table = analytic_table.clone();

  analytic_table.drop_nulls({"note", "score"}, ttb::LogicOp::OR);

  auto filtered = analytic_table.arrow_table();
  ASSERT_EQ(filtered->num_rows(), 2);

  auto id_chunked = filtered->GetColumnByName("id");
  ASSERT_NE(id_chunked, nullptr);
  auto id_filtered = std::static_pointer_cast<arrow::Int64Array>(id_chunked->chunk(0));
  ASSERT_EQ(id_filtered->length(), 2);
  EXPECT_EQ(id_filtered->Value(0), 1);
  EXPECT_EQ(id_filtered->Value(1), 4);

  auto note_chunked = filtered->GetColumnByName("note");
  ASSERT_NE(note_chunked, nullptr);
  auto note_filtered = std::static_pointer_cast<arrow::StringArray>(note_chunked->chunk(0));
  ASSERT_EQ(note_filtered->length(), 2);
  EXPECT_EQ(note_filtered->GetString(0), "a");
  EXPECT_EQ(note_filtered->GetString(1), "d");

  auto score_chunked = filtered->GetColumnByName("score");
  ASSERT_NE(score_chunked, nullptr);
  auto score_filtered = std::static_pointer_cast<arrow::FloatArray>(score_chunked->chunk(0));
  ASSERT_EQ(score_filtered->length(), 2);
  EXPECT_FLOAT_EQ(score_filtered->Value(0), 10.0f);
  EXPECT_FLOAT_EQ(score_filtered->Value(1), 13.0f);

  or_table.drop_nulls({"note", "score"}, ttb::LogicOp::AND);

  auto or_filtered = or_table.arrow_table();
  ASSERT_EQ(or_filtered->num_rows(), 4);

  auto or_id_chunked = or_filtered->GetColumnByName("id");
  ASSERT_NE(or_id_chunked, nullptr);
  auto or_id_filtered = std::static_pointer_cast<arrow::Int64Array>(or_id_chunked->chunk(0));
  ASSERT_EQ(or_id_filtered->length(), 4);
  EXPECT_EQ(or_id_filtered->Value(0), 1);
  EXPECT_EQ(or_id_filtered->Value(1), 2);
  EXPECT_EQ(or_id_filtered->Value(2), 3);
  EXPECT_EQ(or_id_filtered->Value(3), 4);

  auto or_note_chunked = or_filtered->GetColumnByName("note");
  ASSERT_NE(or_note_chunked, nullptr);
  auto or_note_filtered = std::static_pointer_cast<arrow::StringArray>(or_note_chunked->chunk(0));
  ASSERT_EQ(or_note_filtered->length(), 4);
  EXPECT_TRUE(or_note_filtered->IsValid(0));
  EXPECT_TRUE(or_note_filtered->IsNull(1));
  EXPECT_TRUE(or_note_filtered->IsValid(2));
  EXPECT_TRUE(or_note_filtered->IsValid(3));
  EXPECT_EQ(or_note_filtered->GetString(0), "a");
  EXPECT_EQ(or_note_filtered->GetString(2), "c");
  EXPECT_EQ(or_note_filtered->GetString(3), "d");

  auto or_score_chunked = or_filtered->GetColumnByName("score");
  ASSERT_NE(or_score_chunked, nullptr);
  auto or_score_filtered = std::static_pointer_cast<arrow::FloatArray>(or_score_chunked->chunk(0));
  ASSERT_EQ(or_score_filtered->length(), 4);
  EXPECT_TRUE(or_score_filtered->IsValid(0));
  EXPECT_TRUE(or_score_filtered->IsValid(1));
  EXPECT_TRUE(or_score_filtered->IsNull(2));
  EXPECT_TRUE(or_score_filtered->IsValid(3));
  EXPECT_FLOAT_EQ(or_score_filtered->Value(0), 10.0f);
  EXPECT_FLOAT_EQ(or_score_filtered->Value(1), 11.0f);
  EXPECT_FLOAT_EQ(or_score_filtered->Value(3), 13.0f);
}

TEST(AnalyticTable_Test, DropDuplicatesRemovesDuplicateRows) {
  arrow::Int64Builder id_builder;
  arrow::StringBuilder cat_builder;

  ASSERT_TRUE(id_builder.AppendValues({1, 2, 2, 3, 1}).ok());
  ASSERT_TRUE(cat_builder.AppendValues({"a", "b", "b", "c", "a"}).ok());

  std::shared_ptr<arrow::Array> id_array, cat_array;
  ASSERT_TRUE(id_builder.Finish(&id_array).ok());
  ASSERT_TRUE(cat_builder.Finish(&cat_array).ok());

  auto schema =
      arrow::schema({arrow::field("id", arrow::int64()), arrow::field("cat", arrow::utf8())});
  auto table = arrow::Table::Make(schema, {id_array, cat_array});
  ttb::AnalyticTable analytic_table{ttb::utl::shp<arrow::Table>(table)};

  analytic_table.drop_duplicates();

  auto deduped = analytic_table.arrow_table();
  ASSERT_EQ(deduped->num_rows(), 3);
  ASSERT_EQ(deduped->num_columns(), 2);

  auto id_chunked = deduped->GetColumnByName("id");
  ASSERT_NE(id_chunked, nullptr);
  auto id_filtered = std::static_pointer_cast<arrow::Int64Array>(id_chunked->chunk(0));
  ASSERT_EQ(id_filtered->length(), 3);

  auto cat_chunked = deduped->GetColumnByName("cat");
  ASSERT_NE(cat_chunked, nullptr);
  auto cat_filtered = std::static_pointer_cast<arrow::StringArray>(cat_chunked->chunk(0));
  ASSERT_EQ(cat_filtered->length(), 3);

  bool saw_a = false;
  bool saw_b = false;
  bool saw_c = false;
  for (int64_t i = 0; i < id_filtered->length(); ++i) {
    auto id = id_filtered->Value(i);
    auto cat = cat_filtered->GetString(i);
    if (id == 1 && cat == "a")
      saw_a = true;
    else if (id == 2 && cat == "b")
      saw_b = true;
    else if (id == 3 && cat == "c")
      saw_c = true;
    else
      ADD_FAILURE() << "Unexpected row: id=" << id << ", cat=" << cat;
  }

  EXPECT_TRUE(saw_a);
  EXPECT_TRUE(saw_b);
  EXPECT_TRUE(saw_c);
}
