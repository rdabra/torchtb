#include <gtest/gtest.h>

#include "AnalyticTableNumeric.h"
#include "TrainingBundle.h"
#include "XYMatrix.h"

#include <arrow/api.h>
#include <torch/torch.h>

using ttb::XYMatrix;

namespace {

// Helpers
torch::Tensor make_X(int64_t rows, int64_t cols) {
  // Simple deterministic X
  torch::Tensor X = torch::empty({rows, cols}, torch::dtype(torch::kFloat32));
  auto a = X.accessor<float, 2>();
  for (int64_t i = 0; i < rows; ++i)
    for (int64_t j = 0; j < cols; ++j)
      a[i][j] = static_cast<float>(i * 100 + j);
  return X;
}

torch::Tensor make_one_hot_Y(const std::vector<int> &labels, int num_classes) {
  int64_t rows = static_cast<int64_t>(labels.size());
  torch::Tensor Y = torch::zeros({rows, num_classes}, torch::dtype(torch::kFloat32));
  for (int64_t i = 0; i < rows; ++i) {
    Y.index_put_({i, labels[i]}, 1.0f);
  }
  return Y;
}

std::vector<int> argmax_labels(const torch::Tensor &Y) {
  auto idx = Y.argmax(/*dim=*/1, /*keepdim=*/false);
  std::vector<int> out(idx.size(0));
  auto a = idx.accessor<int64_t, 1>();
  for (int64_t i = 0; i < idx.size(0); ++i)
    out[i] = static_cast<int>(a[i]);
  return out;
}

int count_label(const std::vector<int> &v, int label) {
  return static_cast<int>(std::count(v.begin(), v.end(), label));
}

template <typename T, typename Builder, typename ArrT>
std::shared_ptr<arrow::Array> build_arrow_col(const std::vector<T> &vals, Builder &b) {
  for (auto v : vals) {
    auto status = b.Append(v);
    if (!status.ok())
      throw std::runtime_error("Append failed: " + status.ToString());
  }
  std::shared_ptr<arrow::Array> arr;
  auto status = b.Finish(&arr);
  if (!status.ok())
    throw std::runtime_error("Finish failed: " + status.ToString());

  return arr;
}

template <typename T>
ttb::AnalyticTableNumeric<T> make_numeric_table_T(int64_t rows, int cols) {
  std::vector<std::shared_ptr<arrow::Array>> columns;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  columns.reserve(cols);
  fields.reserve(cols);

  for (int c = 0; c < cols; ++c) {
    if constexpr (std::is_same_v<T, float>) {
      arrow::FloatBuilder b;
      std::vector<float> vals(rows);
      for (int64_t r = 0; r < rows; ++r)
        vals[r] = static_cast<float>(r * 10 + c);
      columns.push_back(build_arrow_col<float, arrow::FloatBuilder, arrow::FloatArray>(vals, b));
      fields.push_back(arrow::field("c" + std::to_string(c), arrow::float32()));
    } else if constexpr (std::is_same_v<T, double>) {
      arrow::DoubleBuilder b;
      std::vector<double> vals(rows);
      for (int64_t r = 0; r < rows; ++r)
        vals[r] = static_cast<double>(r * 10 + c);
      columns.push_back(build_arrow_col<double, arrow::DoubleBuilder, arrow::DoubleArray>(vals, b));
      fields.push_back(arrow::field("c" + std::to_string(c), arrow::float64()));
    } else if constexpr (std::is_same_v<T, int64_t>) {
      arrow::Int64Builder b;
      std::vector<int64_t> vals(rows);
      for (int64_t r = 0; r < rows; ++r)
        vals[r] = static_cast<int64_t>(r * 10 + c);
      columns.push_back(build_arrow_col<int64_t, arrow::Int64Builder, arrow::Int64Array>(vals, b));
      fields.push_back(arrow::field("c" + std::to_string(c), arrow::int64()));
    } else { // int
      arrow::Int32Builder b;
      std::vector<int32_t> vals(rows);
      for (int64_t r = 0; r < rows; ++r)
        vals[r] = static_cast<int32_t>(r * 10 + c);
      columns.push_back(build_arrow_col<int32_t, arrow::Int32Builder, arrow::Int32Array>(vals, b));
      fields.push_back(arrow::field("c" + std::to_string(c), arrow::int32()));
    }
  }
  auto schema = arrow::schema(fields);
  auto table = arrow::Table::Make(schema, columns);
  return ttb::AnalyticTableNumeric<T>{std::move(table)};
}

// Build X with a unique row id in column 0 and another feature in column 1.
// Y is one-hot with label = id % num_classes.
// This lets us verify X/Y alignment after shuffling.
std::pair<torch::Tensor, torch::Tensor> make_xy_with_ids(int64_t rows, int num_classes) {
  auto ids = torch::arange(0, rows, torch::dtype(torch::kInt64)); // [rows]
  auto ids_f = ids.to(torch::kFloat32).unsqueeze(1);              // [rows,1]
  auto feat1 = ids_f * 10.0f + 1.0f;                              // [rows,1]
  auto X = torch::cat({ids_f, feat1}, /*dim=*/1);                 // [rows,2]

  auto Y = torch::zeros({rows, num_classes}, torch::dtype(torch::kFloat32));
  Y.scatter_(1, (ids % num_classes).unsqueeze(1), 1.0f); // one-hot

  return {X, Y};
}

std::vector<int64_t> tensor_row_ids(const torch::Tensor &X) {
  auto ids = X.index({torch::indexing::Slice(), 0}).to(torch::kInt64).contiguous();
  std::vector<int64_t> v(ids.size(0));
  std::memcpy(v.data(), ids.data_ptr<int64_t>(), v.size() * sizeof(int64_t));
  return v;
}

// Helper function to create a simple XYMatrix for testing
std::pair<torch::Tensor, torch::Tensor> create_test_data(int rows, int cols_X, int cols_Y) {
  auto X = torch::rand({rows, cols_X});
  auto Y = torch::randint(0, 2, {rows, cols_Y}).to(torch::kFloat32); // One-hot encoded
  return {X, Y};
}

} // namespace

// ------------ Tensor-based constructors ------------

TEST(XYMatrix_Test, AcceptsValid2DTensorsAndStoresShapes) {
  auto X = make_X(6, 3);
  auto Y = make_one_hot_Y({0, 1, 2, 0, 1, 2}, /*num_classes=*/3);

  XYMatrix xy(std::move(X), std::move(Y));
  EXPECT_EQ(xy.X().sizes(), torch::IntArrayRef({6, 3}));
  EXPECT_EQ(xy.Y().sizes(), torch::IntArrayRef({6, 3}));
}

TEST(XYMatrix_Test, ThrowsOnNon2DInputs) {
  auto X1D = torch::tensor({1.0f, 2.0f}, torch::kFloat32);
  auto Y2D = make_one_hot_Y({0, 1}, 2);
  EXPECT_THROW((XYMatrix(std::move(X1D), torch::clone(Y2D))), std::runtime_error);

  auto X2D = make_X(2, 2);
  auto Y1D = torch::tensor({0.0f, 1.0f}, torch::kFloat32);
  EXPECT_THROW((XYMatrix(std::move(X2D), std::move(Y1D))), std::runtime_error);
}

TEST(XYMatrix_Test, ThrowsOnRowCountMismatch) {
  auto X = make_X(4, 2);
  auto Y = make_one_hot_Y({0, 1, 0}, 2); // 3 rows vs 4
  EXPECT_THROW((XYMatrix(std::move(X), std::move(Y))), std::runtime_error);
}

TEST(XYMatrix_Test, SplitsByLastColXIndex) {
  // data: 6 rows, 5 cols. last_col_X = 2 -> X has 3 cols, Y has 2 cols
  auto data = make_X(6, 5);
  int last_col_X = 2;

  XYMatrix xy(std::move(data), last_col_X);

  EXPECT_EQ(xy.X().size(0), 6);
  EXPECT_EQ(xy.Y().size(0), 6);
  EXPECT_EQ(xy.X().size(1), last_col_X + 1);
  EXPECT_EQ(xy.Y().size(1), 5 - (last_col_X + 1));
}

// ------------ Accessors ------------

TEST(XYMatrix_Test, ReturnConstRefsWithExpectedShapes) {
  auto X = make_X(3, 4);
  auto Y = make_one_hot_Y({0, 1, 2}, 3);

  XYMatrix xy(std::move(X), std::move(Y));
  const auto &XR = xy.X();
  const auto &YR = xy.Y();
  EXPECT_EQ(XR.sizes(), torch::IntArrayRef({3, 4}));
  EXPECT_EQ(YR.sizes(), torch::IntArrayRef({3, 3}));
}

// ------------ Template ctor from AnalyticTableNumeric<T> ------------

TEST(XYMatrix_Test, BuildsFromAnalyticTableNumericFloat) {
  constexpr int64_t rows = 5;
  constexpr int xcols = 2;
  constexpr int ycols = 1;

  auto Xtb = make_numeric_table_T<float>(rows, xcols);
  auto Ytb = make_numeric_table_T<float>(rows, ycols);

  XYMatrix xy(std::move(Xtb), std::move(Ytb));
  EXPECT_EQ(xy.X().sizes(), torch::IntArrayRef({rows, xcols}));
  EXPECT_EQ(xy.Y().sizes(), torch::IntArrayRef({rows, ycols}));
}

// ------------ Stratified split ------------

TEST(XYMatrix_Test, PreservesShapeAndCountsPerClassAt50Percent) {
  // Balanced labels: 3 classes, 4 rows each -> 12 rows total
  // pct_eval = 50 -> expect 2 eval per class, 2 train per class
  std::vector<int> labels = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  auto X = make_X(12, 3);
  auto Y = make_one_hot_Y(labels, 3);

  XYMatrix xy(std::move(X), std::move(Y));
  auto bundle = XYMatrix::stratified_split_from_one_hot(std::move(xy), /*pct_eval=*/50);

  const auto &XYtr = bundle.XY_train();
  const auto &XYev = bundle.XY_eval();

  // Shapes
  EXPECT_EQ(XYtr.X().size(1), 3);
  EXPECT_EQ(XYev.X().size(1), 3);
  EXPECT_EQ(XYtr.Y().size(1), 3);
  EXPECT_EQ(XYev.Y().size(1), 3);

  // All rows partitioned
  EXPECT_EQ(XYtr.X().size(0) + XYev.X().size(0), 12);
  EXPECT_EQ(XYtr.Y().size(0) + XYev.Y().size(0), 12);

  // Label distribution
  auto tr_labels = argmax_labels(XYtr.Y());
  auto ev_labels = argmax_labels(XYev.Y());
  for (int c = 0; c < 3; ++c) {
    EXPECT_EQ(count_label(tr_labels, c), 2) << "class " << c << " in train";
    EXPECT_EQ(count_label(ev_labels, c), 2) << "class " << c << " in eval";
  }
}

TEST(XYMatrix_Test, HandlesZeroAndHundredPercent) {
  std::vector<int> labels = {0, 0, 1, 1};
  auto X = make_X(4, 2);
  auto Y = make_one_hot_Y(labels, 2);

  // 0% eval -> all in train
  {
    XYMatrix xy0(torch::clone(X), torch::clone(Y));
    EXPECT_THROW(XYMatrix::stratified_split_from_one_hot(std::move(xy0), 0), ttb::XYMatrixError);
  }

  // 100% eval -> all in eval
  {
    XYMatrix xy1(std::move(X), std::move(Y));
    EXPECT_THROW(XYMatrix::stratified_split_from_one_hot(std::move(xy1), 100), ttb::XYMatrixError);
  }
}

TEST(XYMatrix_Test, SameSeedProducesSamePermutation) {
  constexpr int64_t rows = 32;
  constexpr int classes = 3;

  auto [X1, Y1] = make_xy_with_ids(rows, classes);
  auto [X2, Y2] = make_xy_with_ids(rows, classes);

  XYMatrix a(std::move(X1), std::move(Y1));
  XYMatrix b(std::move(X2), std::move(Y2));

  a.shuffle(/*seed=*/123);
  b.shuffle(/*seed=*/123);

  EXPECT_TRUE(torch::equal(a.X(), b.X()));
  EXPECT_TRUE(torch::equal(a.Y(), b.Y()));
}

TEST(XYMatrix_Test, DifferentSeedsProduceDifferentPermutation) {
  constexpr int64_t rows = 32;
  constexpr int classes = 3;

  auto [X0, Y0] = make_xy_with_ids(rows, classes);
  auto [X1, Y1] = std::make_pair(X0.clone(), Y0.clone());
  auto [X2, Y2] = std::make_pair(X0.clone(), Y0.clone());

  XYMatrix a(std::move(X1), std::move(Y1));
  XYMatrix b(std::move(X2), std::move(Y2));

  a.shuffle(/*seed=*/123);
  b.shuffle(/*seed=*/124);

  // Highly likely different permutations
  EXPECT_FALSE(torch::equal(a.X(), b.X()));

  // Both are permutations: row-id multiset unchanged
  auto ids_orig = tensor_row_ids(X0);
  auto ids_a = tensor_row_ids(a.X());
  auto ids_b = tensor_row_ids(b.X());
  std::sort(ids_orig.begin(), ids_orig.end());
  std::sort(ids_a.begin(), ids_a.end());
  std::sort(ids_b.begin(), ids_b.end());
  EXPECT_EQ(ids_a, ids_orig);
  EXPECT_EQ(ids_b, ids_orig);
}

TEST(XYMatrix_Test, UnseededKeepsXYAlignmentAndIsPermutation) {
  constexpr int64_t rows = 40;
  constexpr int classes = 4;

  auto [X, Y] = make_xy_with_ids(rows, classes);
  XYMatrix xy(std::move(X), std::move(Y));

  // Keep a copy of original row ids to validate permutation
  auto ids_before = tensor_row_ids(xy.X());

  xy.shuffle(); // no seed (non-deterministic)

  // Shapes unchanged
  EXPECT_EQ(xy.X().sizes(), torch::IntArrayRef({rows, 2}));
  EXPECT_EQ(xy.Y().sizes(), torch::IntArrayRef({rows, classes}));

  // XY alignment preserved: label == id % classes for each row
  auto argmax = xy.Y().argmax(1); // int64
  auto a_idx = argmax.accessor<int64_t, 1>();
  for (int64_t i = 0; i < rows; ++i) {
    int64_t id = static_cast<int64_t>(xy.X().index({i, 0}).item<float>());
    EXPECT_EQ(a_idx[i], id % classes) << "row " << i;
  }

  // Row ids are a permutation of original
  auto ids_after = tensor_row_ids(xy.X());
  std::sort(ids_before.begin(), ids_before.end());
  std::sort(ids_after.begin(), ids_after.end());
  EXPECT_EQ(ids_before, ids_after);
}

TEST(XYMatrix_Test, SplitReturnsCorrectTrainEvalSizes) {
  auto [X, Y] = create_test_data(100, 5, 2); // 100 rows, 5 features, 2 classes
  XYMatrix xy_matrix(std::move(X), std::move(Y));

  // Split with 80% training data
  auto result = XYMatrix::split(std::move(xy_matrix), 20);

  EXPECT_EQ(result.XY_train().X().size(0), 80); // 80% for training
  EXPECT_EQ(result.XY_eval().X().size(0), 20);  // 20% for evaluation
}

TEST(XYMatrix_Test, ShuffleSplitMaintainsXYAlignment) {
  auto [X, Y] = create_test_data(100, 5, 2); // 100 rows, 5 features, 2 classes
  XYMatrix xy_matrix(std::move(X), std::move(Y));

  // Shuffle split with 50% training data
  auto result = XYMatrix::shuffle_split(std::move(xy_matrix), 50);

  EXPECT_EQ(result.XY_train().X().size(0), 50); // 50% for training
  EXPECT_EQ(result.XY_eval().X().size(0), 50);  // 50% for evaluation

  // Check that the labels are still aligned with the features
  auto train_labels = result.XY_train().Y();
  auto eval_labels = result.XY_eval().Y();

  EXPECT_EQ(train_labels.size(0), 50);
  EXPECT_EQ(eval_labels.size(0), 50);
}

TEST(XYMatrix_Test, ShuffleSplitWithSeedProducesSameResult) {
  auto [X, Y] = create_test_data(100, 5, 2); // 100 rows, 5 features, 2 classes
  auto X2 = X.clone();
  auto Y2 = Y.clone(); // 100 rows, 5 features, 2 classes

  XYMatrix xy_matrix(std::move(X), std::move(Y));
  XYMatrix xy_matrix2(std::move(X2), std::move(Y2));

  // Shuffle split with a specific seed
  auto result1 = XYMatrix::shuffle_split(std::move(xy_matrix), 50, 123);
  auto result2 = XYMatrix::shuffle_split(std::move(xy_matrix2), 50, 123);

  // Check that the results are the same
  EXPECT_TRUE(torch::equal(result1.XY_train().X(), result2.XY_train().X()));
  EXPECT_TRUE(torch::equal(result1.XY_eval().X(), result2.XY_eval().X()));
}

TEST(XYMatrix_Test, SplitHandlesEdgeCases) {
  auto [X, Y] = create_test_data(10, 5, 2); // 10 rows, 5 features, 2 classes
  XYMatrix xy_matrix(std::move(X), std::move(Y));

  EXPECT_THROW(auto result = XYMatrix::split(std::move(xy_matrix), 0), ttb::XYMatrixError);
  EXPECT_THROW(auto result = XYMatrix::split(std::move(xy_matrix), 100), ttb::XYMatrixError);
}