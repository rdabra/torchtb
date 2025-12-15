#include <gtest/gtest.h>

#include "TrainingBundle.h"
#include "XYMatrix.h"

#include <torch/torch.h>

using ttb::TrainingBundle;
using ttb::XYMatrix;

namespace {

// Build simple X and Y tensors
torch::Tensor make_X(const std::vector<std::vector<float>> &rows) {
  int64_t r = static_cast<int64_t>(rows.size());
  int64_t c = static_cast<int64_t>(rows.empty() ? 0 : rows.front().size());
  auto X = torch::empty({r, c}, torch::dtype(torch::kFloat32));
  auto a = X.accessor<float, 2>();
  for (int64_t i = 0; i < r; ++i)
    for (int64_t j = 0; j < c; ++j)
      a[i][j] = rows[i][j];
  return X;
}

torch::Tensor createOneHotEncoding(const std::vector<int> &labels, int num_classes) {
  int64_t rows = static_cast<int64_t>(labels.size());
  torch::Tensor Y = torch::zeros({rows, num_classes}, torch::dtype(torch::kFloat32));
  for (int64_t i = 0; i < rows; ++i)
    Y.index_put_({i, labels[i]}, 1.0f);
  return Y;
}

} // namespace

// ------------ Constructor and accessors ------------

TEST(TrainingBundle_Test, StoresGivenXYMatrices) {
  auto Xtr = make_X({{0.f, 1.f}, {5.f, 2.f}, {10.f, 3.f}});
  auto Ytr = createOneHotEncoding({0, 1, 2}, 3);
  auto Xev = make_X({{2.5f, 4.f}, {7.5f, 5.f}});
  auto Yev = createOneHotEncoding({1, 0}, 3);

  XYMatrix XY_train(std::move(Xtr), std::move(Ytr));
  XYMatrix XY_eval(std::move(Xev), std::move(Yev));

  TrainingBundle tb(std::move(XY_train), std::move(XY_eval));

  EXPECT_EQ(tb.XY_train().X().sizes(), torch::IntArrayRef({3, 2}));
  EXPECT_EQ(tb.XY_train().Y().sizes(), torch::IntArrayRef({3, 3}));
  EXPECT_EQ(tb.XY_eval().X().sizes(), torch::IntArrayRef({2, 2}));
  EXPECT_EQ(tb.XY_eval().Y().sizes(), torch::IntArrayRef({2, 3}));
}

// ------------ min_max_normz ------------

TEST(TrainingBundle_Test, MinMaxNormzNormalizesColumnConsistently) {
  auto Xtr = make_X({{0.f, 1.f}, {5.f, 2.f}, {10.f, 3.f}}); // col0 min=0 max=10
  auto Ytr = createOneHotEncoding({0, 1, 2}, 3);
  auto Xev = make_X({{2.5f, 4.f}, {7.5f, 5.f}});
  auto Yev = createOneHotEncoding({1, 0}, 3);

  XYMatrix XY_train(std::move(Xtr), std::move(Ytr));
  XYMatrix XY_eval(std::move(Xev), std::move(Yev));

  TrainingBundle tb(std::move(XY_train), std::move(XY_eval));
  auto [mn, mx] = tb.min_max_normz(/*X_col=*/0);
  EXPECT_DOUBLE_EQ(mn, 0.0);
  EXPECT_DOUBLE_EQ(mx, 10.0);

  auto a_tr = tb.XY_train().X().accessor<float, 2>();
  EXPECT_FLOAT_EQ(a_tr[0][0], 0.0f);
  EXPECT_FLOAT_EQ(a_tr[1][0], 0.5f);
  EXPECT_FLOAT_EQ(a_tr[2][0], 1.0f);

  auto a_ev = tb.XY_eval().X().accessor<float, 2>();
  EXPECT_FLOAT_EQ(a_ev[0][0], 0.25f);
  EXPECT_FLOAT_EQ(a_ev[1][0], 0.75f);
}

TEST(TrainingBundle_Test, MinMaxNormzThrowsOnOutOfBoundsColumn) {
  auto Xtr = make_X({{0.f, 1.f}});
  auto Ytr = createOneHotEncoding({0}, 2);
  auto Xev = make_X({{2.5f, 4.f}});
  auto Yev = createOneHotEncoding({1}, 2);

  TrainingBundle tb(XYMatrix(std::move(Xtr), std::move(Ytr)),
                    XYMatrix(std::move(Xev), std::move(Yev)));

  EXPECT_THROW(tb.min_max_normz(/*X_col=*/2), ttb::TrainingBundleError);
}

TEST(TrainingBundle_Test, MinMaxNormzThrowsOnNonFloatingPoint) {
  auto Xtr = torch::tensor({{0, 1}, {5, 2}, {10, 3}}, torch::dtype(torch::kInt64));
  auto Ytr = createOneHotEncoding({0, 1, 2}, 3);
  auto Xev = torch::tensor({{2, 4}, {7, 5}}, torch::dtype(torch::kInt64));
  auto Yev = createOneHotEncoding({1, 0}, 3);

  TrainingBundle tb(XYMatrix(std::move(Xtr), std::move(Ytr)),
                    XYMatrix(std::move(Xev), std::move(Yev)));

  EXPECT_THROW(tb.min_max_normz(/*X_col=*/0), ttb::TrainingBundleError);
}

// ------------ z_score_normz ------------

TEST(TrainingBundle_Test, ZScoreNormzNormalizesColumnConsistently) {
  auto Xtr = make_X({{0.f, 1.f}, {5.f, 2.f}, {10.f, 3.f}}); // mean=5, sigma=sqrt(50/3)
  auto Ytr = createOneHotEncoding({0, 1, 2}, 3);
  auto Xev = make_X({{2.5f, 4.f}, {7.5f, 5.f}});
  auto Yev = createOneHotEncoding({1, 0}, 3);

  TrainingBundle tb(XYMatrix(std::move(Xtr), std::move(Ytr)),
                    XYMatrix(std::move(Xev), std::move(Yev)));

  auto [mean, sigma] = tb.z_score_normz(/*X_col=*/0);
  EXPECT_DOUBLE_EQ(mean, 5.0);
  EXPECT_NEAR(sigma, std::sqrt(50.0 / 3.0), 1e-6);

  auto a_tr = tb.XY_train().X().accessor<float, 2>();
  EXPECT_NEAR(a_tr[0][0], -5.0f / static_cast<float>(sigma), 1e-6);
  EXPECT_NEAR(a_tr[1][0], 0.0f, 1e-6);
  EXPECT_NEAR(a_tr[2][0], 5.0f / static_cast<float>(sigma), 1e-6);

  auto a_ev = tb.XY_eval().X().accessor<float, 2>();
  EXPECT_NEAR(a_ev[0][0], (2.5f - 5.0f) / static_cast<float>(sigma), 1e-6);
  EXPECT_NEAR(a_ev[1][0], (7.5f - 5.0f) / static_cast<float>(sigma), 1e-6);
}

TEST(TrainingBundle_Test, ZScoreNormzThrowsOnOutOfBoundsColumn) {
  auto Xtr = make_X({{0.f}});
  auto Ytr = createOneHotEncoding({0}, 2);
  auto Xev = make_X({{2.5f}});
  auto Yev = createOneHotEncoding({1}, 2);

  TrainingBundle tb(XYMatrix(std::move(Xtr), std::move(Ytr)),
                    XYMatrix(std::move(Xev), std::move(Yev)));

  EXPECT_THROW(tb.z_score_normz(/*X_col=*/1), ttb::TrainingBundleError);
}

TEST(TrainingBundle_Test, ZScoreNormzThrowsOnNonFloatingPoint) {
  auto Xtr = torch::tensor({{0, 1}, {5, 2}, {10, 3}}, torch::dtype(torch::kInt64));
  auto Ytr = createOneHotEncoding({0, 1, 2}, 3);
  auto Xev = torch::tensor({{2, 4}, {7, 5}}, torch::dtype(torch::kInt64));
  auto Yev = createOneHotEncoding({1, 0}, 3);

  TrainingBundle tb(XYMatrix(std::move(Xtr), std::move(Ytr)),
                    XYMatrix(std::move(Xev), std::move(Yev)));

  EXPECT_THROW(tb.z_score_normz(/*X_col=*/0), ttb::TrainingBundleError);
}
