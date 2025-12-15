#include "XYMatrix.h"
#include "AnalyticTableNumeric.h"
#include "Converter.h"
#include "TrainingBundle.h"

#include <algorithm>
#include <random>

template <utl::NumericType T>
ttb::XYMatrix::XYMatrix(ttb::AnalyticTableNumeric<T> &&data, int last_X_col) {
  auto my_data = std::move(data);

  if (last_X_col < 0 || last_X_col >= my_data.n_cols() - 1)
    throw std::runtime_error("Invalid last X column index!");

  auto my_data_T = ttb::Converter::torch_tensor<T>(std::move(my_data));

  this->update_X_Y(std::move(my_data_T), last_X_col);
};

template <utl::NumericType T>
ttb::XYMatrix::XYMatrix(ttb::TbNumeric<T> &&X, ttb::TbNumeric<T> &&Y) {
  auto my_X = std::move(X);
  auto my_Y = std::move(Y);

  if (my_X.n_rows() != my_Y.n_rows())
    throw std::runtime_error("Incompatible tensors!");

  auto X_T = ttb::Converter::torch_tensor<T>(std::move(my_X));
  auto Y_T = ttb::Converter::torch_tensor<T>(std::move(my_Y));
  _X = utl::new_unp<torch::Tensor>(std::move(X_T));
  _Y = utl::new_unp<torch::Tensor>(std::move(Y_T));
}

ttb::XYMatrix::XYMatrix(torch::Tensor &&data, int last_X_col) {
  auto my_data = std::move(data);

  if (my_data.dim() != 2)
    throw std::runtime_error("Tensor is not second order!");

  if (last_X_col < 0 || last_X_col >= my_data.size(1) - 1)
    throw std::runtime_error("Invalid last X column index!");

  this->update_X_Y(std::move(my_data), last_X_col);
}

ttb::XYMatrix::XYMatrix(torch::Tensor &&X, torch::Tensor &&Y) {
  auto my_X = std::move(X);
  auto my_Y = std::move(Y);

  if (my_X.dim() != 2 || my_Y.dim() != 2)
    throw std::runtime_error("Both tensors must be of second order!");

  if (my_X.size(0) != my_Y.size(0))
    throw std::runtime_error("Incompatible tensors!");
  _X = utl::new_unp<torch::Tensor>(std::move(my_X));
  _Y = utl::new_unp<torch::Tensor>(std::move(my_Y));
}

[[nodiscard]] int64_t ttb::XYMatrix::n_rows() const {
  return _X->size(0);
}

[[nodiscard]] int64_t ttb::XYMatrix::n_cols() const {
  return _X->size(1) + _Y->size(1);
}

void ttb::XYMatrix::shuffle(std::optional<unsigned> seed) {
  auto XY = torch::cat({*_X, *_Y}, 1);
  auto n_rows = XY.size(0);
  torch::Tensor shuffled_indices;
  if (!seed.has_value())
    shuffled_indices = torch::randperm(n_rows, XY.options().dtype(torch::kLong));
  else {
    auto gen = torch::make_generator<torch::CPUGeneratorImpl>(seed.value());
    shuffled_indices = torch::randperm(n_rows, gen, XY.options().dtype(torch::kLong));
  }

  XY = XY.index_select(0, shuffled_indices);
  auto last_col_X = static_cast<int>(_X->size(1) - 1);

  this->update_X_Y(std::move(XY), last_col_X);
}

namespace stratified_split_from_one_hot {

auto stratify_row_indices_by_label(const torch::Tensor &Y, int pct_eval,
                                   std::optional<unsigned> seed) {
  auto args = Y.argmax(1);

  std::unordered_map<int64_t, std::vector<int64_t>> train_row_indices_per_label;
  std::unordered_map<int64_t, std::vector<int64_t>> eval_row_indices_per_label;

  auto a_args = args.accessor<int64_t, 1>();
  for (int64_t i{0}; std::cmp_less(i, args.size(0)); ++i)
    train_row_indices_per_label[a_args[i]].emplace_back(i);

  std::mt19937_64 engine;
  if (!seed.has_value())
    engine = std::mt19937_64{std::random_device{}()};
  else
    engine = std::mt19937_64{seed.value()};
  for (auto &[label, rows] : train_row_indices_per_label) {
    auto n_eval_rows = static_cast<int64_t>(rows.size() * pct_eval) / 100;
    for (int64_t j{0}; j < n_eval_rows; ++j) {
      std::ranges::shuffle(rows, engine);
      eval_row_indices_per_label[label].emplace_back(rows.back());
      rows.pop_back();
    }
  }

  return std::make_pair(train_row_indices_per_label, eval_row_indices_per_label);
}

auto stack_stratified_rows(const std::unordered_map<int64_t, std::vector<int64_t>> &label_rows,
                           const torch::Tensor &X, const torch::Tensor &Y) {
  std::vector<torch::Tensor> X_row_list;
  std::vector<torch::Tensor> Y_row_list;
  for (auto &[label, rows] : label_rows)
    for (size_t i{0}; i < rows.size(); ++i) {
      Y_row_list.emplace_back(Y[rows[i]].clone());
      X_row_list.emplace_back(X[rows[i]].clone());
    }

  auto stacked_X = torch::vstack(X_row_list);
  auto stacked_Y = torch::vstack(Y_row_list);

  return std::make_pair(stacked_X, stacked_Y);
}

} // namespace stratified_split_from_one_hot

ttb::TrainingBundle ttb::XYMatrix::stratified_split_from_one_hot(XYMatrix &&XY_matrix, int pct_eval,
                                                                 std::optional<unsigned> seed) {
  auto my_XY_matrix = std::move(XY_matrix);

  if (pct_eval <= 0 || pct_eval >= 100)
    throw ttb::XYMatrixError("Percentage out of bounds");

  auto &X = my_XY_matrix.X();
  auto &Y = my_XY_matrix.Y();

  auto [label_rows_train, label_rows_eval] =
      stratified_split_from_one_hot::stratify_row_indices_by_label(Y, pct_eval, seed);

  auto [X_train, Y_train] =
      stratified_split_from_one_hot::stack_stratified_rows(label_rows_train, X, Y);

  auto [X_eval, Y_eval] =
      stratified_split_from_one_hot::stack_stratified_rows(label_rows_eval, X, Y);

  return {ttb::XYMatrix{std::move(X_train), std::move(Y_train)},
          ttb::XYMatrix{std::move(X_eval), std::move(Y_eval)}};
}

ttb::TrainingBundle
ttb::XYMatrix::shuffle_stratified_split_from_one_hot(XYMatrix &&XY_matrix, int pct_eval,
                                                     std::optional<unsigned> seed) {
  auto my_XY_matrix = std::move(XY_matrix);
  if (pct_eval <= 0 || pct_eval >= 100)
    throw ttb::XYMatrixError("Percentage out of bounds");

  my_XY_matrix.shuffle(seed);

  return ttb::XYMatrix::shuffle_split(std::move(my_XY_matrix), pct_eval);
}

ttb::TrainingBundle ttb::XYMatrix::split(XYMatrix &&XY_matrix, int pct_eval) {
  auto my_XY_matrix = std::move(XY_matrix);

  if (pct_eval <= 0 || pct_eval >= 100)
    throw ttb::XYMatrixError("Percentage out of bounds");

  auto X = my_XY_matrix.X().clone();
  auto Y = my_XY_matrix.Y().clone();

  auto train_size = static_cast<int64_t>(X.size(0) * (100 - pct_eval)) / 100;
  auto X_train = X.narrow_copy(0, 0, train_size);
  auto Y_train = Y.narrow_copy(0, 0, train_size);
  auto X_eval = X.narrow_copy(0, train_size, X.size(0) - train_size);
  auto Y_eval = Y.narrow_copy(0, train_size, X.size(0) - train_size);

  return {ttb::XYMatrix{std::move(X_train), std::move(Y_train)},
          ttb::XYMatrix{std::move(X_eval), std::move(Y_eval)}};
}

ttb::TrainingBundle ttb::XYMatrix::shuffle_split(XYMatrix &&XY_matrix, int pct_eval,
                                                 std::optional<unsigned> seed) {
  auto my_XY_matrix = std::move(XY_matrix);

  if (pct_eval <= 0 || pct_eval >= 100)
    throw ttb::XYMatrixError("Percentage out of bounds");

  my_XY_matrix.shuffle(seed);

  return ttb::XYMatrix::split(std::move(my_XY_matrix), pct_eval);
}

void ttb::XYMatrix::update_X_Y(torch::Tensor &&XY, int last_col_X) {
  auto my_XY = std::move(XY);
  auto X = my_XY.narrow_copy(1, 0, last_col_X + 1);
  auto Y = my_XY.narrow_copy(1, last_col_X + 1, my_XY.size(1) - last_col_X - 1);

  _X = utl::new_unp<torch::Tensor>(std::move(X));
  _Y = utl::new_unp<torch::Tensor>(std::move(Y));
}

[[nodiscard]] torch::Tensor ttb::XYMatrix::reshape(const std::unique_ptr<torch::Tensor> &tensor,
                                                   const std::vector<int64_t> &dims) const {
  auto n_elements_dims = std::accumulate(std::begin(dims), std::end(dims), 1,
                                         [](int a, int b) -> int { return a * b; });
  if (n_elements_dims != tensor->numel())
    throw XYMatrixError("Incompatible dimensions");

  return torch::reshape(*tensor, dims);
}

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define INSTANTIATE_XYTABLE_TEMPLATES(T)                                                           \
  template ttb::XYMatrix::XYMatrix(ttb::AnalyticTableNumeric<T> &&, int);                          \
  template ttb::XYMatrix::XYMatrix(ttb::TbNumeric<T> &&, ttb::TbNumeric<T> &&);

INSTANTIATE_XYTABLE_TEMPLATES(int)
INSTANTIATE_XYTABLE_TEMPLATES(int64_t)
INSTANTIATE_XYTABLE_TEMPLATES(float)
INSTANTIATE_XYTABLE_TEMPLATES(double)

#undef INSTANTIATE_XYTABLE_TEMPLATES
