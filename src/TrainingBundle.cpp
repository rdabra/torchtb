#include "TrainingBundle.h"

std::pair<double, double> ttb::TrainingBundle::min_max_normz(int X_col) {
  this->check_X_index_floating_point(X_col);

  auto &X_train = _XY_train.X();

  double min_val{0.0};
  double max_val{0.0};

  AT_DISPATCH_FLOATING_TYPES(X_train.scalar_type(), "min_max_norm", [&] -> void {
    using cpp_type = scalar_t; // C++ type provided by the macro
    auto a_X_train = X_train.accessor<cpp_type, 2>();

    min_val = static_cast<double>(std::numeric_limits<cpp_type>::max());
    max_val = static_cast<double>(std::numeric_limits<cpp_type>::lowest());

    auto n_rows_train = X_train.size(0);
    for (int64_t i{0}; i < n_rows_train; ++i) {
      auto elem = a_X_train[i][X_col];
      if (elem >= max_val)
        max_val = elem;
      if (elem <= min_val)
        min_val = elem;
    }

    auto &X_eval = _XY_eval.X();
    auto a_X_eval = X_eval.accessor<cpp_type, 2>();

    auto n_rows_eval = X_eval.size(0);
    if (utl::is_zero(min_val - max_val)) {
      for (int64_t i{0}; i < n_rows_train; ++i)
        a_X_train[i][X_col] = 0.0;
      for (int64_t i{0}; i < n_rows_eval; ++i)
        a_X_eval[i][X_col] = 0.0;
      return;
    }

    auto multiplier = static_cast<cpp_type>(1.0 / (max_val - min_val));
    for (int64_t i{0}; i < n_rows_train; ++i)
      a_X_train[i][X_col] = (a_X_train[i][X_col] - min_val) * multiplier;

    for (int64_t i{0}; i < n_rows_eval; ++i)
      a_X_eval[i][X_col] = (a_X_eval[i][X_col] - min_val) * multiplier;
  });

  return {min_val, max_val};
}

std::pair<double, double> ttb::TrainingBundle::z_score_normz(int X_col) {
  this->check_X_index_floating_point(X_col);

  auto &X_train = _XY_train.X();

  double mu{0.0};
  double sigma{0.0};

  AT_DISPATCH_FLOATING_TYPES(X_train.scalar_type(), "z_score_norm", [&] -> void {
    using cpp_type = scalar_t; // C++ type provided by the macro
    auto a_X_train = X_train.accessor<cpp_type, 2>();

    auto n_rows_train = X_train.size(0);
    for (int64_t i{0}; i < n_rows_train; ++i)
      mu += a_X_train[i][X_col];
    mu /= n_rows_train;

    for (int64_t i{0}; i < n_rows_train; ++i) {
      auto dif = a_X_train[i][X_col] - mu;
      sigma += dif * dif;
    }

    sigma = std::sqrt(sigma / n_rows_train);

    auto &X_eval = _XY_eval.X();
    auto a_X_eval = X_eval.accessor<cpp_type, 2>();

    auto n_rows_eval = X_eval.size(0);
    if (utl::is_zero(sigma)) {
      for (int64_t i{0}; i < n_rows_train; ++i)
        a_X_train[i][X_col] = 0.0;
      for (int64_t i{0}; i < n_rows_eval; ++i)
        a_X_eval[i][X_col] = 0.0;
      return;
    }

    auto multiplier = static_cast<cpp_type>(1.0 / sigma);
    for (int64_t i{0}; i < n_rows_train; ++i)
      a_X_train[i][X_col] = (a_X_train[i][X_col] - mu) * multiplier;

    for (int64_t i{0}; i < n_rows_eval; ++i)
      a_X_eval[i][X_col] = (a_X_eval[i][X_col] - mu) * multiplier;
  });

  return {mu, sigma};
}

void ttb::TrainingBundle::check_X_index_floating_point(int X_col) {
  auto &X_train = _XY_train.X();

  if (X_col < 0 || X_col >= X_train.size(1))
    throw TrainingBundleError("Index out of bounds");

  if (!X_train.is_floating_point())
    throw TrainingBundleError("Matrices are not floating point");
}
