#ifndef XYTABLE_H
#define XYTABLE_H
#pragma once
#include <memory>

#include "AnalyticTableNumeric.h"

namespace ttb {

class TrainingBundle;

class XYMatrix {
  public:
    XYMatrix() = delete;
    XYMatrix(const XYMatrix &) = delete;
    XYMatrix(XYMatrix &&) = default;
    XYMatrix &operator=(const XYMatrix &) = delete;
    XYMatrix &operator=(XYMatrix &&) = default;
    ~XYMatrix() = default;

    template <utl::NumericType T>
    XYMatrix(ttb::TbNumeric<T> &&data, int last_col_X);

    template <utl::NumericType T>
    XYMatrix(ttb::TbNumeric<T> &&X, ttb::TbNumeric<T> &&Y);

    XYMatrix(torch::Tensor &&data, int last_col_X);
    XYMatrix(torch::Tensor &&X, torch::Tensor &&Y);

    [[nodiscard]] int64_t n_rows() const;
    [[nodiscard]] int64_t n_cols() const;

    [[nodiscard]] const torch::Tensor &X() const { return *_X; };
    [[nodiscard]] const torch::Tensor &Y() const { return *_Y; };

    void shuffle(std::optional<unsigned> seed = std::nullopt);

    static ttb::TrainingBundle
    stratified_split_from_one_hot(XYMatrix &&XY_matrix, int pct_eval,
                                  std::optional<unsigned> seed = std::nullopt);

    static ttb::TrainingBundle
    shuffle_stratified_split_from_one_hot(XYMatrix &&XY_matrix, int pct_eval,
                                          std::optional<unsigned> seed = std::nullopt);

    static ttb::TrainingBundle split(XYMatrix &&XY_matrix, int pct_eval);

    static ttb::TrainingBundle shuffle_split(XYMatrix &&XY_matrix, int pct_eval,
                                             std::optional<unsigned> seed = std::nullopt);

  private:
    std::unique_ptr<torch::Tensor> _X{nullptr};
    std::unique_ptr<torch::Tensor> _Y{nullptr};

    void update_X_Y(torch::Tensor &&XY, int last_col_X);
    [[nodiscard]] torch::Tensor reshape(const std::unique_ptr<torch::Tensor> &tensor,
                                        const std::vector<int64_t> &dims) const;
};

class TrainingBundle {
  public:
    TrainingBundle(ttb::XYMatrix &&XY_train, ttb::XYMatrix &&XY_eval)
        : _XY_train{std::move(XY_train)}, _XY_eval{std::move(XY_eval)} {}

    [[nodiscard]] const ttb::XYMatrix &XY_train() const { return this->_XY_train; }
    [[nodiscard]] const ttb::XYMatrix &XY_eval() const { return this->_XY_eval; }

  private:
    ttb::XYMatrix _XY_train;
    ttb::XYMatrix _XY_eval;
};

class XYMatrixError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace ttb
#endif