#ifndef TRAININGBUNDLE_H
#define TRAININGBUNDLE_H
#pragma once

#include "XYMatrix.h"

namespace ttb {
class TrainingBundle {
  public:
    TrainingBundle(ttb::XYMatrix &&XY_train, ttb::XYMatrix &&XY_eval)
        : _XY_train{std::move(XY_train)}, _XY_eval{std::move(XY_eval)} {}

    [[nodiscard]] const ttb::XYMatrix &XY_train() const { return this->_XY_train; }
    [[nodiscard]] const ttb::XYMatrix &XY_eval() const { return this->_XY_eval; }

    [[nodiscard]] const torch::Tensor &X_train() const { return this->_XY_train.X(); }
    [[nodiscard]] const torch::Tensor &Y_train() const { return this->_XY_train.Y(); }
    [[nodiscard]] const torch::Tensor &X_eval() const { return this->_XY_eval.X(); }
    [[nodiscard]] const torch::Tensor &Y_eval() const { return this->_XY_eval.Y(); }

    /**
     * @brief Performs min-max normalization of the specifed X column (tensors must be floating
     * point)
     *
     * @param X_col Column to be normalized
     * @return std::pair<double, double> Min and Max values of the column
     */
    std::pair<double, double> min_max_normz(int X_col);

    /**
     * @brief Performs the z_score normalization of the specified X column (tensors must be floating
     * point)
     *
     * @param X_col Column to be normalized
     * @return std::pair<double, double> Mean and Standard Deviation of the column
     */
    std::pair<double, double> z_score_normz(int X_col);

  private:
    ttb::XYMatrix _XY_train;
    ttb::XYMatrix _XY_eval;

    void check_X_index_floating_point(int X_col);
};

class TrainingBundleError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace ttb

#endif