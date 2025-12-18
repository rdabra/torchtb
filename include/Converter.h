#ifndef TORCHCONVERTER_H
#define TORCHCONVERTER_H
#pragma once

#include "AnalyticTableNumeric.h"
#include "CSV_IO.h"
#include "detail/utils.h"
#include <torch/data/dataloader_options.h>

#include "Parquet_IO.h"
#include <ATen/core/TensorBody.h>
#include <arrow/array/array_base.h>
#include <torch/data/dataloader.h>

namespace ttb {

class Converter {
  public:
    Converter() = delete;
    Converter(const Converter &) = delete;
    Converter(Converter &&) = delete;
    Converter &operator=(const Converter &) = delete;
    Converter &operator=(Converter &&) = delete;
    ~Converter() = default;

    template <utl::NumericType T>
    static torch::Tensor torch_tensor(ttb::AnalyticTableNumeric<T> &&data);

    template <utl::NumericType T>
    static torch::Tensor torch_tensor(ttb::CSV_IO &&reader);

    template <utl::NumericType T>
    static torch::Tensor torch_tensor(ttb::Parquet_IO &&reader);

    template <utl::NumericType T>
    static ttb::AnalyticTableNumeric<T> analytic_table(torch::Tensor &&tensor);
};

class ConverterError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace ttb
#endif