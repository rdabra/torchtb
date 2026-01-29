#ifndef TORCHCONVERTER_H
#define TORCHCONVERTER_H
#pragma once

#include "AnalyticTableNumeric.h"
#include "IO_FileCSV.h"
#include "detail/utils.h"
#include <torch/data/dataloader_options.h>

#include "IO_FileParquet.h"
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

    template <ttb::utl::NumericType T>
    static torch::Tensor torch_tensor(ttb::AnalyticTableNumeric<T> &&data);

    template <ttb::utl::NumericType T>
    static torch::Tensor torch_tensor(ttb::IO_FileCSV &&reader);

    template <ttb::utl::NumericType T>
    static torch::Tensor torch_tensor(ttb::IO_FileParquet &&reader);

    template <ttb::utl::NumericType T>
    static ttb::AnalyticTableNumeric<T> analytic_table(torch::Tensor &&tensor);
};

class ConverterError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace ttb
#endif
