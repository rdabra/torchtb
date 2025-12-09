#include "Converter.h"
#include "AnalyticTable.h"
#include "detail/utils.h"

#include <ATen/ops/from_blob.h>
#include <arrow/api.h>
#include <arrow/array/data.h>
#include <arrow/array/util.h>
#include <arrow/buffer.h>
#include <arrow/chunked_array.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <c10/core/TensorOptions.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <torch/data/dataloader.h>
#include <torch/data/dataloader_options.h>
#include <torch/data/datasets/tensor.h>
#include <torch/types.h>

namespace torch_tensor {

template <utl::NumericType T>
torch::Tensor to_tensor(const utl::shp<arrow::Array> &arr) {
  if (arr->null_count() != 0)
    throw std::runtime_error("Column has nulls");

  auto casted_col = std::static_pointer_cast<utl::ArrowArrayType<T>>(arr);
  auto torch_type = utl::torch_type<T>();
  auto opt = torch::TensorOptions().dtype(torch_type);
  auto tensor = torch::empty({casted_col->length(), 1}, opt);
  std::memcpy(tensor.template data_ptr<T>(), casted_col->raw_values(),
              casted_col->length() * sizeof(T));
  return tensor;
}

} // namespace torch_tensor

template <utl::NumericType T>
torch::Tensor ttb::Converter::torch_tensor(ttb::AnalyticTableNumeric<T> &&data) {
  auto my_data = std::move(data);
  std::vector<torch::Tensor> tensors;
  for (int i{0}; i < my_data.n_cols(); ++i) {
    auto column = my_data.copy_cols({i});
    auto arr = column->arrow_table()->column(0)->chunk(0);
    tensors.emplace_back(torch_tensor::to_tensor<T>(arr));
  }
  my_data.reset();

  // Concatenate tensors along axis (order) 1
  return torch::cat(tensors, 1);
}

template <utl::NumericType T>
std::expected<ttb::AnalyticTableNumeric<T>, utl::ReturnCode>
ttb::Converter::analytic_table(torch::Tensor &&tensor) {
  if (tensor.sizes().size() != 2)
    throw ttb::ConverterError("Tensor is not of second order");
  if (!tensor.device().is_cpu())
    throw ttb::ConverterError("Tensor is not stored in CPU");

  auto my_tensor = std::move(tensor);
  my_tensor = my_tensor.to(utl::torch_type<T>());

  auto n_rows = my_tensor.size(0);
  auto n_cols = my_tensor.size(1);

  /// Letting arrow manage memory
  auto pool = arrow::default_memory_pool();

  std::vector<utl::shp<arrow::Field>> fields;
  std::vector<utl::shp<arrow::ChunkedArray>> cols;

  for (int64_t j{0}; j < n_cols; ++j) {
    auto column = my_tensor.index({torch::indexing::Slice(), j}).contiguous();

    auto r_buf = arrow::AllocateBuffer(n_rows * int64_t(sizeof(T)), pool);
    if (!r_buf.ok())
      return std::unexpected(utl::map_status(r_buf.status()));

    utl::shp<arrow::Buffer> buf = r_buf.MoveValueUnsafe();
    std::memcpy(buf->mutable_data(), column.data_ptr<T>(), n_rows * sizeof(T));

    auto array = std::make_shared<utl::ArrowArrayType<T>>(n_rows, buf, nullptr, 0);

    std::string name = "col_" + std::to_string(j + 1);
    fields.emplace_back(arrow::field(name, utl::arrow_dtype<T>()));
    cols.emplace_back(std::make_shared<arrow::ChunkedArray>(array));
  }

  auto schema = arrow::schema(fields);

  return ttb::AnalyticTableNumeric<T>{arrow::Table::Make(schema, cols, n_rows)};
};

template <utl::NumericType T>
std::expected<torch::Tensor, utl::ReturnCode> ttb::Converter::torch_tensor(ttb::CSV_IO &&reader) {
  auto my_reader = std::move(reader);

  auto r_read = my_reader.read();
  if (!r_read)
    return std::unexpected(r_read.error());

  return ttb::Converter::torch_tensor<T>(std::move(r_read.value()));
}

template <utl::NumericType T>
std::expected<torch::Tensor, utl::ReturnCode>
ttb::Converter::torch_tensor(ttb::Parquet_IO &&reader) {
  auto my_reader = std::move(reader);
  auto r_data = my_reader.read();
  if (!r_data)
    return std::unexpected(r_data.error());

  return ttb::Converter::torch_tensor<T>(std::move(r_data.value()));
}

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define INSTANTIATE_CONVERTER_FUNCS(T)                                                             \
  template torch::Tensor ttb::Converter::torch_tensor(ttb::AnalyticTableNumeric<T> &&);            \
  template std::expected<ttb::AnalyticTableNumeric<T>, utl::ReturnCode>                            \
  ttb::Converter::analytic_table(torch::Tensor &&t);                                               \
  template std::expected<torch::Tensor, utl::ReturnCode> ttb::Converter::torch_tensor<T>(          \
      ttb::CSV_IO &&);                                                                             \
  template std::expected<torch::Tensor, utl::ReturnCode> ttb::Converter::torch_tensor<T>(          \
      ttb::Parquet_IO &&);

INSTANTIATE_CONVERTER_FUNCS(int)
INSTANTIATE_CONVERTER_FUNCS(int64_t)
INSTANTIATE_CONVERTER_FUNCS(float)
INSTANTIATE_CONVERTER_FUNCS(double)

#undef INSTANTIATE_CONVERTER_FUNCS
