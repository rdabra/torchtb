#include "AnalyticTableNumeric.h"

#include <algorithm>

namespace to_dtype {

utl::ReturnCode cast_table(utl::shp<arrow::Table> &arrow_tb, utl::shp<arrow::DataType> type) {
  std::vector<utl::shp<arrow::ChunkedArray>> casted_columns;
  std::vector<utl::shp<arrow::Field>> casted_fields;
  for (int i{0}; i < arrow_tb->num_columns(); ++i) {
    auto column = arrow_tb->column(i);

    arrow::compute::CastOptions cast_options;
    cast_options.to_type = type;
    cast_options.allow_int_overflow = false;
    cast_options.allow_float_truncate = true;

    auto casted_datum = arrow::compute::Cast(arrow::Datum(column), cast_options);
    if (!casted_datum.ok())
      return utl::map_status(casted_datum.status());

    casted_fields.emplace_back(arrow::field(arrow_tb->field(i)->name(), type));
    casted_columns.emplace_back(casted_datum.MoveValueUnsafe().chunked_array());
  }

  auto casted_table = arrow::Table::Make(arrow::schema(casted_fields), casted_columns);

  arrow_tb = casted_table;
  return utl::ReturnCode::Ok;
}

} // namespace to_dtype

template <utl::NumericType T>
utl::ReturnCode ttb::AnalyticTableNumeric<T>::to_dtype() {
  return to_dtype::cast_table(_arrow_tb, utl::arrow_dtype<T>());
}

template <utl::NumericType T>
ttb::AnalyticTableNumeric<T>::AnalyticTableNumeric(
    std::unordered_map<std::string, std::vector<T>> &&field_and_data)
    : _arrow_dtype(utl::arrow_dtype<T>()) {
  // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
  _arrow_tb = make_numeric_table(std::move(field_and_data));
}

namespace make_numeric_table {

template <utl::NumericType T>
std::pair<std::vector<utl::shp<arrow::Field>>, std::vector<utl::shp<arrow::Array>>>
make_fields_columns(const std::unordered_map<std::string, std::vector<T>> &field_col_data) {
  auto first_item = std::begin(field_col_data);
  auto n_rows{first_item->second.size()};

  auto n_fields = field_col_data.size();
  std::vector<utl::shp<arrow::Field>> fields;
  std::vector<utl::shp<arrow::Array>> columns;
  fields.reserve(n_fields);
  columns.reserve(n_fields);

  auto dtype = utl::arrow_dtype<T>();
  for (auto &[field, col_data] : field_col_data) {
    fields.emplace_back(arrow::field(field, dtype));

    auto builder = utl::new_unp<utl::ArrowBuilderType<T>>(arrow::default_memory_pool());
    auto status = builder->Resize(n_rows);
    if (!status.ok())
      throw ttb::AnalyticTableError("Could not mount arrow column!");

    for (int64_t i{0}; std::cmp_less(i, n_rows); ++i)
      builder->UnsafeAppend(col_data[i]);

    auto r_array = builder->Finish();
    if (!r_array.ok())
      throw ttb::AnalyticTableError("Could not finish arrow column!");
    columns.emplace_back(r_array.MoveValueUnsafe());
  }
  auto schema = arrow::schema(fields);

  return {fields, columns};
}

} // namespace make_numeric_table

template <utl::NumericType T>
utl::shp<arrow::Table> ttb::AnalyticTableNumeric<T>::make_numeric_table(
    std::unordered_map<std::string, std::vector<T>> &&field_and_data) {
  auto my_field_and_data = std::move(field_and_data);

  auto first_item = std::begin(my_field_and_data);
  auto n_rows{first_item->second.size()};
  for (auto &[field, col_data] : my_field_and_data)
    if (col_data.size() != n_rows)
      throw ttb::AnalyticTableError("Inconsistent col_data_size");

  auto [fields, columns] = make_numeric_table::make_fields_columns(my_field_and_data);

  auto schema = arrow::schema(fields);

  return arrow::Table::Make(schema, columns, n_rows);
}

namespace argmax {

template <utl::NumericType T>
std::vector<int64_t> argmax_row(const utl::shp<arrow::Table> &arrow_tb) {
  auto n_cols = arrow_tb->num_columns();

  std::vector<int64_t> resp;
  resp.reserve(n_cols);

  for (int j{0}; j < n_cols; ++j) {
    int64_t max_idx = 0;
    int64_t global_idx = 0;
    auto max_val = std::numeric_limits<T>::lowest();
    auto chunks = arrow_tb->column(j)->chunks();
    for (auto chunked_col : chunks) {
      auto array = std::static_pointer_cast<utl::ArrowArrayType<T>>(chunked_col);
      for (int64_t i{0}; i < array->length(); ++i, ++global_idx) {
        auto val = array->Value(i);
        if (val > max_val) {
          max_val = val;
          max_idx = global_idx;
        }
      }
    }
    resp.emplace_back(max_idx);
  }

  return resp;
}

template <utl::NumericType T>
std::vector<utl::shp<utl::ArrowArrayType<T>>>
fragment_columns(const utl::shp<arrow::Table> &arrow_tb) {
  std::vector<utl::shp<utl::ArrowArrayType<T>>> resp;
  auto n_cols = arrow_tb->num_columns();

  for (int64_t j{0}; j < n_cols; ++j) {
    auto chunks = arrow_tb->column(static_cast<int>(j))->chunks();
    auto maybe_arr = arrow::Concatenate(chunks, arrow::default_memory_pool());
    if (!maybe_arr.ok())
      throw ttb::DataTableNumericError(maybe_arr.status().ToString());
    auto arr = std::static_pointer_cast<utl::ArrowArrayType<T>>(*maybe_arr);
    resp.emplace_back(std::move(arr));
  }
  return resp;
}

template <utl::NumericType T>
std::vector<int64_t> argmax_col(const utl::shp<arrow::Table> &arrow_tb) {
  auto n_rows = arrow_tb->num_rows();
  auto n_cols = arrow_tb->num_columns();

  std::vector<int64_t> resp;
  resp.reserve(n_rows);

  auto columns = argmax::fragment_columns<T>(arrow_tb);

  for (int64_t i{0}; i < n_rows; ++i) {
    auto max_idx = 0;
    auto max_val = columns[max_idx]->Value(i);
    for (int j{1}; j < n_cols; ++j) {
      auto val = columns[j]->Value(i);
      if (val > max_val) {
        max_idx = j;
        max_val = val;
      }
    }
    resp.emplace_back(max_idx);
  }

  return resp;
}

} // namespace argmax

template <utl::NumericType T>
std::vector<int64_t> ttb::AnalyticTableNumeric<T>::argmax(Axis axis) const {
  if (this->n_rows() == 0 || this->n_cols() == 0)
    return {};

  switch (axis) {
  case Axis::ROW:
    return argmax::argmax_row<T>(this->_arrow_tb);
  case Axis::COLUMN:
    return argmax::argmax_col<T>(this->_arrow_tb);
  default:
    throw ttb::DataTableNumericError("Invalid axis");
  }
}

template <utl::NumericType T>
utl::ReturnCode ttb::AnalyticTableNumeric<T>::one_hot_expand(int col_index) {
  auto res = ttb::AnalyticTable::one_hot_expand(col_index);
  if (res != utl::ReturnCode::Ok)
    return res;
  this->to_dtype();

  return res;
}

template class ttb::AnalyticTableNumeric<int>;
template class ttb::AnalyticTableNumeric<int64_t>;
template class ttb::AnalyticTableNumeric<float>;
template class ttb::AnalyticTableNumeric<double>;