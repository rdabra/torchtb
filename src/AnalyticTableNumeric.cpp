#include "AnalyticTableNumeric.h"

#include <algorithm>

namespace to_dtype {

void cast_table(ttb::utl::shp<arrow::Table> &arrow_tb, ttb::utl::shp<arrow::DataType> type) {
  std::vector<ttb::utl::shp<arrow::ChunkedArray>> casted_columns;
  std::vector<ttb::utl::shp<arrow::Field>> casted_fields;
  for (int i{0}; i < arrow_tb->num_columns(); ++i) {
    auto column = arrow_tb->column(i);

    arrow::compute::CastOptions cast_options;
    cast_options.to_type = type;
    cast_options.allow_int_overflow = false;
    cast_options.allow_float_truncate = true;

    auto casted_datum = arrow::compute::Cast(arrow::Datum(column), cast_options);
    if (!casted_datum.ok())
      throw ttb::AnalyticTableNumericError{casted_datum.status().ToString()};

    casted_fields.emplace_back(arrow::field(arrow_tb->field(i)->name(), type));
    casted_columns.emplace_back(casted_datum.MoveValueUnsafe().chunked_array());
  }

  auto casted_table = arrow::Table::Make(arrow::schema(casted_fields), casted_columns);

  arrow_tb = casted_table;
}

} // namespace to_dtype

template <ttb::utl::NumericType T>
void ttb::AnalyticTableNumeric<T>::to_dtype() {
  to_dtype::cast_table(_arrow_tb, ttb::utl::arrow_dtype<T>());
}

template <ttb::utl::NumericType T>
ttb::AnalyticTableNumeric<T>::AnalyticTableNumeric(
    std::unordered_map<std::string, std::vector<T>> &&field_and_data)
    : _arrow_dtype(ttb::utl::arrow_dtype<T>()) {
  // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
  _arrow_tb = make_numeric_table(std::move(field_and_data));
}

namespace make_numeric_table {

template <ttb::utl::NumericType T>
std::pair<std::vector<ttb::utl::shp<arrow::Field>>, std::vector<ttb::utl::shp<arrow::Array>>>
make_fields_columns(const std::unordered_map<std::string, std::vector<T>> &field_col_data) {
  auto first_item = std::begin(field_col_data);
  auto n_rows{first_item->second.size()};

  auto n_fields = field_col_data.size();
  std::vector<ttb::utl::shp<arrow::Field>> fields;
  std::vector<ttb::utl::shp<arrow::Array>> columns;
  fields.reserve(n_fields);
  columns.reserve(n_fields);

  auto dtype = ttb::utl::arrow_dtype<T>();
  for (auto &[field, col_data] : field_col_data) {
    fields.emplace_back(arrow::field(field, dtype));

    auto builder = ttb::utl::new_unp<ttb::utl::ArrowBuilderType<T>>(arrow::default_memory_pool());
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

template <ttb::utl::NumericType T>
ttb::utl::shp<arrow::Table> ttb::AnalyticTableNumeric<T>::make_numeric_table(
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

template <ttb::utl::NumericType T>
std::vector<int64_t> argmax_row(const ttb::utl::shp<arrow::Table> &arrow_tb) {
  auto n_cols = arrow_tb->num_columns();

  std::vector<int64_t> resp;
  resp.reserve(n_cols);

  for (int j{0}; j < n_cols; ++j) {
    int64_t max_idx = 0;
    int64_t global_idx = 0;
    auto max_val = std::numeric_limits<T>::lowest();
    auto chunks = arrow_tb->column(j)->chunks();
    for (auto chunked_col : chunks) {
      auto array = std::static_pointer_cast<ttb::utl::ArrowArrayType<T>>(chunked_col);
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

template <ttb::utl::NumericType T>
std::vector<ttb::utl::shp<ttb::utl::ArrowArrayType<T>>>
fragment_columns(const ttb::utl::shp<arrow::Table> &arrow_tb) {
  std::vector<ttb::utl::shp<ttb::utl::ArrowArrayType<T>>> resp;
  auto n_cols = arrow_tb->num_columns();

  for (int64_t j{0}; j < n_cols; ++j) {
    auto chunks = arrow_tb->column(static_cast<int>(j))->chunks();
    auto maybe_arr = arrow::Concatenate(chunks, arrow::default_memory_pool());
    if (!maybe_arr.ok())
      throw ttb::AnalyticTableNumericError(maybe_arr.status().ToString());
    auto arr = std::static_pointer_cast<ttb::utl::ArrowArrayType<T>>(*maybe_arr);
    resp.emplace_back(std::move(arr));
  }
  return resp;
}

template <ttb::utl::NumericType T>
std::vector<int64_t> argmax_col(const ttb::utl::shp<arrow::Table> &arrow_tb) {
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

template <ttb::utl::NumericType T>
std::vector<int64_t> ttb::AnalyticTableNumeric<T>::argmax(Axis axis) const {
  if (this->n_rows() == 0 || this->n_cols() == 0)
    return {};

  switch (axis) {
  case Axis::ROW:
    return argmax::argmax_row<T>(this->_arrow_tb);
  case Axis::COLUMN:
    return argmax::argmax_col<T>(this->_arrow_tb);
  default:
    throw ttb::AnalyticTableNumericError("Invalid axis");
  }
}

template <ttb::utl::NumericType T>
void ttb::AnalyticTableNumeric<T>::one_hot_expand(int col_index) {
  ttb::AnalyticTable::one_hot_expand(col_index);
  this->to_dtype();
}

template <ttb::utl::NumericType T>
inline void ttb::AnalyticTableNumeric<T>::null_to_zero() {
  ttb::utl::initialize_arrow_compute();

  auto fields = _arrow_tb->schema()->fields();

  std::vector<ttb::utl::shp<arrow::ChunkedArray>> new_cols;
  new_cols.reserve(_arrow_tb->num_columns());

  for (int i{0}; i < _arrow_tb->num_columns(); ++i) {
    auto col = _arrow_tb->column(i);
    auto type = col->type();

    auto zero = arrow::MakeScalar<T>(0);

    std::vector<ttb::utl::shp<arrow::Array>> new_chunks;
    new_chunks.reserve(col->num_chunks());

    for (auto &chunk : col->chunks()) {
      auto r_is_null = arrow::compute::CallFunction("is_null", {chunk});
      if (!r_is_null.ok())
        throw ttb::AnalyticTableNumericError(r_is_null.status().ToString());

      auto r_new_chunk = arrow::compute::CallFunction(
          "if_else", {r_is_null.MoveValueUnsafe(), arrow::Datum{zero}, arrow::Datum{chunk}});
      if (!r_new_chunk.ok())
        throw ttb::AnalyticTableNumericError(r_new_chunk.status().ToString());
      new_chunks.emplace_back(r_new_chunk.MoveValueUnsafe().make_array());
    }

    new_cols.emplace_back(ttb::utl::new_shp<arrow::ChunkedArray>(std::move(new_chunks), type));
  }

  auto new_table = arrow::Table::Make(arrow::schema(std::move(fields)), std::move(new_cols),
                                      _arrow_tb->num_rows());

  _arrow_tb = new_table;
}

template class ttb::AnalyticTableNumeric<int>;
template class ttb::AnalyticTableNumeric<int64_t>;
template class ttb::AnalyticTableNumeric<float>;
template class ttb::AnalyticTableNumeric<double>;
