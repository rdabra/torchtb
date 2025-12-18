#include "AnalyticTable.h"
#include "detail/utils.h"

#include <algorithm>
#include <arrow/api.h>
#include <arrow/chunked_array.h>
#include <arrow/compute/api.h>
#include <arrow/compute/api_vector.h>
#include <arrow/compute/cast.h>
#include <arrow/pretty_print.h>
#include <arrow/table.h>
#include <arrow/type_fwd.h>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>

std::vector<std::string> ttb::AnalyticTable::col_names() const {
  return _arrow_tb->schema()->field_names();
  ;
}

std::vector<std::string> ttb::AnalyticTable::col_dtypes() const {
  std::vector<std::string> resp;
  for (auto &elem : _arrow_tb->schema()->fields())
    resp.emplace_back(elem->type()->ToString());

  return resp;
}

std::optional<int> ttb::AnalyticTable::col_index(std::string name) const {
  auto names = this->col_names();
  if (auto it = std::ranges::find(names, name); it != std::end(names))
    return std::distance(std::begin(names), it);
  else
    return std::nullopt;
}

int64_t ttb::AnalyticTable::n_rows() const {
  return _arrow_tb->num_rows();
}

int ttb::AnalyticTable::n_cols() const {
  return _arrow_tb->num_columns();
}

void ttb::AnalyticTable::remove_col(int col_index) {
  if (col_index < 0 || std::cmp_greater_equal(col_index, this->n_cols()))
    throw AnalyticTableError("col_index out of bounds");

  auto aux = _arrow_tb->RemoveColumn(col_index);
  if (!aux.ok())
    throw AnalyticTableError(aux.status().ToString());

  _arrow_tb = aux.MoveValueUnsafe();
}

void ttb::AnalyticTable::keep_cols(std::vector<int> indices) {
  auto aux = _arrow_tb->SelectColumns(indices);
  if (!aux.ok())
    throw AnalyticTableError(aux.status().ToString());

  _arrow_tb = aux.MoveValueUnsafe();
}

void ttb::AnalyticTable::bottom_append(const AnalyticTable &table) {
  if (this->n_cols() != table.n_cols())
    throw AnalyticTableError("Number of columns do not match");

  arrow::ConcatenateTablesOptions opts;

  auto resp = arrow::ConcatenateTables({_arrow_tb, table._arrow_tb});
  if (!resp.ok())
    throw AnalyticTableError(resp.status().ToString());

  this->_arrow_tb = resp.MoveValueUnsafe();
}

void ttb::AnalyticTable::right_append(const AnalyticTable &table) {
  if (this->n_rows() != table.n_rows())
    throw AnalyticTableError("Number of rows do not match");

  utl::shp<arrow::Table> resp = this->_arrow_tb;

  for (int i{0}; i < table.n_cols(); ++i) {
    auto field = table._arrow_tb->schema()->field(i);
    if (resp->schema()->GetFieldIndex(field->name()) != -1)
      field = field->WithName(field->name() + "_r");

    auto aux = resp->AddColumn(resp->num_columns(), field, table._arrow_tb->column(i));
    if (!aux.ok())
      throw AnalyticTableError(aux.status().ToString());

    resp = aux.MoveValueUnsafe();
  }

  this->_arrow_tb = resp;
}

void ttb::AnalyticTable::append(const AnalyticTable &table, const ttb::Axis &axis) {
  switch (axis) {
  case ttb::Axis::COLUMN:
    this->right_append(table);
    return;
  case ttb::Axis::ROW:
    this->bottom_append(table);
    return;
  default:
    return;
  }
}

void ttb::AnalyticTable::rename_cols(const std::vector<std::string> &names) {
  if (std::cmp_not_equal(names.size(), _arrow_tb->num_columns()))
    throw AnalyticTableError("Number of columns do not match");

  auto r_table = _arrow_tb->RenameColumns(names);
  if (!r_table.ok())
    throw AnalyticTableError(r_table.status().ToString());

  _arrow_tb = r_table.MoveValueUnsafe();
}

void ttb::AnalyticTable::slice(int64_t row_offset, int64_t row_length) {
  auto sliced = this->sliced(row_offset, row_length);

  _arrow_tb = std::move(sliced._arrow_tb);
}

void ttb::AnalyticTable::reorder_cols(const std::vector<int> &indices) {
  if (std::cmp_not_equal(indices.size(), this->n_cols()))
    throw AnalyticTableError("Invalid indices size");

  std::vector<int> seen;
  for (size_t i{0}; i < indices.size(); ++i) {
    if (indices[i] < 0 || indices[i] >= this->n_cols() ||
        std::ranges::find(seen, indices[i]) != std::end(seen))
      throw AnalyticTableError("Invalid indices");
    seen.emplace_back(indices[i]);
  }

  auto r = _arrow_tb->SelectColumns(indices);
  if (!r.ok())
    throw AnalyticTableError(r.status().ToString());

  _arrow_tb = r.MoveValueUnsafe();
}

void ttb::AnalyticTable::move_column(int from_index, int to_index) {
  auto n_cols = this->n_cols();
  if (from_index < 0 || to_index < 0 || to_index >= n_cols || from_index >= n_cols)
    throw AnalyticTableError("indices out of bounds");
  if (from_index == to_index)
    return;

  std::vector<int> indexes{std::from_range, std::views::iota(0, n_cols)};

  auto n_swaps = abs(to_index - from_index);
  int step = to_index > from_index ? 1 : -1;
  for (auto k{0}, i{from_index}; k < n_swaps; ++k) {
    int next = i + step;
    std::swap(indexes[i], indexes[next]);
    i = next;
  }

  this->reorder_cols(indexes);
}

void ttb::AnalyticTable::sort(int col_index, ttb::SortOrder mode) {
  if (col_index < 0 || col_index >= this->n_cols())
    throw AnalyticTableError("Index out of bounds");

  /// Required by arrow for some compute functions
  utl::initialize_arrow_compute();

  auto col_name = this->col_names()[col_index];
  auto order = mode == ttb::SortOrder::ASC ? arrow::compute::SortOrder::Ascending
                                           : arrow::compute::SortOrder::Descending;
  arrow::compute::SortOptions opts{{arrow::compute::SortKey{col_name, order}}};

  auto r_indices = arrow::compute::SortIndices(_arrow_tb, opts);
  if (!r_indices.ok())
    throw AnalyticTableError(r_indices.status().ToString());

  auto r_datum = arrow::compute::Take(_arrow_tb, r_indices.MoveValueUnsafe(),
                                      arrow::compute::TakeOptions::NoBoundsCheck());

  if (!r_datum.ok())
    throw AnalyticTableError(r_datum.status().ToString());

  _arrow_tb = r_datum.MoveValueUnsafe().table();
}

namespace one_hot_expand {

utl::shp<arrow::Array> to_array(const ttb::AnalyticTable &col_clone) {
  auto chunks = col_clone.arrow_table()->column(0)->chunks();

  auto r_col_as_array = arrow::Concatenate(chunks, arrow::default_memory_pool());
  if (!r_col_as_array.ok())
    throw ttb::AnalyticTableError(r_col_as_array.status().ToString());

  auto column = r_col_as_array.MoveValueUnsafe();

  return column;
}

utl::shp<arrow::Array> build_one_hot_col(const utl::shp<arrow::Array> &col_as_array,
                                         const utl::shp<arrow::Scalar> &distinct_value) {
  arrow::Int32Builder builder(arrow::default_memory_pool());
  auto n_rows = col_as_array->length();
  auto status = builder.Resize(n_rows);
  if (!status.ok())
    throw ttb::AnalyticTableError(status.ToString());

  for (int64_t i{0}; i < n_rows; ++i) {
    auto r_value = col_as_array->GetScalar(i);
    if (!r_value.ok())
      throw ttb::AnalyticTableError(r_value.status().ToString());

    auto value = r_value.MoveValueUnsafe();
    auto flag = value->Equals(*distinct_value) ? 1 : 0;
    builder.UnsafeAppend(flag);
  }

  auto r_array = builder.Finish();
  if (!r_array.ok())
    throw ttb::AnalyticTableError(r_array.status().ToString());

  return r_array.ValueUnsafe();
};

} // namespace one_hot_expand

void ttb::AnalyticTable::one_hot_expand(int col_index) {
  if (col_index < 0 || col_index >= this->n_cols())
    throw AnalyticTableError("Index out of bounds");

  auto col_clone = this->copy_cols({col_index});
  auto col_array = one_hot_expand::to_array(col_clone);
  auto field_names = arrow::compute::Unique(col_array).ValueOrDie();

  auto n_fields = field_names->length();
  auto prefix = this->col_names()[col_index] + "_";
  std::vector<utl::shp<arrow::Field>> fields;
  std::vector<utl::shp<arrow::Array>> one_hot_cols;
  fields.reserve(n_fields);
  one_hot_cols.reserve(n_fields);

  for (int64_t j{0}; j < n_fields; ++j) {
    auto r_field_name = field_names->GetScalar(j);
    if (!r_field_name.ok())
      throw AnalyticTableError(r_field_name.status().ToString());

    auto field_name = r_field_name.ValueUnsafe();
    fields.emplace_back(arrow::field(prefix + field_name->ToString(), arrow::int32()));

    auto one_hot_col = one_hot_expand::build_one_hot_col(col_array, field_name);

    one_hot_cols.emplace_back(std::move(one_hot_col));
  }

  auto schema = arrow::schema(fields);
  ttb::AnalyticTable table{arrow::Table::Make(schema, one_hot_cols, col_array->length())};
  this->append(table, ttb::Axis::COLUMN);
  this->remove_col(col_index);
}

ttb::AnalyticTable ttb::AnalyticTable::extract_column(int col_index) {
  auto ncols = this->n_cols();
  if (ncols == 1)
    throw AnalyticTableError("Table has one column");

  if (col_index < 0 || col_index >= ncols)
    throw AnalyticTableError("col_index out of bounds");

  this->move_column(col_index, ncols - 1);

  return this->right_extract_of(ncols - 2);
}

ttb::AnalyticTable ttb::AnalyticTable::sliced(int64_t row_offset, int64_t row_length) const {
  if (row_offset < 0 || row_length < 0 ||
      std::cmp_greater_equal(row_offset + row_length, this->n_rows()))
    throw AnalyticTableError("Invalid parameters");

  auto sliced_view = _arrow_tb->Slice(row_offset, row_length);
  auto r_sliced_cp = sliced_view->CombineChunks(arrow::default_memory_pool());
  if (!r_sliced_cp.ok())
    throw ttb::AnalyticTableError(r_sliced_cp.status().ToString());

  return AnalyticTable{r_sliced_cp.MoveValueUnsafe()};
}

void ttb::AnalyticTable::print_head(int64_t n_rows) const {
  auto head = _arrow_tb->Slice(0, std::min<int64_t>(this->n_rows(), n_rows));
  auto stat = arrow::PrettyPrint(*head, 2, &std::cout);
  if (!stat.ok())
    std::cout << stat;
}

void ttb::AnalyticTable::print_tail(int64_t n_rows) const {
  auto head = _arrow_tb->Slice(std::max<int64_t>(this->n_rows() - n_rows, 0), n_rows);
  auto stat = arrow::PrettyPrint(*head, 2, &std::cout);
  if (!stat.ok())
    std::cout << stat;
}

void ttb::AnalyticTable::reset() {
  _arrow_tb.reset();
}

ttb::AnalyticTable ttb::AnalyticTable::copy_cols(std::vector<int> indices) const {
  auto r_table = _arrow_tb->SelectColumns(indices);
  if (!r_table.ok())
    throw ttb::AnalyticTableError(r_table.status().ToString());

  return AnalyticTable{r_table.MoveValueUnsafe()};
}

ttb::AnalyticTable ttb::AnalyticTable::right_extract_of(int col_index) {
  if (col_index < 0 || col_index > this->n_cols() - 2)
    throw AnalyticTableError("col_index out of bounds");

  std::vector<int> indices;
  for (int j{col_index + 1}; j < this->n_cols(); ++j)
    indices.emplace_back(j);

  auto r_extracted = _arrow_tb->SelectColumns(indices);
  if (!r_extracted.ok())
    throw AnalyticTableError(r_extracted.status().ToString());

  auto last_index = col_index + 1;
  while (this->n_cols() > last_index) {
    auto aux = _arrow_tb->RemoveColumn(last_index);
    if (!aux.ok())
      throw AnalyticTableError(aux.status().ToString());

    _arrow_tb = aux.MoveValueUnsafe();
  }

  return AnalyticTable{r_extracted.MoveValueUnsafe()};
}

ttb::AnalyticTable ttb::AnalyticTable::clone() const {
  auto n_rows = this->n_rows();

  arrow::Int64Builder b;
  auto status = b.Resize(n_rows);
  if (!status.ok())
    throw AnalyticTableError(status.ToString());

  for (int64_t i{0}; i < n_rows; ++i)
    b.UnsafeAppend(i);
  auto r_indices = b.Finish();
  if (!r_indices.ok())
    throw AnalyticTableError(r_indices.status().ToString());

  auto r_datum = arrow::compute::Take(_arrow_tb, r_indices.ValueUnsafe());
  if (!r_datum.ok())
    throw AnalyticTableError(r_datum.status().ToString());

  auto table = r_datum.MoveValueUnsafe();

  auto aux = table.table();

  return AnalyticTable{std::move(aux)};
}
