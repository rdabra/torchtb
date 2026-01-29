#include "IO_File.h"

#include "Converter.h"

template <utl::NumericType T>
inline ttb::AnalyticTableNumeric<T> ttb::IO_File::read_numeric() const {
  auto table = this->read();

  return ttb::AnalyticTableNumeric<T>{std::move(table)};
}

template <utl::NumericType T>
void ttb::IO_File::write_tensor(torch::Tensor &&tensor) const {
  auto table = ttb::Converter::analytic_table<T>(std::move(tensor));

  this->write(table);
};

template <utl::NumericType T>
void ttb::IO_File::write_matrix(ttb::XYMatrix &&xy_matrix) const {
  auto my_xy_matrix = std::move(xy_matrix);

  auto X = my_xy_matrix.X().clone();
  auto Y = my_xy_matrix.Y().clone();
  auto XY = torch::cat({std::move(X), std::move(Y)}, 1);

  this->write_tensor<T>(std::move(XY));
}

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define INSTANTIATE_IO_FILE_TEMPLATES(T)                                                           \
  template ttb::AnalyticTableNumeric<T> ttb::IO_File::read_numeric<T>() const;                     \
  template void ttb::IO_File::write_tensor<T>(torch::Tensor && tensor) const;                      \
  template void ttb::IO_File::write_matrix<T>(ttb::XYMatrix &&) const;

INSTANTIATE_IO_FILE_TEMPLATES(int);
INSTANTIATE_IO_FILE_TEMPLATES(int64_t)
INSTANTIATE_IO_FILE_TEMPLATES(float)
INSTANTIATE_IO_FILE_TEMPLATES(double)

#undef INSTANTIATE_IO_FILE_TEMPLATES
