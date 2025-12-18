#include "CSV_IO.h"
#include "Parquet_IO.h"
#include "XYMatrix.h"
#include <chrono>
#include <iostream>

int main() {
  auto time1 = std::chrono::system_clock::now();

  std::unordered_map<std::string, std::vector<int>> data = {{"a", {1}}, {"b", {5}}, {"c", {3}}};

  auto aux = ttb::TbNumeric<int>::make_numeric_table(std::move(data));
  ttb::TbNumeric<int> tb{std::move(aux)};
  auto result = tb.argmax(ttb::Axis::COLUMN);

  std::cout << "tb.n_rows(): " << tb.n_rows() << std::endl;
  std::cout << "result[0]: " << result[0] << std::endl;

  ttb::Parquet_IO p{"/home/roberto/my_works/personal/torchtb/data/test_T.parquet"};
  p.write(tb);

  auto time2 = std::chrono::system_clock::now();

  auto duration = time2 - time1;

  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(duration) << std::endl;

  return 0;
}
