#include "CSV_IO.h"
#include "Converter.h"
#include "Parquet_IO.h"
#include "detail/utils.h"
#include <chrono>
#include <filesystem>
#include <stdexcept>

int main(int argc, char *argv[]) {

  if (argc < 3)
    throw std::runtime_error("Input and output file paths must be informed.");

  std::filesystem::path in_path{argv[1]};
  std::filesystem::path out_path{argv[2]};

  if (utl::to_lower(in_path.extension()) != ".csv") {
    std::cout << utl::to_lower(in_path.extension()) << std::endl;
    throw std::runtime_error("Input file is not a csv file");
  }

  if (utl::to_lower(out_path.extension()) != ".parquet")
    throw std::runtime_error("Input file is not a csv file");

  auto time1 = std::chrono::system_clock::now();

  std::cout << "Reading input file..." << std::endl;
  auto in_file = ttb::CSV_IO{in_path, false};
  auto r_in_data = in_file.read();
  if (!r_in_data)
    throw std::runtime_error("Could not read input file");

  auto T = ttb::Converter::torch_tensor<float>(std::move(r_in_data.value()));

  std::cout << "Writing output file..." << std::endl;
  auto out_file = ttb::Parquet_IO{out_path};
  auto r_write = out_file.write<float>(std::move(T));
  if (r_write != utl::ReturnCode::Ok)
    throw std::runtime_error("Could not write output file");

  auto time2 = std::chrono::system_clock::now();

  auto duration = time2 - time1;

  std::cout << "CSV file sucessfully converted in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(duration);

  return 0;
}
