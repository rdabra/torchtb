#include "Converter.h"
#include "IO_FileCSV.h"
#include "IO_FileParquet.h"
#include "detail/ttbutils.h"
#include <chrono>
#include <filesystem>
#include <stdexcept>

int main(int argc, char *argv[]) {

  if (argc < 3)
    throw std::runtime_error("Input and output file paths must be informed.");

  std::filesystem::path in_path{argv[1]};
  std::filesystem::path out_path{argv[2]};

  if (ttb::utl::to_lower(in_path.extension()) != ".csv") {
    std::cout << ttb::utl::to_lower(in_path.extension()) << std::endl;
    throw std::runtime_error("Input file is not a csv file");
  }

  if (ttb::utl::to_lower(out_path.extension()) != ".parquet")
    throw std::runtime_error("Input file is not a csv file");

  auto time1 = std::chrono::system_clock::now();

  std::cout << "Reading input file..." << std::endl;
  auto in_file = ttb::IO_FileCSV{in_path, false};
  auto r_in_data = in_file.read();

  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  auto T = ttb::Converter::torch_tensor<float>(std::move(r_in_data), device);

  std::cout << "Writing output file..." << std::endl;
  auto out_file = ttb::IO_FileParquet{out_path};
  out_file.write_tensor<float>(std::move(T));

  auto time2 = std::chrono::system_clock::now();

  auto duration = time2 - time1;

  std::cout << "CSV file sucessfully converted in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(duration);

  return 0;
}
