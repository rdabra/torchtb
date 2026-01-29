// Standard Library (commonly used across project)
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Torch / ATen
#include <ATen/core/TensorBody.h>
#include <c10/core/TensorOptions.h>
#include <torch/data/dataloader.h>
#include <torch/data/dataloader_options.h>
#include <torch/data/datasets/tensor.h>
#include <torch/torch.h>
#include <torch/types.h>
// ATen ops actually used
#include <ATen/ops/from_blob.h>

// Arrow (umbrella + specific internals used)
#include <arrow/api.h>
#include <arrow/acero/exec_plan.h>
#include <arrow/acero/options.h>
#include <arrow/array/array_base.h>
#include <arrow/array/data.h>
#include <arrow/array/util.h>
#include <arrow/buffer.h>
#include <arrow/chunked_array.h>
#include <arrow/compute/api.h>
#include <arrow/compute/api_vector.h>
#include <arrow/compute/cast.h>
#include <arrow/csv/api.h>
#include <arrow/csv/options.h>
#include <arrow/csv/reader.h>
#include <arrow/csv/writer.h>
#include <arrow/io/api.h>
#include <arrow/io/file.h>
#include <arrow/io/type_fwd.h>
#include <arrow/pretty_print.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/type_traits.h>

// Parquet I/O (used by IO_FileParquet)
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/platform.h>
#include <parquet/properties.h>
#include <parquet/type_fwd.h>

// Tests
#include <gtest/gtest.h>
