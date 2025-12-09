// Standard Library (commonly used across project)
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Torch / ATen
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
#include <arrow/array/data.h>
#include <arrow/array/util.h>
#include <arrow/buffer.h>
#include <arrow/compute/api.h>
#include <arrow/io/api.h>

// Parquet I/O (used by Parquet_IO)
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>
#include <parquet/type_fwd.h>