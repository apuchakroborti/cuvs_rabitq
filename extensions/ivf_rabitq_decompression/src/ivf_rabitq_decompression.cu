#include <ivf_rabitq_decompression.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/gemm.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cuvs::extensions::ivf_rabitq_decompression {
namespace {

template <typename StreamT> cudaStream_t native_stream(StreamT stream) {
  if constexpr (requires { stream.get(); }) {
    return stream.get();
  } else {
    return stream.value();
  }
}

void cuda_check(cudaError_t status, char const *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

template <typename T>
void read_exact(std::ifstream &input, T *output, std::size_t count,
                std::string const &field) {
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::runtime_error("IVF-RaBitQ index size overflow while reading " +
                             field);
  }
  auto bytes = count * sizeof(T);
  input.read(reinterpret_cast<char *>(output),
             static_cast<std::streamsize>(bytes));
  if (input.gcount() != static_cast<std::streamsize>(bytes)) {
    throw std::runtime_error("truncated IVF-RaBitQ index while reading " +
                             field);
  }
}

template <typename T>
std::vector<T> read_vector(std::ifstream &input, std::size_t count,
                           std::string const &field) {
  std::vector<T> values(count);
  read_exact(input, values.data(), count, field);
  return values;
}

template <typename T, typename DeviceVector>
void upload(DeviceVector &destination, std::vector<T> const &source,
            cudaStream_t stream, char const *operation) {
  if (source.empty())
    return;
  cuda_check(cudaMemcpyAsync(destination.data_handle(), source.data(),
                             source.size() * sizeof(T), cudaMemcpyHostToDevice,
                             stream),
             operation);
}

__device__ uint32_t unpack_extended_code(uint8_t const *row_codes,
                                         std::size_t coordinate,
                                         unsigned ex_bits,
                                         std::size_t row_bytes) {
  if (ex_bits == 8)
    return row_codes[coordinate];
  std::size_t bit_position = coordinate * ex_bits;
  std::size_t byte_index = bit_position / 8;
  unsigned bit_offset = static_cast<unsigned>(bit_position % 8);
  uint32_t window = static_cast<uint32_t>(row_codes[byte_index]) << 8;
  if (byte_index + 1 < row_bytes)
    window |= row_codes[byte_index + 1];
  unsigned shift = 16 - bit_offset - ex_bits;
  return (window >> shift) & ((1u << ex_bits) - 1u);
}

__global__ void decode_restore_centroid_kernel(
    uint32_t const *short_codes, float const *short_factors,
    uint8_t const *long_codes, float const *extended_factors,
    uint32_t const *cluster_ids, float const *centroids, std::size_t row_begin,
    std::size_t batch_rows, std::size_t padded_dim, unsigned ex_bits,
    float *rotated) {
  std::size_t local_element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t total = batch_rows * padded_dim;
  if (local_element >= total)
    return;

  std::size_t local_row = local_element / padded_dim;
  std::size_t coordinate = local_element - local_row * padded_dim;
  std::size_t index_row = row_begin + local_row;
  std::size_t words_per_row = padded_dim / 32;
  uint32_t word = short_codes[index_row * words_per_row + coordinate / 32];
  bool positive = ((word >> (31 - coordinate % 32)) & 1u) != 0;

  float signed_code;
  float scale;
  if (ex_bits == 0) {
    signed_code = positive ? 0.5f : -0.5f;
    scale = -0.5f * short_factors[index_row * 3 + 1];
  } else {
    std::size_t long_bytes = padded_dim * ex_bits / 8;
    auto stored = unpack_extended_code(long_codes + index_row * long_bytes,
                                       coordinate, ex_bits, long_bytes);
    uint32_t mask = (1u << ex_bits) - 1u;
    uint32_t magnitude = positive ? stored : mask - stored;
    signed_code = positive ? static_cast<float>(magnitude) + 0.5f
                           : -(static_cast<float>(magnitude) + 0.5f);
    scale = -0.5f * extended_factors[index_row * 2 + 1];
  }
  if (!isfinite(scale))
    scale = 0.0f;

  auto cluster = cluster_ids[index_row];
  rotated[local_element] =
      centroids[static_cast<std::size_t>(cluster) * padded_dim + coordinate] +
      scale * signed_code;
}

__global__ void trim_reorder_kernel(float const *padded, uint32_t const *ids,
                                    std::size_t row_begin,
                                    std::size_t batch_rows, std::size_t dim,
                                    std::size_t padded_dim,
                                    float *reconstructed) {
  std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t total = batch_rows * dim;
  if (element >= total)
    return;
  std::size_t local_row = element / dim;
  std::size_t coordinate = element - local_row * dim;
  std::size_t index_row = row_begin + local_row;
  reconstructed[static_cast<std::size_t>(ids[index_row]) * dim + coordinate] =
      padded[local_row * padded_dim + coordinate];
}

__global__ void squared_error_blocks_kernel(float const *lhs, float const *rhs,
                                            std::size_t count,
                                            double *block_sums) {
  extern __shared__ double partial[];
  double sum = 0.0;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    double difference = static_cast<double>(lhs[i]) - rhs[i];
    sum += difference * difference;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    block_sums[blockIdx.x] = partial[0];
}

} // namespace

struct compressed_index::impl {
  impl(raft::resources const &res, std::size_t n_rows, std::size_t dimension,
       std::size_t padded_dimension, std::size_t lists, std::size_t extra_bits)
      : rows(n_rows), dim(dimension), padded_dim(padded_dimension),
        n_lists(lists), ex_bits(extra_bits),
        rotation(raft::make_device_vector<float, int64_t>(res, padded_dim *
                                                                   padded_dim)),
        centroids(raft::make_device_vector<float, int64_t>(
            res, n_lists * padded_dim)),
        short_codes(raft::make_device_vector<uint32_t, int64_t>(
            res, rows * (padded_dim / 32))),
        short_factors(raft::make_device_vector<float, int64_t>(res, rows * 3)),
        long_codes(raft::make_device_vector<uint8_t, int64_t>(
            res, rows * padded_dim * ex_bits / 8)),
        extended_factors(
            raft::make_device_vector<float, int64_t>(res, rows * 2)),
        ids(raft::make_device_vector<uint32_t, int64_t>(res, rows)),
        cluster_ids(raft::make_device_vector<uint32_t, int64_t>(res, rows)) {}

  std::size_t rows;
  std::size_t dim;
  std::size_t padded_dim;
  std::size_t n_lists;
  std::size_t ex_bits;
  raft::device_vector<float, int64_t> rotation;
  raft::device_vector<float, int64_t> centroids;
  raft::device_vector<uint32_t, int64_t> short_codes;
  raft::device_vector<float, int64_t> short_factors;
  raft::device_vector<uint8_t, int64_t> long_codes;
  raft::device_vector<float, int64_t> extended_factors;
  raft::device_vector<uint32_t, int64_t> ids;
  raft::device_vector<uint32_t, int64_t> cluster_ids;
};

compressed_index::compressed_index(std::unique_ptr<impl> state)
    : state_(std::move(state)) {}
compressed_index::compressed_index(compressed_index &&) noexcept = default;
compressed_index &
compressed_index::operator=(compressed_index &&) noexcept = default;
compressed_index::~compressed_index() = default;

std::size_t compressed_index::rows() const noexcept { return state_->rows; }
std::size_t compressed_index::dim() const noexcept { return state_->dim; }
std::size_t compressed_index::padded_dim() const noexcept {
  return state_->padded_dim;
}
std::size_t compressed_index::n_lists() const noexcept {
  return state_->n_lists;
}
std::size_t compressed_index::bits_per_dim() const noexcept {
  return state_->ex_bits + 1;
}

compressed_index load(raft::resources const &res, std::string const &filename) {
  std::ifstream input(filename, std::ios::binary);
  if (!input)
    throw std::runtime_error("failed to open IVF-RaBitQ index: " + filename);

  std::size_t rows{}, dim{}, n_lists{}, ex_bits{};
  bool legacy_batch_flag{};
  float query_scaling_factors[2]{};
  read_exact(input, &rows, 1, "row count");
  read_exact(input, &dim, 1, "dimension");
  read_exact(input, &n_lists, 1, "list count");
  read_exact(input, &ex_bits, 1, "extra-bit count");
  read_exact(input, &legacy_batch_flag, 1, "legacy batch flag");
  read_exact(input, query_scaling_factors, 2, "query scaling factors");
  (void)legacy_batch_flag;
  (void)query_scaling_factors;

  if (rows == 0 || dim == 0 || n_lists == 0 || ex_bits > 8) {
    throw std::runtime_error("invalid IVF-RaBitQ index metadata");
  }
  std::size_t padded_dim = (dim + 63) / 64 * 64;
  auto cluster_sizes =
      read_vector<std::size_t>(input, n_lists, "cluster sizes");
  std::size_t row_sum = 0;
  for (auto size : cluster_sizes) {
    if (size > rows - row_sum)
      throw std::runtime_error("invalid IVF-RaBitQ cluster sizes");
    row_sum += size;
  }
  if (row_sum != rows)
    throw std::runtime_error("IVF-RaBitQ cluster sizes do not sum to rows");

  auto rotation =
      read_vector<float>(input, padded_dim * padded_dim, "rotation matrix");
  auto centroids =
      read_vector<float>(input, n_lists * padded_dim, "rotated centroids");
  auto short_codes =
      read_vector<uint32_t>(input, rows * (padded_dim / 32), "short codes");
  auto short_factors = read_vector<float>(input, rows * 3, "short factors");
  auto long_codes = read_vector<uint8_t>(input, rows * padded_dim * ex_bits / 8,
                                         "extended codes");
  auto extended_factors =
      read_vector<float>(input, rows * 2, "extended factors");
  auto ids = read_vector<uint32_t>(input, rows, "vector IDs");

  char trailing;
  if (input.read(&trailing, 1))
    throw std::runtime_error("unexpected data after IVF-RaBitQ index");

  std::vector<uint32_t> cluster_ids(rows);
  std::size_t offset = 0;
  for (std::size_t cluster = 0; cluster < n_lists; ++cluster) {
    std::fill_n(cluster_ids.begin() + offset, cluster_sizes[cluster],
                static_cast<uint32_t>(cluster));
    offset += cluster_sizes[cluster];
  }

  auto state = std::make_unique<compressed_index::impl>(
      res, rows, dim, padded_dim, n_lists, ex_bits);
  auto stream = raft::resource::get_cuda_stream(res);
  auto cuda_stream = native_stream(stream);
  upload(state->rotation, rotation, cuda_stream, "upload rotation matrix");
  upload(state->centroids, centroids, cuda_stream, "upload centroids");
  upload(state->short_codes, short_codes, cuda_stream, "upload short codes");
  upload(state->short_factors, short_factors, cuda_stream,
         "upload short factors");
  upload(state->long_codes, long_codes, cuda_stream, "upload extended codes");
  upload(state->extended_factors, extended_factors, cuda_stream,
         "upload extended factors");
  upload(state->ids, ids, cuda_stream, "upload vector IDs");
  upload(state->cluster_ids, cluster_ids, cuda_stream, "upload cluster IDs");
  cuda_check(cudaStreamSynchronize(cuda_stream),
             "IVF-RaBitQ index upload sync");
  return compressed_index(std::move(state));
}

stage_times decompress(
    raft::resources const &res, compressed_index const &index,
    std::size_t batch_rows, int warmup, int iterations,
    raft::device_matrix_view<float, int64_t, raft::row_major> reconstructed) {
  auto const &state = *index.state_;
  if (reconstructed.extent(0) != static_cast<int64_t>(state.rows) ||
      reconstructed.extent(1) != static_cast<int64_t>(state.dim)) {
    throw std::runtime_error(
        "reconstructed matrix shape does not match IVF-RaBitQ index");
  }
  if (warmup < 0 || iterations < 1)
    throw std::runtime_error("invalid timing iteration count");
  if (batch_rows == 0)
    batch_rows = state.rows;
  batch_rows = std::min(batch_rows, state.rows);

  auto rotated = raft::make_device_matrix<float, int64_t>(res, batch_rows,
                                                          state.padded_dim);
  auto unrotated = raft::make_device_matrix<float, int64_t>(res, batch_rows,
                                                            state.padded_dim);
  auto stream = raft::resource::get_cuda_stream(res);
  auto cuda_stream = native_stream(stream);
  cudaEvent_t begin{}, end{};
  cuda_check(cudaEventCreate(&begin), "create decompression start event");
  cuda_check(cudaEventCreate(&end), "create decompression stop event");

  auto time_stage = [&](auto &&operation, bool timed) {
    if (timed)
      cuda_check(cudaEventRecord(begin, cuda_stream), "record stage start");
    operation();
    if (!timed)
      return 0.0;
    cuda_check(cudaEventRecord(end, cuda_stream), "record stage stop");
    cuda_check(cudaEventSynchronize(end), "wait for decompression stage");
    float elapsed = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed, begin, end),
               "measure decompression stage");
    return static_cast<double>(elapsed);
  };

  stage_times totals;
  auto run_pass = [&](bool timed) {
    for (std::size_t offset = 0; offset < state.rows; offset += batch_rows) {
      std::size_t count = std::min(batch_rows, state.rows - offset);
      std::size_t decode_elements = count * state.padded_dim;
      totals.decode_restore_centroid_ms += time_stage(
          [&] {
            decode_restore_centroid_kernel<<<(decode_elements + 255) / 256, 256,
                                             0, cuda_stream>>>(
                state.short_codes.data_handle(),
                state.short_factors.data_handle(),
                state.long_codes.data_handle(),
                state.extended_factors.data_handle(),
                state.cluster_ids.data_handle(), state.centroids.data_handle(),
                offset, count, state.padded_dim,
                static_cast<unsigned>(state.ex_bits), rotated.data_handle());
            cuda_check(cudaPeekAtLastError(),
                       "launch IVF-RaBitQ decode kernel");
          },
          timed);

      totals.inverse_rotate_ms += time_stage(
          [&] {
            auto x =
                raft::make_device_matrix_view<float, int64_t, raft::row_major>(
                    rotated.data_handle(), count, state.padded_dim);
            // The serialized row-major P is viewed as column-major P^T,
            // yielding x * P^T.
            auto inverse =
                raft::make_device_matrix_view<float, int64_t, raft::col_major>(
                    const_cast<float *>(state.rotation.data_handle()),
                    state.padded_dim, state.padded_dim);
            auto z =
                raft::make_device_matrix_view<float, int64_t, raft::row_major>(
                    unrotated.data_handle(), count, state.padded_dim);
            raft::linalg::gemm(res, x, inverse, z);
          },
          timed);

      std::size_t trim_elements = count * state.dim;
      totals.trim_reorder_ms += time_stage(
          [&] {
            trim_reorder_kernel<<<(trim_elements + 255) / 256, 256, 0,
                                  cuda_stream>>>(
                unrotated.data_handle(), state.ids.data_handle(), offset, count,
                state.dim, state.padded_dim, reconstructed.data_handle());
            cuda_check(cudaPeekAtLastError(),
                       "launch IVF-RaBitQ trim/reorder kernel");
          },
          timed);
    }
    cuda_check(cudaStreamSynchronize(cuda_stream),
               "IVF-RaBitQ decompression sync");
  };

  for (int pass = 0; pass < warmup; ++pass)
    run_pass(false);
  for (int pass = 0; pass < iterations; ++pass)
    run_pass(true);

  cuda_check(cudaEventDestroy(begin), "destroy decompression start event");
  cuda_check(cudaEventDestroy(end), "destroy decompression stop event");
  totals.decode_restore_centroid_ms /= iterations;
  totals.inverse_rotate_ms /= iterations;
  totals.trim_reorder_ms /= iterations;
  totals.kernel_total_ms = totals.decode_restore_centroid_ms +
                           totals.inverse_rotate_ms + totals.trim_reorder_ms;
  return totals;
}

double mean_squared_error(
    raft::resources const &res,
    raft::device_matrix_view<const float, int64_t, raft::row_major> original,
    raft::device_matrix_view<const float, int64_t, raft::row_major>
        reconstructed) {
  if (original.extent(0) != reconstructed.extent(0) ||
      original.extent(1) != reconstructed.extent(1)) {
    throw std::runtime_error("MSE matrix shapes do not match");
  }
  std::size_t count =
      static_cast<std::size_t>(original.extent(0)) * original.extent(1);
  if (count == 0)
    throw std::runtime_error("MSE input is empty");
  constexpr unsigned threads = 256;
  unsigned blocks = static_cast<unsigned>(
      std::min<std::size_t>((count + threads - 1) / threads, 4096));
  auto block_sums = raft::make_device_vector<double, int64_t>(res, blocks);
  auto stream = raft::resource::get_cuda_stream(res);
  auto cuda_stream = native_stream(stream);
  squared_error_blocks_kernel<<<blocks, threads, threads * sizeof(double),
                                cuda_stream>>>(original.data_handle(),
                                               reconstructed.data_handle(),
                                               count, block_sums.data_handle());
  cuda_check(cudaPeekAtLastError(), "launch IVF-RaBitQ MSE kernel");
  std::vector<double> host_sums(blocks);
  cuda_check(cudaMemcpyAsync(host_sums.data(), block_sums.data_handle(),
                             blocks * sizeof(double), cudaMemcpyDeviceToHost,
                             cuda_stream),
             "copy IVF-RaBitQ MSE partials");
  cuda_check(cudaStreamSynchronize(cuda_stream), "IVF-RaBitQ MSE sync");
  double total = 0.0;
  for (auto value : host_sums)
    total += value;
  return total / static_cast<double>(count);
}

} // namespace cuvs::extensions::ivf_rabitq_decompression
