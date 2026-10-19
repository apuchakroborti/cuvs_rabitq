#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace cuvs::extensions::ivf_rabitq_decompression {

struct stage_times {
  double decode_restore_centroid_ms{};
  double inverse_rotate_ms{};
  double trim_reorder_ms{};
  double kernel_total_ms{};
};

/** Device-resident view of an author-serialized cuVS IVF-RaBitQ index. */
class compressed_index {
public:
  compressed_index(compressed_index const &) = delete;
  compressed_index &operator=(compressed_index const &) = delete;
  compressed_index(compressed_index &&) noexcept;
  compressed_index &operator=(compressed_index &&) noexcept;
  ~compressed_index();

  [[nodiscard]] std::size_t rows() const noexcept;
  [[nodiscard]] std::size_t dim() const noexcept;
  [[nodiscard]] std::size_t padded_dim() const noexcept;
  [[nodiscard]] std::size_t n_lists() const noexcept;
  [[nodiscard]] std::size_t bits_per_dim() const noexcept;

private:
  struct impl;
  explicit compressed_index(std::unique_ptr<impl> state);
  std::unique_ptr<impl> state_;

  friend compressed_index load(raft::resources const &, std::string const &);
  friend stage_times
  decompress(raft::resources const &, compressed_index const &, std::size_t,
             int, int,
             raft::device_matrix_view<float, int64_t, raft::row_major>);
};

/** Parse and upload an index produced by
 * cuvs::neighbors::ivf_rabitq::serialize. */
compressed_index load(raft::resources const &res, std::string const &filename);

/**
 * Reconstruct all vectors in original row order.
 *
 * Allocations and compressed-index H2D upload are outside the returned timings.
 * Timings cover only GPU decode/centroid restoration, inverse rotation, and
 * trim/reorder work.
 */
stage_times decompress(
    raft::resources const &res, compressed_index const &index,
    std::size_t batch_rows, int warmup, int iterations,
    raft::device_matrix_view<float, int64_t, raft::row_major> reconstructed);

/** GPU MSE between original and reconstructed row-major float matrices. */
double mean_squared_error(
    raft::resources const &res,
    raft::device_matrix_view<const float, int64_t, raft::row_major> original,
    raft::device_matrix_view<const float, int64_t, raft::row_major>
        reconstructed);

} // namespace cuvs::extensions::ivf_rabitq_decompression
