#pragma once

#include <cuvs/neighbors/ivf_rabitq.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs_rabitq_ext::ivf_build_profiling {

struct timings {
  double setup_ms{};
  double training_sample_ms{};
  double balanced_kmeans_ms{};
  double assignment_ms{};
  double index_rotation_init_ms{};
  double layout_rotate_quantize_ms{};
  double total_ms{};
};

struct result {
  cuvs::neighbors::ivf_rabitq::index<int64_t> index;
  timings time;
};

/**
 * Device-resident IVF-RaBitQ build with synchronization at stage boundaries.
 *
 * This is an additive profiling wrapper around the same cuVS operations used by
 * ivf_rabitq::build. The low-level construct_on_gpu operation is intentionally
 * reported as one fused stage because list layout, dense rotation, residual
 * formation, and RaBitQ code/factor generation are not separate public APIs.
 */
result build_profiled(
  raft::resources const& handle,
  cuvs::neighbors::ivf_rabitq::index_params const& params,
  raft::device_matrix_view<const float, int64_t, raft::row_major> dataset);

}  // namespace cuvs_rabitq_ext::ivf_build_profiling
