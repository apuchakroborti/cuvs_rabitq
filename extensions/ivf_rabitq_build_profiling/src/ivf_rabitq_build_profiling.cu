#include "ivf_rabitq_build_profiling.hpp"

#include "ivf_rabitq/gpu_index/ivf_gpu.cuh"

#include <cuvs/cluster/kmeans.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/sample_rows.cuh>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <chrono>
#include <optional>
#include <utility>

namespace cuvs_rabitq_ext::ivf_build_profiling {
namespace {

// Use a monotonic CPU clock to measure elapsed wall-clock time.
using clock_type = std::chrono::steady_clock;

// Measure a stage, including its host-side work and stream synchronization.
template <typename Function>
double time_stage(raft::resources const& handle, Function&& function)
{
  // Step 1: Finish preceding work on the handle's stream before starting the timer.
  raft::resource::sync_stream(handle);

  // Step 2: Start timing and execute the supplied stage.
  auto begin = clock_type::now();
  function();

  // Step 3: Wait for the handle's stream to finish before calculating elapsed time.
  raft::resource::sync_stream(handle);
  return std::chrono::duration<double, std::milli>(clock_type::now() - begin).count();
}

}  // namespace

// Build an IVF-RaBitQ index from a device-resident dataset and record stage timings.
result build_profiled(
  raft::resources const& handle,
  cuvs::neighbors::ivf_rabitq::index_params const& params,
  raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
{
  using cuvs::neighbors::ivf_rabitq::index;

  // Step 1: Read the dataset shape and validate the minimum build requirements.
  auto const n_rows = dataset.extent(0);
  auto const dim    = dataset.extent(1);
  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");
  RAFT_EXPECTS(n_rows >= params.n_lists, "number of rows can't be less than n_lists");

  // Step 2: Initialize timing results, finish prior stream work, and start the
  // total build timer. The input dataset is already on the GPU.
  timings measured{};
  raft::resource::sync_stream(handle);
  auto total_begin = clock_type::now();

  // Step 3: Obtain the CUDA stream and allocate device storage for cluster centers
  // (n_lists * dim floats) and per-row cluster assignments (n_rows labels).
  auto stream = raft::resource::get_cuda_stream(handle);
  rmm::device_uvector<float> centers(0, stream);
  rmm::device_uvector<uint32_t> labels(0, stream);
  measured.setup_ms = time_stage(handle, [&] {
    centers.resize(static_cast<size_t>(params.n_lists) * dim, stream);
    labels.resize(n_rows, stream);
  });

  // Step 4: Limit the training sample size by the available rows and the configured
  // maximum training points per cluster multiplied by the number of clusters.
  size_t const n_rows_train =
    std::min(static_cast<size_t>(n_rows),
             static_cast<size_t>(params.max_train_points_per_cluster) * params.n_lists);

  // Step 5: Allocate the training matrix and sample dataset rows using seed 137.
  // Allocation is included in total_ms, but occurs outside training_sample_ms.
  auto trainset = raft::make_device_matrix<float, int64_t>(handle, n_rows_train, dim);
  measured.training_sample_ms = time_stage(handle, [&] {
    raft::random::RngState random_state{137};
    raft::matrix::sample_rows<float, int64_t>(handle, random_state, dataset, trainset.view());
  });

  // Step 6: Configure balanced k-means with the requested iteration count and
  // expanded squared Euclidean distance, then train centers on the sampled rows.
  cuvs::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.n_iters = params.kmeans_n_iters;
  kmeans_params.metric  = cuvs::distance::DistanceType::L2Expanded;
  auto centers_view = raft::make_device_matrix_view<float, int64_t>(
    centers.data(), params.n_lists, dim);
  measured.balanced_kmeans_ms = time_stage(handle, [&] {
    cuvs::cluster::kmeans::fit(
      handle, kmeans_params, raft::make_const_mdspan(trainset.view()), centers_view);
  });

  // Step 7: Assign every dataset row to a cluster using the trained centers.
  // These views reference the existing device buffers without copying their data.
  auto labels_view = raft::make_device_vector_view<uint32_t, int64_t>(labels.data(), n_rows);
  auto centers_const_view = raft::make_device_matrix_view<const float, int64_t>(
    centers.data(), params.n_lists, dim);
  measured.assignment_ms = time_stage(handle, [&] {
    cuvs::cluster::kmeans::predict(handle, kmeans_params, dataset, centers_const_view, labels_view);
  });

  // Step 8: Construct the index inside the timed stage. The optional holds the
  // resulting index so the following stage can populate it.
  std::optional<index<int64_t>> built_index;
  measured.index_rotation_init_ms = time_stage(handle, [&] {
    // The IVFGPU/index constructor allocates quantizer state and generates the
    // dense padded rotation matrix used by the compression path.
    built_index.emplace(handle, n_rows, dim, params.n_lists, params.bits_per_dim);
  });

  // Step 9: Delegate GPU index construction to construct_on_gpu, supplying the
  // original vectors, trained centers, cluster labels, and fast-quantization flag.
  // This stage groups the layout, rotation, and quantization work into one timing.
  measured.layout_rotate_quantize_ms = time_stage(handle, [&] {
    built_index->rabitq_index().construct_on_gpu(
      dataset.data_handle(), centers.data(), labels.data(), params.fast_quantize_flag);
  });

  // Step 10: Record total elapsed build time, including work between timed stages.
  // The final time_stage call has already synchronized the handle's stream.
  // This measurement ends before returning the index and destroying local buffers.
  measured.total_ms =
    std::chrono::duration<double, std::milli>(clock_type::now() - total_begin).count();

  // Step 11: Move the completed index into the result alongside its timing values.
  return result{std::move(*built_index), measured};
}

}  // namespace cuvs_rabitq_ext::ivf_build_profiling