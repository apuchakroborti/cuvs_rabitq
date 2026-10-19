/*
 * Additive RaBitQ decompression extension. This file is not part of upstream cuVS.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rabitq_decompression.hpp"

#include <raft/core/cublas_macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/device_uvector.hpp>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace cuvs::extensions::rabitq_decompression {
namespace {

constexpr int kBlockSize = 256;

template <typename StreamT>
cudaStream_t native_stream(StreamT stream)
{
  if constexpr (requires { stream.get(); }) {
    return stream.get();
  } else {
    return stream.value();
  }
}

__device__ __forceinline__ float flip_sign(float value, uint32_t bit)
{
  return __uint_as_float(__float_as_uint(value) ^ (bit << 31));
}

template <typename CodeT>
__global__ void decode_residuals_kernel(const CodeT* __restrict__ codes,
                                        const float* __restrict__ delta,
                                        const float* __restrict__ vl,
                                        float* __restrict__ residuals,
                                        int64_t count,
                                        int64_t padded_dim)
{
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    const int64_t row = idx / padded_dim;
    residuals[idx]    = static_cast<float>(codes[idx]) * delta[row] + vl[row];
  }
}

__global__ void add_row_kernel(float* __restrict__ rows,
                               const float* __restrict__ row,
                               int64_t count,
                               int64_t padded_dim)
{
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) { rows[idx] += row[idx % padded_dim]; }
}

__global__ void trim_rows_kernel(const float* __restrict__ padded,
                                 float* __restrict__ output,
                                 int64_t count,
                                 int64_t dim,
                                 int64_t padded_dim)
{
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    const int64_t row = idx / dim;
    const int64_t col = idx - row * dim;
    output[idx]       = padded[row * padded_dim + col];
  }
}

/** Reverse the four FHT-Kac rounds in reverse order, entirely on the GPU. */
__global__ void inverse_fht_kac_kernel(const float* input,
                                       float* output,
                                       const uint8_t* __restrict__ flip,
                                       int n_rows,
                                       int padded_dim,
                                       int trunc_dim,
                                       float fht_scale,
                                       float final_scale)
{
  const int row = blockIdx.x;
  if (row >= n_rows) return;

  extern __shared__ float values[];
  const int tid    = threadIdx.x;
  const float* src = input + static_cast<size_t>(row) * padded_dim;
  for (int i = tid; i < padded_dim; i += blockDim.x)
    values[i] = src[i];
  __syncthreads();

  const bool is_power_of_two = trunc_dim == padded_dim;
  const int half             = padded_dim / 2;
  const int words_per_round  = padded_dim / 32;
  const uint32_t* flip_words = reinterpret_cast<const uint32_t*>(flip);

  for (int round = 3; round >= 0; --round) {
    if (!is_power_of_two) {
      for (int i = tid; i < half; i += blockDim.x) {
        const float a    = values[i];
        const float b    = values[i + half];
        values[i]        = a + b;
        values[i + half] = a - b;
      }
      __syncthreads();
    }

    const int base = (!is_power_of_two && (round & 1)) ? padded_dim - trunc_dim : 0;
    for (int stride = 1; stride < trunc_dim; stride <<= 1) {
      const int pairs = trunc_dim / 2;
      for (int pair = tid; pair < pairs; pair += blockDim.x) {
        const int group = pair / stride;
        const int lo    = base + group * (2 * stride) + pair % stride;
        const int hi    = lo + stride;
        const float a   = values[lo];
        const float b   = values[hi];
        values[lo]      = a + b;
        values[hi]      = a - b;
      }
      __syncthreads();
    }

    if (!is_power_of_two) {
      for (int i = tid; i < trunc_dim; i += blockDim.x)
        values[base + i] *= fht_scale;
      __syncthreads();
    }

    for (int i = tid; i < padded_dim; i += blockDim.x) {
      const uint32_t word = flip_words[round * words_per_round + (i >> 5)];
      values[i]           = flip_sign(values[i], (word >> (i & 31)) & 1u);
    }
    __syncthreads();
  }

  float* dst = output + static_cast<size_t>(row) * padded_dim;
  for (int i = tid; i < padded_dim; i += blockDim.x)
    dst[i] = values[i] * final_scale;
}

__global__ void mse_kernel(const float* __restrict__ original,
                           const float* __restrict__ reconstructed,
                           size_t count,
                           double* sum)
{
  __shared__ double block_sum[kBlockSize];
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  double local        = 0.0;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    const double diff = static_cast<double>(original[i]) - reconstructed[i];
    local += diff * diff;
  }
  block_sum[threadIdx.x] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) block_sum[threadIdx.x] += block_sum[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(sum, block_sum[0]);
}

int64_t floor_power_of_two(int64_t value)
{
  int64_t result = 1;
  while (result <= value / 2)
    result <<= 1;
  return result;
}

void validate_peers(compression::quantizer const& quantizer,
                    compression::rotator const& rotator)
{
  RAFT_EXPECTS(quantizer.dim > 0 && quantizer.padded_dim >= quantizer.dim,
               "rabitq_decompression: invalid quantizer dimensions");
  RAFT_EXPECTS(rotator.padded_dim == quantizer.padded_dim,
               "rabitq_decompression: rotator and quantizer padded dimensions differ");
}

template <typename CodeT>
void decode_impl(raft::resources const& res,
                 compression::quantizer const& quantizer,
                 raft::device_matrix_view<const CodeT, int64_t, raft::row_major> codes,
                 raft::device_vector_view<const float, int64_t> delta,
                 raft::device_vector_view<const float, int64_t> vl,
                 raft::device_matrix_view<float, int64_t, raft::row_major> residuals)
{
  const int64_t n_rows = codes.extent(0);
  RAFT_EXPECTS(codes.extent(1) == quantizer.padded_dim,
               "rabitq_decompression: code width must equal padded_dim");
  RAFT_EXPECTS(delta.extent(0) == n_rows && vl.extent(0) == n_rows,
               "rabitq_decompression: delta and vl must hold one value per row");
  RAFT_EXPECTS(residuals.extent(0) == n_rows && residuals.extent(1) == quantizer.padded_dim,
               "rabitq_decompression: residual shape mismatch");
  if (n_rows == 0) return;
  const int64_t count = n_rows * quantizer.padded_dim;
  const int grid      = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
  // Next: decode_residuals_kernel()
  // (/home/achakroborti1/data-compressors/EmbdC/cuvs_rabitq/extensions/rabitq_decompression/src/rabitq_decompression.cu)
  // completes stage 1 by reconstructing each value as code * delta[row] + vl[row].
  decode_residuals_kernel<<<grid,
                            kBlockSize,
                            0,
                            native_stream(raft::resource::get_cuda_stream(res))>>>(
    codes.data_handle(),
    delta.data_handle(),
    vl.data_handle(),
    residuals.data_handle(),
    count,
    quantizer.padded_dim);
  RAFT_CUDA_TRY(cudaGetLastError());
}

void add_row(cudaStream_t stream,
             float* rows,
             const float* row,
             int64_t n_rows,
             int64_t padded_dim)
{
  const int64_t count = n_rows * padded_dim;
  if (count == 0) return;
  add_row_kernel<<<static_cast<int>((count + kBlockSize - 1) / kBlockSize),
                   kBlockSize,
                   0,
                   stream>>>(rows, row, count, padded_dim);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename CodeT>
void inverse_transform_impl(
  raft::resources const& res,
  compression::quantizer const& quantizer,
  compression::rotator const& rotator,
  raft::device_matrix_view<const CodeT, int64_t, raft::row_major> codes,
  raft::device_vector_view<const float, int64_t> delta,
  raft::device_vector_view<const float, int64_t> vl,
  std::optional<raft::device_vector_view<const float, int64_t>> centroid,
  raft::device_matrix_view<float, int64_t, raft::row_major> reconstructed)
{
  validate_peers(quantizer, rotator);
  const int64_t n_rows = codes.extent(0);
  RAFT_EXPECTS(reconstructed.extent(0) == n_rows &&
                 reconstructed.extent(1) == quantizer.dim,
               "rabitq_decompression: reconstructed shape mismatch");
  if (centroid.has_value()) {
    RAFT_EXPECTS(centroid->extent(0) == quantizer.dim,
                 "rabitq_decompression: centroid dimension mismatch");
  }

  auto stream      = raft::resource::get_cuda_stream(res);
  auto cuda_stream = native_stream(stream);
  auto workspace   = raft::resource::get_workspace_resource_ref(res);
  rmm::device_uvector<float> rotated(
    static_cast<size_t>(n_rows) * quantizer.padded_dim, stream, workspace);
  auto rotated_view = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
    rotated.data(), n_rows, quantizer.padded_dim);
  decode_impl(res, quantizer, codes, delta, vl, rotated_view);
  if (n_rows == 0) return;

  rmm::device_uvector<float> padded_centroid(
    centroid.has_value() ? static_cast<size_t>(quantizer.padded_dim) : 0, stream, workspace);
  rmm::device_uvector<float> rotated_centroid(
    centroid.has_value() ? static_cast<size_t>(quantizer.padded_dim) : 0, stream, workspace);
  if (centroid.has_value()) {
    RAFT_CUDA_TRY(cudaMemsetAsync(
      padded_centroid.data(), 0, quantizer.padded_dim * sizeof(float), cuda_stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(padded_centroid.data(),
                                  centroid->data_handle(),
                                  quantizer.dim * sizeof(float),
                                  cudaMemcpyDeviceToDevice,
                                  cuda_stream));
    auto padded_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      padded_centroid.data(), 1, quantizer.padded_dim);
    auto rotated_centroid_view = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
      rotated_centroid.data(), 1, quantizer.padded_dim);
    compression::rotate(res, rotator, padded_view, rotated_centroid_view);
    add_row(cuda_stream, rotated.data(), rotated_centroid.data(), n_rows, quantizer.padded_dim);
  }

  rmm::device_uvector<float> unrotated(
    static_cast<size_t>(n_rows) * quantizer.padded_dim, stream, workspace);
  auto rotated_const = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    rotated.data(), n_rows, quantizer.padded_dim);
  auto unrotated_view = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
    unrotated.data(), n_rows, quantizer.padded_dim);
  inverse_rotate(res, rotator, rotated_const, unrotated_view);
  trim_rows(cuda_stream,
            unrotated.data(),
            reconstructed.data_handle(),
            n_rows,
            quantizer.dim,
            quantizer.padded_dim);
}

}  // namespace

void inverse_transform_residuals(
  raft::resources const& res,
  compression::quantizer const& quantizer,
  raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> codes,
  raft::device_vector_view<const float, int64_t> delta,
  raft::device_vector_view<const float, int64_t> vl,
  raft::device_matrix_view<float, int64_t, raft::row_major> residuals)
{
  // Next: decode_impl()
  // (/home/achakroborti1/data-compressors/EmbdC/cuvs_rabitq/extensions/rabitq_decompression/src/rabitq_decompression.cu)
  // validates uint8 buffers and launches decompression stage 1.
  decode_impl(res, quantizer, codes, delta, vl, residuals);
}

void inverse_transform_residuals(
  raft::resources const& res,
  compression::quantizer const& quantizer,
  raft::device_matrix_view<const uint16_t, int64_t, raft::row_major> codes,
  raft::device_vector_view<const float, int64_t> delta,
  raft::device_vector_view<const float, int64_t> vl,
  raft::device_matrix_view<float, int64_t, raft::row_major> residuals)
{
  // Next: decode_impl()
  // (/home/achakroborti1/data-compressors/EmbdC/cuvs_rabitq/extensions/rabitq_decompression/src/rabitq_decompression.cu)
  // validates uint16 buffers and launches decompression stage 1.
  decode_impl(res, quantizer, codes, delta, vl, residuals);
}

void inverse_rotate(raft::resources const& res,
                    compression::rotator const& rotator,
                    raft::device_matrix_view<const float, int64_t, raft::row_major> input,
                    raft::device_matrix_view<float, int64_t, raft::row_major> output)
{
  const int64_t n_rows = input.extent(0);
  RAFT_EXPECTS(input.extent(1) == rotator.padded_dim && output.extent(0) == n_rows &&
                 output.extent(1) == rotator.padded_dim,
               "rabitq_decompression: inverse-rotation shape mismatch");
  if (n_rows == 0) return;

  if (rotator.kind == compression::rotator_kind::matmul) {
    RAFT_EXPECTS(input.data_handle() != output.data_handle(),
                 "rabitq_decompression: matrix inverse rotation cannot operate in place");
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    auto handle       = raft::resource::get_cublas_handle(res);
    RAFT_CUBLAS_TRY(
      cublasSetStream(handle, native_stream(raft::resource::get_cuda_stream(res))));
    // Next: cublasSgemm() (CUDA cuBLAS). Multiplication by the transposed dense
    // rotation matrix completes decompression stage 2 for the matmul rotator.
    RAFT_CUBLAS_TRY(cublasSgemm(handle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                static_cast<int>(rotator.padded_dim),
                                static_cast<int>(n_rows),
                                static_cast<int>(rotator.padded_dim),
                                &alpha,
                                rotator.rotation_matrix.data_handle(),
                                static_cast<int>(rotator.padded_dim),
                                input.data_handle(),
                                static_cast<int>(rotator.padded_dim),
                                &beta,
                                output.data_handle(),
                                static_cast<int>(rotator.padded_dim)));
    return;
  }

  const int64_t trunc_dim   = floor_power_of_two(rotator.padded_dim);
  const bool power_of_two   = trunc_dim == rotator.padded_dim;
  const float fht_scale     = 1.0f / std::sqrt(static_cast<float>(trunc_dim));
  const float final_scale =
    power_of_two ? fht_scale * fht_scale * fht_scale * fht_scale : 0.25f;
  const size_t shared_bytes = static_cast<size_t>(rotator.padded_dim) * sizeof(float);
  auto kernel               = &inverse_fht_kac_kernel;
  if (shared_bytes >= 48 * 1024) {
    RAFT_CUDA_TRY(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)));
  }
  // Next: inverse_fht_kac_kernel()
  // (/home/achakroborti1/data-compressors/EmbdC/cuvs_rabitq/extensions/rabitq_decompression/src/rabitq_decompression.cu)
  // reverses four FHT-Kac rounds and sign flips, completing stage 2.
  kernel<<<static_cast<int>(n_rows),
           kBlockSize,
           shared_bytes,
           native_stream(raft::resource::get_cuda_stream(res))>>>(
    input.data_handle(),
    output.data_handle(),
    rotator.flip_bits.data_handle(),
    static_cast<int>(n_rows),
    static_cast<int>(rotator.padded_dim),
    static_cast<int>(trunc_dim),
    fht_scale,
    final_scale);
  RAFT_CUDA_TRY(cudaGetLastError());
}

void trim_rows(cudaStream_t stream,
               const float* padded,
               float* output,
               int64_t n_rows,
               int64_t dim,
               int64_t padded_dim)
{
  const int64_t count = n_rows * dim;
  if (count == 0) return;
  // Next: trim_rows_kernel()
  // (/home/achakroborti1/data-compressors/EmbdC/cuvs_rabitq/extensions/rabitq_decompression/src/rabitq_decompression.cu)
  // copies original dimensions, removes rotation padding, and completes stage 3.
  trim_rows_kernel<<<static_cast<int>((count + kBlockSize - 1) / kBlockSize),
                     kBlockSize,
                     0,
                     stream>>>(padded, output, count, dim, padded_dim);
  RAFT_CUDA_TRY(cudaGetLastError());
}

void inverse_transform(
  raft::resources const& res,
  compression::quantizer const& quantizer,
  compression::rotator const& rotator,
  raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> codes,
  raft::device_vector_view<const float, int64_t> delta,
  raft::device_vector_view<const float, int64_t> vl,
  std::optional<raft::device_vector_view<const float, int64_t>> centroid,
  raft::device_matrix_view<float, int64_t, raft::row_major> reconstructed)
{
  inverse_transform_impl(
    res, quantizer, rotator, codes, delta, vl, centroid, reconstructed);
}

void inverse_transform(
  raft::resources const& res,
  compression::quantizer const& quantizer,
  compression::rotator const& rotator,
  raft::device_matrix_view<const uint16_t, int64_t, raft::row_major> codes,
  raft::device_vector_view<const float, int64_t> delta,
  raft::device_vector_view<const float, int64_t> vl,
  std::optional<raft::device_vector_view<const float, int64_t>> centroid,
  raft::device_matrix_view<float, int64_t, raft::row_major> reconstructed)
{
  inverse_transform_impl(
    res, quantizer, rotator, codes, delta, vl, centroid, reconstructed);
}

double mean_squared_error(
  raft::resources const& res,
  raft::device_matrix_view<const float, int64_t, raft::row_major> original,
  raft::device_matrix_view<const float, int64_t, raft::row_major> reconstructed)
{
  RAFT_EXPECTS(original.extent(0) == reconstructed.extent(0) &&
                 original.extent(1) == reconstructed.extent(1),
               "rabitq_decompression: MSE input shapes must match");
  const size_t count = static_cast<size_t>(original.extent(0)) * original.extent(1);
  RAFT_EXPECTS(count > 0, "rabitq_decompression: MSE input must not be empty");
  auto stream        = native_stream(raft::resource::get_cuda_stream(res));
  double* device_sum = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(&device_sum, sizeof(double), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(device_sum, 0, sizeof(double), stream));
  const int grid = static_cast<int>(std::min<size_t>((count + kBlockSize - 1) / kBlockSize, 4096));
  mse_kernel<<<grid, kBlockSize, 0, stream>>>(
    original.data_handle(), reconstructed.data_handle(), count, device_sum);
  RAFT_CUDA_TRY(cudaGetLastError());
  double host_sum = 0.0;
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(&host_sum, device_sum, sizeof(double), cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  RAFT_CUDA_TRY(cudaFreeAsync(device_sum, stream));
  return host_sum / static_cast<double>(count);
}

}  // namespace cuvs::extensions::rabitq_decompression
