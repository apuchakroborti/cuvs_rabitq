/*
 * Additive RaBitQ decompression extension. This file is not part of upstream cuVS.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/preprocessing/quantize/rabitq.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace cuvs::extensions::rabitq_decompression {

namespace compression = cuvs::preprocessing::quantize::rabitq;

/** Decode reconstruction factors into centered, rotated, padded residuals. */
void inverse_transform_residuals(
  raft::resources const& res,
  compression::quantizer const& quantizer,
  raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> codes,
  raft::device_vector_view<const float, int64_t> delta,
  raft::device_vector_view<const float, int64_t> vl,
  raft::device_matrix_view<float, int64_t, raft::row_major> residuals);

/** @copydoc inverse_transform_residuals */
void inverse_transform_residuals(
  raft::resources const& res,
  compression::quantizer const& quantizer,
  raft::device_matrix_view<const uint16_t, int64_t, raft::row_major> codes,
  raft::device_vector_view<const float, int64_t> delta,
  raft::device_vector_view<const float, int64_t> vl,
  raft::device_matrix_view<float, int64_t, raft::row_major> residuals);

/** Apply the exact inverse of the configured RaBitQ random rotation. */
void inverse_rotate(raft::resources const& res,
                    compression::rotator const& rotator,
                    raft::device_matrix_view<const float, int64_t, raft::row_major> input,
                    raft::device_matrix_view<float, int64_t, raft::row_major> output);

/** Copy the leading unpadded dimensions from every reconstructed row. */
void trim_rows(cudaStream_t stream,
               const float* padded,
               float* output,
               int64_t n_rows,
               int64_t dim,
               int64_t padded_dim);

/**
 * Complete asynchronous reconstruction: decode, restore an optional centroid,
 * inverse-rotate, and remove padding.
 */
void inverse_transform(
  raft::resources const& res,
  compression::quantizer const& quantizer,
  compression::rotator const& rotator,
  raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> codes,
  raft::device_vector_view<const float, int64_t> delta,
  raft::device_vector_view<const float, int64_t> vl,
  std::optional<raft::device_vector_view<const float, int64_t>> centroid,
  raft::device_matrix_view<float, int64_t, raft::row_major> reconstructed);

/** @copydoc inverse_transform */
void inverse_transform(
  raft::resources const& res,
  compression::quantizer const& quantizer,
  compression::rotator const& rotator,
  raft::device_matrix_view<const uint16_t, int64_t, raft::row_major> codes,
  raft::device_vector_view<const float, int64_t> delta,
  raft::device_vector_view<const float, int64_t> vl,
  std::optional<raft::device_vector_view<const float, int64_t>> centroid,
  raft::device_matrix_view<float, int64_t, raft::row_major> reconstructed);

/** Element-wise GPU MSE; synchronizes the resource stream to return a host scalar. */
double mean_squared_error(
  raft::resources const& res,
  raft::device_matrix_view<const float, int64_t, raft::row_major> original,
  raft::device_matrix_view<const float, int64_t, raft::row_major> reconstructed);

}  // namespace cuvs::extensions::rabitq_decompression
