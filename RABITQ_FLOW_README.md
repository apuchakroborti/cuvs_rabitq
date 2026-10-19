The standalone RaBitQ path is:

`main()` → `metric_func_for_mode()` → `rabitq_metric()` → `_standalone_rabitq_run()` → `rabitq_compress_bench` → cuVS `rabitq::transform()` → custom decompression extension.

## Step-by-step flow

1. Python metric dispatch

File: [s2_model_abs_error_metrics.py](/home/achakroborti1/data-compressors/EmbdC/scripts/s2_model_abs_error_metrics.py:2481)

```text
metric_func_for_mode()
  └─ rabitq_metric()
```

For `--quant-error-modes rabitq`, `metric_func_for_mode()` creates a lambda that calls:

```python
rabitq_metric(model_name, bits, "rabitq", task)
```

2. Select standalone RaBitQ

File: [s2_model_abs_error_metrics.py](/home/achakroborti1/data-compressors/EmbdC/scripts/s2_model_abs_error_metrics.py:1979)

```text
rabitq_metric()
  └─ _standalone_rabitq_run()
```

This line distinguishes the modes:

```python
run = _standalone_rabitq_run if quant_error_mode == "rabitq" else _ivf_rabitq_run
```

`cr_rabitq`, MSE, and PSNR all reuse the same reconstructed result.

3. Prepare input and launch native binary

File: [s2_model_abs_error_metrics.py](/home/achakroborti1/data-compressors/EmbdC/scripts/s2_model_abs_error_metrics.py:1810)

Method:

```python
_standalone_rabitq_run()
```

Important internal calls:

```text
_qlrpcomp_original_embeddings()
_rabitq_write_raw()
_run_native_command()
_rabitq_load_fvecs()
_add_standalone_rabitq_timing_breakdowns()
```

It launches:

```text
cuvs_rabitq_scripts/build/rabitq_compress_bench
```

with `--bits-per-dim`, `--rotator`, `--fast`, `--batch-size`, `--warmup`, and `--iterations`.

Artifacts are written under:

```text
$EMBDC_PROJECT_DATA_DIR/s2_model_abs_error_metrics/rabitq/
  <model>/tensor_metrics/bits_<bits>/
    metrics.json
    rabitq.log
    reconstructed.pt
```

4. Native executable entry point

File: [rabitq_compress_bench.cu](/home/achakroborti1/data-compressors/EmbdC/cuvs_rabitq_scripts/rabitq_compress_bench.cu:495)

```text
main()
  ├─ parse_args()
  ├─ load_raw()
  └─ run<CodeT>()
```

Code type selection:

```text
bits 1–8 → run<uint8_t>()
bits 9   → run<uint16_t>()
```

The primary benchmark method is:

[rabitq_compress_bench.cu](/home/achakroborti1/data-compressors/EmbdC/cuvs_rabitq_scripts/rabitq_compress_bench.cu:246)

```cpp
template <typename CodeT>
int run(...)
```

5. Initialize the RaBitQ pipeline

Inside `run()`:

```cpp
rabitq::make_pipeline(res, params, data.dim);
```

Implementation: [rabitq.cu](/home/achakroborti1/data-compressors/EmbdC/cuvs_rabitq/cpp/src/preprocessing/quantize/rabitq.cu:352)

This builds:

```text
make_pipeline()
  ├─ make_rotator()
  └─ make_quantizer()
```

This is measured as `init_ms`, separately from compression throughput.

6. Compression path

Benchmark call: [rabitq_compress_bench.cu](/home/achakroborti1/data-compressors/EmbdC/cuvs_rabitq_scripts/rabitq_compress_bench.cu:286)

```text
transform_batch()
  └─ rabitq::transform()
```

The host-to-device copy occurs before the start event, so it is excluded.

Internal cuVS flow:

```text
rabitq::transform()
  └─ transform_impl()
      └─ detail::quantize_data()
          └─ quantize_data_impl()
              ├─ prepare_residuals()
              │   └─ pad_and_rotate()
              └─ quantize_fused_on_residuals()
                  └─ sa_quantize_fused_kernel<<<...>>>()
```

Relevant files:

- Public transform implementation: [rabitq.cu](/home/achakroborti1/data-compressors/EmbdC/cuvs_rabitq/cpp/src/preprocessing/quantize/rabitq.cu:125)
- Rotation/padding orchestration: [pipeline.cu](/home/achakroborti1/data-compressors/EmbdC/cuvs_rabitq/cpp/src/preprocessing/quantize/detail/rabitq/pipeline.cu:52)
- Fused quantization kernel: [quantize.cu](/home/achakroborti1/data-compressors/EmbdC/cuvs_rabitq/cpp/src/preprocessing/quantize/detail/rabitq/quantize.cu:60)

7. Decompression path

Benchmark entry: [rabitq_compress_bench.cu](/home/achakroborti1/data-compressors/EmbdC/cuvs_rabitq_scripts/rabitq_compress_bench.cu:330)

```text
decompress_batch()
  ├─ inverse_transform_residuals()
  ├─ inverse_rotate()
  └─ trim_rows()
```

Implementation: [rabitq_decompression.cu](/home/achakroborti1/data-compressors/EmbdC/cuvs_rabitq/extensions/rabitq_decompression/src/rabitq_decompression.cu:306)

Detailed flow:

```text
inverse_transform_residuals()
  └─ decode_impl()
      └─ decode_residuals_kernel<<<...>>>()

inverse_rotate()
  ├─ matrix-multiply inverse rotation, or
  └─ inverse FHT-KAC kernel

trim_rows()
  └─ trim_rows_kernel<<<...>>>()
```

Each of these three stages receives its own CUDA-event timing.

8. Reconstruction and metrics

After the timed benchmark loops, the binary performs another untimed pass for:

```text
transform
decompression
MSE calculation
device-to-host copy
save_fvecs()
```

Therefore, MSE computation, reconstructed D2H transfer, and file writing are excluded from `decompress_kernel_ms`.

9. Return to Python

The executable prints one CSV row. Python then:

```text
_run_native_command()
  ├─ parses CSV
  ├─ loads reconstructed `.fvecs`
  ├─ computes logical 16-bit throughput
  ├─ writes metrics.json
  └─ calculates MSE/PSNR from reconstructed tensors
```

## Timing boundary summary

Standalone RaBitQ compression timing includes:

- RaBitQ padding when required
- forward rotation
- residual preparation
- temporary stream-ordered workspace allocation inside `transform_impl()`
- fused quantization kernel
- CUDA stream work between the start and stop events

It excludes:

- loading embeddings
- raw-file creation
- host-to-device input transfer
- pipeline/rotator initialization
- warm-up iterations
- reconstruction
- device-to-host output transfer
- MSE/PSNR calculation
- saving files

Decompression timing includes only:

```text
decode kernel + inverse rotation + trim kernel
```

That means the RaBitQ throughput in `metrics.json` is effectively GPU codec-path timing, while using the logical original byte count selected by the Python script for the reported GB/s.