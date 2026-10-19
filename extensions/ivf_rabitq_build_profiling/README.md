# IVF-RaBitQ build profiling extension

This additive extension exposes synchronized stage timings for the device-resident cuVS
IVF-RaBitQ build used by the benchmark scripts. It does not modify the cuVS author sources.

Reported stages are setup, training-row sampling, balanced K-means, full-dataset assignment,
index/dense-rotation initialization, and IVF layout plus rotation/residual/quantization.

The final construction stage remains combined because `IVFGPU::construct_on_gpu()` fuses list
histogram/prefix/scatter work with sequential per-list dense rotation, residual formation, and
RaBitQ code/factor generation. Splitting those operations exactly would require changing that
author method. `build_other_ms` accounts for small allocations and host orchestration between the
synchronized stages, so the breakdown can be reconciled with `build_ms`.
