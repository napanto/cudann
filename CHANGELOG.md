# Changelog

## [0.1.0] - 2026-09

First release: direct CUDA translation of syclnn 0.2.0 with the same public API,
file layout and numerics.

- `Options.blas = "tiled"`: hand-written BLAS (a 16x16 tiled GEMM with shared memory,
  a row-per-work-item GEMV, reductions for asum/nrm2) for the "same kernel in
  the three programming models" comparison (E7 of the study).

- `cudann/config.hpp`: identical to syclnn's (namespace apart); `Options.queue`
  gains a meaning for `graph`, `Options.streams` selects the number of streams.
- `cudann/activations.hpp`: `__host__ __device__` versions of the same functions.
- `cudann/device.hpp`: `cudaGetDeviceProperties` enumeration, `select_device`
  with the same spellings as syclnn (`gpu`, `cuda:N`, `index:N`, name substring).
- `cudann/blas.hpp`: cuBLAS handle wrapper (host pointer mode for alpha/beta,
  device pointer mode for asum/nrm2 so that the penalty never synchronises).
- `cudann/profile.hpp`: `cudaEvent` pairs around every launch, folded at the
  epoch synchronisation; suspended during graph capture.
- `cudann/network.hpp`: `cudaMalloc` / `cudaMallocManaged` / `cudaHostAlloc`
  tensors (`Options.memory`), pinned staging, one stream (`queue=in_order`),
  N streams with `cudaStreamWaitEvent` dependencies reproducing syclnn's
  fine-grained event graph (`out_of_order`, layer l on stream l mod N), or one
  captured CUDA Graph per batch replayed every epoch (`graph`; per-step
  scalars such as the learning-rate schedule and Adam's bias correction travel
  through a tiny H2D copy node so the replay stays exact). Kernels:
  `activate_k`, `output_delta_loss_k` (block reduction + one `atomicAdd(double)`
  per block; `loss_reduction=false` = one atomic per thread), `hidden_delta_k`,
  `bias_grad_k`, fused `update_k`, `penalty_k`, `gather_k`. Every syclnn
  ablation switch (`bias_gemv`, `direct_input`, `fine_deps`,
  `specialized_kernels`, `derivative_from_output`, `host_adam_correction`,
  `workgroup_size` = block size, `persistent_workspace`, `pinned_host`) is
  honoured; `join_kernels` has no CUDA counterpart.
- Build: CMake `LANGUAGES CXX CUDA`, `CUDANN_CUDA_ARCHS` (default `61;80`),
  `CUDANN_FAST_MATH` off by default (parity); `CUDANN_HIP=ON` converts the
  sources with `hipify-perl` at configure time and builds with hipcc + hipBLAS
  (used for the AMD portability experiment and for local validation on the
  RX 7900 XTX).
- Same pybind11 module layout as syclnn (`Network_double`, ... `Options`,
  `devices()`, `build_info()`), tests through `fnn-testkit`, C++ driver
  `bench/train_bench.cu`, ctest smoke test, CI compile job, container image on
  `ghcr.io/napanto/fnn-cuda`.
