# cudann

CUDA feed-forward neural network library: the 1:1 port of
[syclnn](https://github.com/napanto/syclnn) to plain CUDA + cuBLAS, with the same
public API, the same numerics (validated by the shared
[fnn-testkit](https://github.com/napanto/fnn-bench) parity suite) and the same
`Options` switches, plus the CUDA-only experiments of the *SYCL vs CUDA vs
OpenMP* study (Accelerated Computing course, University of Pisa):

* **memory model**: `memory=device` (`cudaMalloc` + explicit copies),
  `shared` (`cudaMallocManaged`, page migration), `host` (`cudaHostAlloc`
  mapped zero-copy); pinned staging for the dataset (`pinned_host`);
* **synchronisation model**: `queue=in_order` (one stream), `out_of_order`
  (`streams=N` streams + `cudaStreamWaitEvent` dependencies reproducing
  syclnn's event graph), `graph` (one CUDA Graph captured per batch and
  replayed every epoch: the launch-overhead killer for tiny networks);
* per-phase profiler from `cudaEvent` pairs (`profile=True`, `net.profile`).

`network.hpp` is written so that a side-by-side diff with syclnn's shows exactly
what changes between SYCL and CUDA: USM → `cudaMalloc`, `queue.submit` →
`<<<grid, block, 0, stream>>>`, events → `cudaEvent`/`cudaStreamWaitEvent`,
oneMath → cuBLAS, `sycl::reduction` → block reduction + `atomicAdd`.

## Quick start

```python
import numpy as np, cudann

layers = [cudann.LayerDescription(784), cudann.LayerDescription(512, cudann.ActivationType.ReLU),
          cudann.LayerDescription(10, cudann.ActivationType.Sigmoid)]
net = cudann.Network(layers, 0.1, dtype="float", queue="graph", profile=True, seed=1,
                     momentum=cudann.MomentumConfig_float(cudann.Classical, 0.9))
losses = net.train(X.ravel(), Y.ravel(), n_samples=len(X), batch_size=256, max_epochs=5)
print(net.device_name, net.profile)
```

The API is that of syclnn 0.2 (`Network_double` / `Network_float`, the
`*_double` / `*_float` configuration classes, `Options`, `devices()`,
`build_info()`); see syclnn's README for the option table. `Options.device`
accepts `default`, `gpu`, `cuda:N`, `index:N` or a name substring. Only
`blas="auto"` (cuBLAS) exists.

## Run the published image

`ghcr.io/napanto/cudann` is the library installed in the `fnn-cuda` toolchain image (CUDA 12.9,
cuBLAS), built by CI from the `Containerfile` on every push, with device code for `sm_61` and
`sm_80`. One command on a host with an NVIDIA driver (>= 525) and the container toolkit's CDI spec:

```sh
podman run --rm -it --device nvidia.com/gpu=all ghcr.io/napanto/cudann python -c "import cudann; print(cudann.devices())"
```

The HIPified build for AMD GPUs is made from the `fnn-rocm` toolchain image (`fnn-bench/containers`)
and is not published as a library image.

## Building

```sh
# NVIDIA: nvcc 12.9 (last toolkit for Pascal), fat binary sm_61 + sm_80
export CUDANN_CUDA_ARCHS="61;80" CMAKE_BUILD_PARALLEL_LEVEL=8
pip install -v .

# AMD (portability experiment): hipify-perl at configure time + hipcc + hipBLAS
export CUDANN_HIP=ON CUDANN_HIP_ARCHS=gfx1100 CXX=hipcc
pip install -v .

# C++ driver + ctest smoke test
cmake -S . -B build -G Ninja && cmake --build build -j8 && ctest --test-dir build
./build/train_bench --layers 784,1024,10 --samples 8192 --batch 256 --epochs 5 --dtype float --profile
```

The `ghcr.io/napanto/fnn-cuda` image (CUDA 12.9, gcc-13 host compiler, Python
venv) has everything for the NVIDIA build; the `fnn-rocm` image of the study (`ghcr.io/napanto/fnn-rocm`, or the equivalent distrobox built by fnn-bench's `scripts/rocm-toolchain.sh`)
(ROCm 7.2.4) for the HIP build.

## Tests

```sh
pip install "fnn-testkit @ git+https://github.com/napanto/fnn-bench#subdirectory=testkit"
pytest                                       # --backend cudann, both dtypes, device 0
pytest --dtype float --option queue=graph --option memory=shared
```

Status (2026-09-03): the HIPified build passes the whole parity suite on the
RX 7900 XTX (ROCm 7.2.4, hipBLAS) in double and float for every switch
(`memory=device|shared|host`, `queue=in_order|out_of_order|graph`, `streams=1|8`,
all kernel ablations). On HIP, graph mode captures from a single stream
(multi-stream fork/join capture crashes in ROCm 7.2); the multi-stream graph is a
CUDA-only path validated on NVIDIA hardware. NVIDIA runs (GTX 1080 Ti, A30) are
tracked in `fnn-bench/docs/toolchains.md`.

## License

LGPL-3.0-only. Copyright (C) 2026 Antonio Napolitano.
