// SPDX-License-Identifier: LGPL-3.0-only
// cudann - CUDA feed-forward neural network library (1:1 port of syclnn 0.2)
// Copyright (C) 2026 Antonio Napolitano
//
// network.hpp: Network<T>, the same multi-layer perceptron as syclnn, written
// directly in CUDA: cudaMalloc/cudaMallocManaged/cudaHostAlloc tensors
// (column-major, one sample per column), cuBLAS for GEMM/GEMV/asum/nrm2, hand
// written __global__ kernels for everything else.  Work is ordered with streams
// and events: one stream (Options::queue = InOrder), several streams with
// cudaStreamWaitEvent dependencies that reproduce syclnn's event graph
// (OutOfOrder), or one captured CUDA Graph per batch replayed every epoch
// (Graph).  The host synchronises once per epoch to read the loss.
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <exception>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cudann/activations.hpp"
#include "cudann/blas.hpp"
#include "cudann/config.hpp"
#include "cudann/device.hpp"
#include "cudann/profile.hpp"

namespace cudann {

// ============================================================================
//                                   kernels
// ============================================================================
namespace kernels {

template <ActivationType A> struct ActF {
    template <typename T> __device__ T operator()(T z) const { return Act<A>::template f<T>(z); }
};
struct ActRT {
    ActivationType a;
    template <typename T> __device__ T operator()(T z) const { return activate(a, z); }
};
template <ActivationType A, bool FromOut> struct DfF {
    template <typename T> __device__ T operator()(T o, T z) const {
        return FromOut ? Act<A>::template df_out<T>(o, z) : Act<A>::template df<T>(z);
    }
};
struct DfRT {
    ActivationType a;
    bool from_out;
    template <typename T> __device__ T operator()(T o, T z) const {
        return from_out ? derivative_from_output(a, o, z) : derivative(a, z);
    }
};

/// net[i] += bias[i % M]; out[i] = f(net[i])
template <typename T, typename F>
__global__ void activate_k(T *net, T *out, const T *__restrict__ bias, std::size_t n, std::size_t M, F f) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const T z = net[i] + bias[i % M];
        net[i] = z;
        out[i] = f(z);
    }
}

/// delta = (t - o) f'(net); loss += 0.5 (t - o)^2  (block reduction + one atomicAdd per block)
template <typename T, typename D>
__global__ void output_delta_loss_k(const T *__restrict__ targets, const T *__restrict__ out, const T *__restrict__ net,
                                    T *delta, std::size_t n, D df, double *acc) {
    __shared__ double sdata[1024];
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    double v = 0.0;
    if (i < n) {
        const T o = out[i];
        const T e = targets[i] - o;
        delta[i] = e * df(o, net[i]);
        const double ed = static_cast<double>(e);
        v = 0.5 * ed * ed;
    }
    sdata[threadIdx.x] = v;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(acc, sdata[0]);
}

/// 0.1-style: every thread does an atomicAdd on one scalar of type T
template <typename T, typename D>
__global__ void output_delta_loss_atomic_k(const T *__restrict__ targets, const T *__restrict__ out,
                                           const T *__restrict__ net, T *delta, std::size_t n, D df, T *acc) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const T o = out[i];
        const T e = targets[i] - o;
        delta[i] = e * df(o, net[i]);
        const double ed = static_cast<double>(e);
        atomicAdd(acc, static_cast<T>(0.5 * ed * ed));
    }
}

/// delta[i] *= f'(net[i])
template <typename T, typename D>
__global__ void hidden_delta_k(T *delta, const T *__restrict__ out, const T *__restrict__ net, std::size_t n, D df) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        delta[i] *= df(out[i], net[i]);
}

/// gb[j] = sum_n delta[j + M n] / B  (0.1: one thread per output neuron)
template <typename T> __global__ void bias_grad_k(const T *__restrict__ delta, T *gb, std::size_t M, std::size_t B) {
    const std::size_t j = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < M) {
        T s = T(0);
        for (std::size_t n = 0; n < B; ++n)
            s += delta[j + M * n];
        gb[j] = s / static_cast<T>(B);
    }
}

/// Per-step scalars (updated per batch through a small H2D copy, so that a
/// captured graph can be replayed with a different learning rate / Adam step).
template <typename T> struct StepScalars {
    T lr_eff, c1, c2, t;
};

template <typename T> struct UpdateParams {
    int strategy; // AdaptiveLearningRate<T>::Strategy
    bool momentum, l1, l2, host_corr;
    T mu, lr, eps, beta1, beta2, lambda1, lambda2;
};

template <typename T>
__device__ inline void step_fn(T &p, T grad, T reg_grad, T &m, T &v, const UpdateParams<T> &u, const StepScalars<T> &s) {
    using Strategy = typename AdaptiveLearningRate<T>::Strategy;
    const T g = grad - reg_grad; // ascent direction of (t - o)
    T step;
    switch (static_cast<Strategy>(u.strategy)) {
    case Strategy::Constant:
    case Strategy::LinearDecay:
        if (u.momentum) {
            m = u.mu * m + s.lr_eff * g;
            step = m;
        } else {
            step = s.lr_eff * g;
        }
        break;
    case Strategy::AdaGrad: {
        const T G = -g;
        v += G * G;
        step = (u.lr / (dev_sqrt(v) + u.eps)) * g;
    } break;
    case Strategy::RMSProp: {
        const T G = -g;
        v = u.beta1 * v + (T(1) - u.beta1) * G * G;
        step = (u.lr / (dev_sqrt(v) + u.eps)) * g;
    } break;
    case Strategy::Adam:
    default: {
        const T G = -g;
        m = u.beta1 * m + (T(1) - u.beta1) * G;
        v = u.beta2 * v + (T(1) - u.beta2) * G * G;
        T m_hat, v_hat;
        if (u.host_corr) {
            m_hat = m * s.c1;
            v_hat = v * s.c2;
        } else {
            m_hat = m / (T(1) - dev_pow(u.beta1, s.t));
            v_hat = v / (T(1) - dev_pow(u.beta2, s.t));
        }
        step = -(u.lr * m_hat / (dev_sqrt(v_hat) + u.eps));
    } break;
    }
    p += step;
}

/// One fused launch over the weights and biases of one layer.
template <typename T>
__global__ void update_k(T *W, const T *__restrict__ gW, T *mW, T *vW, std::size_t nw, T *b, const T *__restrict__ gb,
                         T *mb, T *vb, std::size_t nb, UpdateParams<T> u, const StepScalars<T> *__restrict__ sp) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nw + nb)
        return;
    const StepScalars<T> s = *sp;
    if (i < nw) {
        const T w = W[i];
        T r = T(0);
        if (u.l1)
            r += u.lambda1 * sgn(w);
        if (u.l2)
            r += u.lambda2 * w;
        step_fn(W[i], gW[i], r, mW[i], vW[i], u, s);
    } else {
        const std::size_t j = i - nw;
        step_fn(b[j], gb[j], T(0), mb[j], vb[j], u, s);
    }
}

/// acc[1] = sum_l lambda1 asum_l + 0.5 lambda2 nrm2_l^2
template <typename T>
__global__ void penalty_k(const T *__restrict__ tmp, std::size_t L, bool l1, bool l2, T lambda1, T lambda2, double *acc) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        double p = 0.0;
        for (std::size_t l = 0; l < L; ++l) {
            if (l1)
                p += static_cast<double>(lambda1) * static_cast<double>(tmp[2 * l]);
            if (l2) {
                const double nrm = static_cast<double>(tmp[2 * l + 1]);
                p += 0.5 * static_cast<double>(lambda2) * nrm * nrm;
            }
        }
        acc[1] = p;
    }
}

template <typename T>
__global__ void gather_k(const T *__restrict__ src, T *dst, const std::uint32_t *__restrict__ perm, std::size_t rows,
                         std::size_t total) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total) {
        const std::size_t c = i / rows, r = i - c * rows;
        dst[i] = src[r + rows * perm[c]];
    }
}

template <typename T> __global__ void fill_k(T *p, T v, std::size_t n) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        p[i] = v;
}

} // namespace kernels

// ============================================================================
//                                   helpers
// ============================================================================
namespace detail {

template <typename T> inline T *dev_alloc(std::size_t n, MemoryKind kind, T **mapped = nullptr) {
    if (n == 0)
        return nullptr;
    void *p = nullptr;
    switch (kind) {
    case MemoryKind::Device: check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc"); break;
    case MemoryKind::Shared: check(cudaMallocManaged(&p, n * sizeof(T)), "cudaMallocManaged"); break;
    case MemoryKind::Host:
        check(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped), "cudaHostAlloc");
        if (mapped)
            check(cudaHostGetDevicePointer(reinterpret_cast<void **>(mapped), p, 0), "cudaHostGetDevicePointer");
        break;
    }
    return static_cast<T *>(p);
}

/// RAII device / managed / mapped-host allocation (move-only).
template <typename T> class DevBuffer {
  public:
    DevBuffer() = default;
    DevBuffer(std::size_t n, MemoryKind kind) : m_n(n), m_kind(kind) {
        m_ptr = dev_alloc<T>(n, kind, &m_dev);
        if (kind != MemoryKind::Host)
            m_dev = m_ptr;
    }
    DevBuffer(const DevBuffer &) = delete;
    DevBuffer &operator=(const DevBuffer &) = delete;
    DevBuffer(DevBuffer &&o) noexcept : m_n(o.m_n), m_kind(o.m_kind), m_ptr(o.m_ptr), m_dev(o.m_dev) {
        o.m_ptr = o.m_dev = nullptr;
        o.m_n = 0;
    }
    DevBuffer &operator=(DevBuffer &&o) noexcept {
        if (this != &o) {
            release();
            m_n = o.m_n;
            m_kind = o.m_kind;
            m_ptr = o.m_ptr;
            m_dev = o.m_dev;
            o.m_ptr = o.m_dev = nullptr;
            o.m_n = 0;
        }
        return *this;
    }
    ~DevBuffer() { release(); }
    void release() noexcept {
        if (m_ptr) {
            if (m_kind == MemoryKind::Host)
                (void)cudaFreeHost(m_ptr);
            else
                (void)cudaFree(m_ptr);
            m_ptr = m_dev = nullptr;
            m_n = 0;
        }
    }
    /// pointer usable by kernels / cuBLAS
    T *data() const { return m_dev; }
    /// host pointer (only for MemoryKind::Host / Shared)
    T *host() const { return m_ptr; }
    std::size_t size() const { return m_n; }
    explicit operator bool() const { return m_ptr != nullptr; }

  private:
    std::size_t m_n = 0;
    MemoryKind m_kind = MemoryKind::Device;
    T *m_ptr = nullptr;
    T *m_dev = nullptr;
};

/// Pinned host allocation (cudaHostAlloc default flags), move-only.
template <typename T> class PinnedBuffer {
  public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(std::size_t n) : m_n(n) {
        if (n)
            check(cudaHostAlloc(reinterpret_cast<void **>(&m_ptr), n * sizeof(T), cudaHostAllocDefault), "cudaHostAlloc");
    }
    PinnedBuffer(const PinnedBuffer &) = delete;
    PinnedBuffer &operator=(const PinnedBuffer &) = delete;
    PinnedBuffer(PinnedBuffer &&o) noexcept : m_n(o.m_n), m_ptr(o.m_ptr) {
        o.m_ptr = nullptr;
        o.m_n = 0;
    }
    PinnedBuffer &operator=(PinnedBuffer &&o) noexcept {
        if (this != &o) {
            release();
            m_n = o.m_n;
            m_ptr = o.m_ptr;
            o.m_ptr = nullptr;
            o.m_n = 0;
        }
        return *this;
    }
    ~PinnedBuffer() { release(); }
    void release() noexcept {
        if (m_ptr) {
            (void)cudaFreeHost(m_ptr);
            m_ptr = nullptr;
            m_n = 0;
        }
    }
    T *data() const { return m_ptr; }
    std::size_t size() const { return m_n; }
    explicit operator bool() const { return m_ptr != nullptr; }

  private:
    std::size_t m_n = 0;
    T *m_ptr = nullptr;
};

inline std::vector<std::uint32_t> shuffle_permutation(std::uint32_t n, std::mt19937 &g) {
    std::vector<std::uint32_t> perm(n);
    for (std::uint32_t i = 0; i < n; ++i)
        perm[i] = i;
    for (std::uint32_t i = 0; i + 1 < n; ++i) {
        std::uint32_t j = i + static_cast<std::uint32_t>(g()) % (n - i);
        std::swap(perm[i], perm[j]);
    }
    return perm;
}

/// A completed piece of work: the event recorded after it and the stream it ran on.
struct Ev {
    cudaEvent_t e = nullptr;
    int stream = -1;
};

} // namespace detail

// ============================================================================
//                                   Network
// ============================================================================
template <typename T> class Network {
    static_assert(std::is_floating_point_v<T>, "Network<T> requires float or double");

  public:
    using history_t = std::vector<std::vector<std::vector<T>>>;

    Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
            AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom,
            const std::vector<std::vector<T>> &initial_weights, const std::vector<std::vector<T>> &initial_biases,
            Options options = Options{});
    Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
            AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom, unsigned seed,
            Options options = Options{});
    Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
            AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom, Options options = Options{});
    ~Network() noexcept;
    Network(const Network &) = delete;
    Network &operator=(const Network &) = delete;

    std::vector<T> train(const std::vector<T> &input_samples, const std::vector<T> &target_samples, unsigned num_samples,
                         unsigned batch_size, unsigned max_epochs);
    std::vector<T> predict(const std::vector<T> &input_samples, unsigned num_samples, unsigned batch_size = 0);

    std::pair<history_t, history_t> weights_biases() const { return {m_weights_history, m_biases_history}; }
    std::vector<std::vector<T>> weights() const;
    std::vector<std::vector<T>> biases() const;
    void set_weights_biases(const std::vector<std::vector<T>> &new_weights, const std::vector<std::vector<T>> &new_biases);

    const Profile &profile() const { return m_prof.profile(); }
    void reset_profile() { m_prof.reset(); }
    const Options &options() const { return m_opts; }
    const std::vector<LayerDescription> &layers() const { return m_layers; }
    std::string device_name() const { return m_info.name; }
    DeviceInfo device_info() const { return m_info; }
    std::string blas_backend() const { return m_blas.handwritten() ? "tiled" : compiled_blas_backends().front(); }
    std::size_t parameter_count() const;
    std::size_t num_layers() const { return m_layers.size(); }

  private:
    using Buf = detail::DevBuffer<T>;
    using Ev = detail::Ev;
    using ev_list = std::vector<Ev>;

    // ---- configuration ----
    std::vector<LayerDescription> m_layers;
    T m_lr{};
    Regularization<T> m_reg;
    MomentumConfig<T> m_mom;
    BackPropagation m_bp;
    AdaptiveLearningRate<T> m_adapt;
    StopCriteria<T> m_stop;
    Options m_opts;
    MemoryKind m_kind = MemoryKind::Device;
    std::size_t m_L = 0;
    unsigned m_block = 256; ///< threads per block for element-wise kernels

    // ---- CUDA ----
    int m_ordinal = 0;
    DeviceInfo m_info;
    std::vector<cudaStream_t> m_streams;
    std::vector<cudaEvent_t> m_events; ///< dependency events (no timing), recycled every epoch
    std::size_t m_event_pos = 0;
    Blas m_blas;
    Profiler m_prof;
    bool m_capturing = false;

    // ---- parameters and optimiser state ----
    std::vector<Buf> m_W, m_b, m_mW, m_vW, m_mb, m_vb;

    // ---- workspace ----
    std::size_t m_cap = 0;
    std::vector<Buf> m_act, m_net, m_delta, m_gW, m_gb;
    Buf m_ones;
    detail::DevBuffer<double> m_loss_acc;
    Buf m_loss_acc_t;
    Buf m_reg_tmp;
    detail::PinnedBuffer<double> m_host_scalars;
    detail::PinnedBuffer<T> m_host_scalar_t;

    // ---- bookkeeping ----
    unsigned m_adam_step = 0;
    history_t m_weights_history, m_biases_history;

    // ---- helpers ----
    void init_device();
    void init_parameters(const std::vector<std::vector<T>> *w, const std::vector<std::vector<T>> *b, unsigned seed);
    void ensure_workspace(std::size_t batch);
    void release_workspace();
    void snapshot_history();
    void sync_all();
    cudaStream_t stream(int i) const { return m_streams[static_cast<std::size_t>(i) % m_streams.size()]; }
    int stream_for_layer(std::size_t l) const { return static_cast<int>(l % m_streams.size()); }
    Ev record(int s);
    template <typename F> Ev submit(int s, const ev_list &deps, Phase phase, F launch, std::uint64_t bytes = 0);
    unsigned grid(std::size_t n) const { return static_cast<unsigned>((n + m_block - 1) / m_block); }

    Ev forward_layer(std::size_t l, const T *in, std::size_t B, int s, const ev_list &deps);
    Ev output_delta_loss(const T *targets, std::size_t B, int s, const ev_list &deps);
    Ev hidden_delta(std::size_t l, std::size_t B, int s, const ev_list &deps);
    void gradients(std::size_t l, const T *in, std::size_t B, int s, const ev_list &deps, Ev &ev_w, Ev &ev_b);
    Ev update(std::size_t l, const kernels::StepScalars<T> *sp, int s, const ev_list &deps);
    Ev penalty(const ev_list &deps);
    void run_batch(const T *x_batch, const T *targets, std::size_t cur, const kernels::StepScalars<T> *sp,
                   const ev_list &first_deps, bool fine, bool direct, ev_list &ev_upd, ev_list &ev_act, ev_list &ev_delta,
                   ev_list &ev_gw, ev_list &ev_gb, ev_list &chain_next);
};

/* ============================================================================
                                  IMPLEMENTATION
============================================================================ */

template <typename T> void Network<T>::init_device() {
    m_ordinal = select_device(m_opts.device);
    check(cudaSetDevice(m_ordinal), "cudaSetDevice");
    m_info = describe(m_ordinal);
    m_kind = m_opts.memory;
    check_blas_option(m_opts.blas);
    m_blas = Blas(use_handwritten(m_opts.blas));
    if (m_opts.workgroup_size) {
        // block sizes must be powers of two <= 1024 for the loss reduction
        unsigned b = 1;
        while (b * 2 <= std::min<unsigned>(m_opts.workgroup_size, 1024))
            b *= 2;
        m_block = b;
    }
    std::size_t S = (m_opts.queue == QueueOrder::InOrder) ? 1 : std::max<unsigned>(1, m_opts.streams);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    // HIP's multi-stream graph capture (fork/join through events) crashes on ROCm 7.2:
    // capture the graph from a single stream there.
    if (m_opts.queue == QueueOrder::Graph)
        S = 1;
#endif
    for (std::size_t i = 0; i < S; ++i) {
        cudaStream_t s;
        check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate");
        m_streams.push_back(s);
    }
    m_prof = Profiler(m_opts.profile);
    m_host_scalars = detail::PinnedBuffer<double>(2);
    m_host_scalar_t = detail::PinnedBuffer<T>(1);
}

template <typename T>
Network<T>::Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
                    AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom,
                    const std::vector<std::vector<T>> &initial_weights, const std::vector<std::vector<T>> &initial_biases,
                    Options options)
    : m_layers(std::move(layers)), m_lr(learning_rate), m_reg(reg), m_mom(mom), m_bp(bp), m_adapt(adapt), m_stop(stop),
      m_opts(std::move(options)) {
    init_device();
    init_parameters(&initial_weights, &initial_biases, 0);
}

template <typename T>
Network<T>::Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
                    AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom, unsigned seed,
                    Options options)
    : m_layers(std::move(layers)), m_lr(learning_rate), m_reg(reg), m_mom(mom), m_bp(bp), m_adapt(adapt), m_stop(stop),
      m_opts(std::move(options)) {
    init_device();
    init_parameters(nullptr, nullptr, seed);
}

template <typename T>
Network<T>::Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
                    AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom, Options options)
    : Network(std::move(layers), learning_rate, reg, bp, adapt, stop, mom, static_cast<unsigned>(std::time(nullptr)),
              std::move(options)) {}

template <typename T> Network<T>::~Network() noexcept {
    for (auto s : m_streams)
        (void)cudaStreamSynchronize(s);
    (void)cudaGetLastError();
    m_W.clear();
    m_b.clear();
    m_mW.clear();
    m_vW.clear();
    m_mb.clear();
    m_vb.clear();
    release_workspace();
    for (auto e : m_events)
        (void)cudaEventDestroy(e);
    for (auto s : m_streams)
        (void)cudaStreamDestroy(s);
}

template <typename T>
void Network<T>::init_parameters(const std::vector<std::vector<T>> *w, const std::vector<std::vector<T>> *b, unsigned seed) {
    if (m_layers.size() < 2)
        throw std::invalid_argument("cudann: a network needs at least an input and an output layer");
    for (const auto &l : m_layers)
        if (l.neurons == 0)
            throw std::invalid_argument("cudann: layer neuron count must be greater than 0");
    m_L = m_layers.size() - 1;
    const bool random_init = (w == nullptr || b == nullptr);
    if (!random_init) {
        if (w->size() != m_L)
            throw std::invalid_argument("cudann: initial weights have " + std::to_string(w->size()) + " layers, expected " +
                                        std::to_string(m_L));
        if (b->size() != m_L)
            throw std::invalid_argument("cudann: initial biases have " + std::to_string(b->size()) + " layers, expected " +
                                        std::to_string(m_L));
        for (std::size_t l = 0; l < m_L; ++l) {
            const std::size_t nw = std::size_t(m_layers[l].neurons) * m_layers[l + 1].neurons;
            const std::size_t n_out = m_layers[l + 1].neurons;
            if ((*w)[l].size() != nw)
                throw std::invalid_argument("cudann: initial weights of layer " + std::to_string(l) + " have " +
                                            std::to_string((*w)[l].size()) + " values, expected " + std::to_string(nw));
            if ((*b)[l].size() != n_out)
                throw std::invalid_argument("cudann: initial biases of layer " + std::to_string(l) + " have " +
                                            std::to_string((*b)[l].size()) + " values, expected " + std::to_string(n_out));
        }
    }
    std::mt19937 rng(seed == 0 ? static_cast<unsigned>(std::time(nullptr)) : seed);
    std::uniform_real_distribution<T> dist(T(-0.1), T(0.1));
    std::vector<std::vector<T>> host_w(m_L), host_b(m_L);
    for (std::size_t l = 0; l < m_L; ++l) {
        const std::size_t nw = std::size_t(m_layers[l].neurons) * m_layers[l + 1].neurons;
        const std::size_t n_out = m_layers[l + 1].neurons;
        if (random_init) {
            host_w[l].resize(nw);
            host_b[l].resize(n_out);
            for (auto &x : host_w[l])
                x = dist(rng);
            for (auto &x : host_b[l])
                x = dist(rng);
        } else {
            host_w[l] = (*w)[l];
            host_b[l] = (*b)[l];
        }
    }
    m_W.clear();
    m_b.clear();
    m_mW.clear();
    m_vW.clear();
    m_mb.clear();
    m_vb.clear();
    cudaStream_t s = stream(0);
    try {
        for (std::size_t l = 0; l < m_L; ++l) {
            const std::size_t nw = host_w[l].size(), n_out = host_b[l].size();
            m_W.emplace_back(nw, m_kind);
            m_b.emplace_back(n_out, m_kind);
            m_mW.emplace_back(nw, m_kind);
            m_vW.emplace_back(nw, m_kind);
            m_mb.emplace_back(n_out, m_kind);
            m_vb.emplace_back(n_out, m_kind);
            check(cudaMemcpyAsync(m_W[l].data(), host_w[l].data(), nw * sizeof(T), cudaMemcpyHostToDevice, s), "memcpy W");
            check(cudaMemcpyAsync(m_b[l].data(), host_b[l].data(), n_out * sizeof(T), cudaMemcpyHostToDevice, s), "memcpy b");
            check(cudaMemsetAsync(m_mW[l].data(), 0, nw * sizeof(T), s), "memset");
            check(cudaMemsetAsync(m_vW[l].data(), 0, nw * sizeof(T), s), "memset");
            check(cudaMemsetAsync(m_mb[l].data(), 0, n_out * sizeof(T), s), "memset");
            check(cudaMemsetAsync(m_vb[l].data(), 0, n_out * sizeof(T), s), "memset");
        }
        m_prof.timed_wait([&] { check(cudaStreamSynchronize(s), "cudaStreamSynchronize"); });
    } catch (...) {
        (void)cudaStreamSynchronize(s);
        (void)cudaGetLastError();
        throw;
    }
    m_weights_history.clear();
    m_biases_history.clear();
    m_weights_history.push_back(std::move(host_w));
    m_biases_history.push_back(std::move(host_b));
}

template <typename T> std::size_t Network<T>::parameter_count() const {
    std::size_t n = 0;
    for (std::size_t l = 0; l < m_L; ++l)
        n += std::size_t(m_layers[l].neurons) * m_layers[l + 1].neurons + m_layers[l + 1].neurons;
    return n;
}

// ---------------------------------------------------------------- streams / events

template <typename T> void Network<T>::sync_all() {
    m_prof.timed_wait([&] {
        for (auto s : m_streams)
            check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
    });
    m_prof.flush();
    m_event_pos = 0; // every dependency event is complete now
}

template <typename T> detail::Ev Network<T>::record(int s) {
    if (m_streams.size() == 1 && !m_capturing)
        return Ev{nullptr, s}; // one stream: order is implicit
    if (m_event_pos == m_events.size()) {
        cudaEvent_t e;
        check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "cudaEventCreate");
        m_events.push_back(e);
    }
    cudaEvent_t e = m_events[m_event_pos++];
    check(cudaEventRecord(e, stream(s)), "cudaEventRecord");
    return Ev{e, s};
}

template <typename T>
template <typename F>
detail::Ev Network<T>::submit(int s, const ev_list &deps, Phase phase, F launch, std::uint64_t bytes) {
    cudaStream_t st = stream(s);
    for (const Ev &d : deps)
        if (d.e && d.stream != s)
            check(cudaStreamWaitEvent(st, d.e, 0), "cudaStreamWaitEvent");
    m_prof.record(phase, st, [&] { launch(st); }, bytes);
    check(cudaGetLastError(), "kernel launch");
    return record(s);
}

// ---------------------------------------------------------------- memory

template <typename T> void Network<T>::ensure_workspace(std::size_t batch) {
    if (batch <= m_cap && !m_act.empty())
        return;
    release_workspace();
    try {
        for (std::size_t l = 0; l <= m_L; ++l) {
            const bool needed = (l > 0) || !m_opts.direct_input;
            m_act.emplace_back(needed ? std::size_t(m_layers[l].neurons) * batch : 0, m_kind);
        }
        for (std::size_t l = 0; l < m_L; ++l) {
            const std::size_t n_in = m_layers[l].neurons, n_out = m_layers[l + 1].neurons;
            m_net.emplace_back(n_out * batch, m_kind);
            m_delta.emplace_back(n_out * batch, m_kind);
            m_gW.emplace_back(n_out * n_in, m_kind);
            m_gb.emplace_back(n_out, m_kind);
        }
        m_ones = Buf(batch, m_kind);
        m_loss_acc = detail::DevBuffer<double>(2, m_kind);
        m_loss_acc_t = Buf(1, m_kind);
        m_reg_tmp = Buf(2 * m_L, m_kind);
        cudaStream_t s = stream(0);
        kernels::fill_k<T><<<grid(batch), m_block, 0, s>>>(m_ones.data(), T(1), batch);
        check(cudaGetLastError(), "fill");
        check(cudaMemsetAsync(m_loss_acc.data(), 0, 2 * sizeof(double), s), "memset");
        check(cudaMemsetAsync(m_loss_acc_t.data(), 0, sizeof(T), s), "memset");
        check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
    } catch (...) {
        for (auto s : m_streams)
            (void)cudaStreamSynchronize(s);
        (void)cudaGetLastError();
        release_workspace();
        throw;
    }
    m_cap = batch;
}

template <typename T> void Network<T>::release_workspace() {
    if (!m_act.empty() || m_ones)
        for (auto s : m_streams)
            (void)cudaStreamSynchronize(s);
    m_act.clear();
    m_net.clear();
    m_delta.clear();
    m_gW.clear();
    m_gb.clear();
    m_ones.release();
    m_loss_acc.release();
    m_loss_acc_t.release();
    m_reg_tmp.release();
    m_cap = 0;
}

template <typename T> std::vector<std::vector<T>> Network<T>::weights() const {
    std::vector<std::vector<T>> out(m_L);
    for (std::size_t l = 0; l < m_L; ++l) {
        out[l].resize(m_W[l].size());
        check(cudaMemcpy(out[l].data(), m_W[l].data(), m_W[l].size() * sizeof(T), cudaMemcpyDeviceToHost), "memcpy D2H");
    }
    return out;
}

template <typename T> std::vector<std::vector<T>> Network<T>::biases() const {
    std::vector<std::vector<T>> out(m_L);
    for (std::size_t l = 0; l < m_L; ++l) {
        out[l].resize(m_b[l].size());
        check(cudaMemcpy(out[l].data(), m_b[l].data(), m_b[l].size() * sizeof(T), cudaMemcpyDeviceToHost), "memcpy D2H");
    }
    return out;
}

template <typename T> void Network<T>::snapshot_history() {
    m_weights_history.push_back(weights());
    m_biases_history.push_back(biases());
}

template <typename T>
void Network<T>::set_weights_biases(const std::vector<std::vector<T>> &new_weights,
                                    const std::vector<std::vector<T>> &new_biases) {
    if (new_weights.size() != m_L || new_biases.size() != m_L)
        throw std::invalid_argument("cudann: layer count mismatch in set_weights_biases");
    for (std::size_t l = 0; l < m_L; ++l) {
        if (new_weights[l].size() != m_W[l].size())
            throw std::invalid_argument("cudann: weight size mismatch for layer " + std::to_string(l));
        if (new_biases[l].size() != m_b[l].size())
            throw std::invalid_argument("cudann: bias size mismatch for layer " + std::to_string(l));
    }
    sync_all();
    for (std::size_t l = 0; l < m_L; ++l) {
        check(cudaMemcpy(m_W[l].data(), new_weights[l].data(), m_W[l].size() * sizeof(T), cudaMemcpyHostToDevice), "memcpy W");
        check(cudaMemcpy(m_b[l].data(), new_biases[l].data(), m_b[l].size() * sizeof(T), cudaMemcpyHostToDevice), "memcpy b");
    }
    m_weights_history.clear();
    m_biases_history.clear();
    m_weights_history.push_back(new_weights);
    m_biases_history.push_back(new_biases);
}

// ---------------------------------------------------------------- phases

template <typename T>
detail::Ev Network<T>::forward_layer(std::size_t l, const T *in, std::size_t B, int s, const ev_list &deps) {
    const std::size_t M = m_layers[l + 1].neurons, K = m_layers[l].neurons;
    T *net = m_net[l].data();
    T *out = m_act[l + 1].data();
    const T *bias = m_b[l].data();
    const T *W = m_W[l].data();
    Ev ge = submit(s, deps, Phase::Gemm, [&](cudaStream_t st) {
        m_blas.gemm(st, false, false, int(M), int(B), int(K), T(1), W, int(M), in, int(K), T(0), net, int(M));
    });
    const std::size_t n = M * B;
    const ActivationType act = m_layers[l + 1].activation;
    const unsigned g = grid(n), blk = m_block;
    if (m_opts.specialized_kernels) {
        return dispatch_activation(act, [&](auto tag) {
            constexpr ActivationType A = decltype(tag)::value;
            return submit(s, {ge}, Phase::Act, [&](cudaStream_t st) {
                kernels::activate_k<T, kernels::ActF<A>><<<g, blk, 0, st>>>(net, out, bias, n, M, kernels::ActF<A>{});
            });
        });
    }
    return submit(s, {ge}, Phase::Act, [&](cudaStream_t st) {
        kernels::activate_k<T, kernels::ActRT><<<g, blk, 0, st>>>(net, out, bias, n, M, kernels::ActRT{act});
    });
}

template <typename T>
detail::Ev Network<T>::output_delta_loss(const T *targets, std::size_t B, int s, const ev_list &deps) {
    const std::size_t l = m_L - 1;
    const std::size_t M = m_layers[m_L].neurons;
    const std::size_t n = M * B;
    const T *net = m_net[l].data();
    const T *out = m_act[m_L].data();
    T *delta = m_delta[l].data();
    const ActivationType act = m_layers[m_L].activation;
    const bool from_out = m_opts.derivative_from_output;
    const unsigned g = grid(n), blk = m_block;
    double *acc = m_loss_acc.data();
    T *acc_t = m_loss_acc_t.data();
    auto launch = [&](auto df) {
        return submit(s, deps, Phase::Loss, [&](cudaStream_t st) {
            if (m_opts.loss_reduction)
                kernels::output_delta_loss_k<T, decltype(df)><<<g, blk, 0, st>>>(targets, out, net, delta, n, df, acc);
            else
                kernels::output_delta_loss_atomic_k<T, decltype(df)><<<g, blk, 0, st>>>(targets, out, net, delta, n, df, acc_t);
        });
    };
    if (m_opts.specialized_kernels) {
        return dispatch_activation(act, [&](auto tag) {
            constexpr ActivationType A = decltype(tag)::value;
            if (from_out)
                return launch(kernels::DfF<A, true>{});
            return launch(kernels::DfF<A, false>{});
        });
    }
    return launch(kernels::DfRT{act, from_out});
}

template <typename T>
detail::Ev Network<T>::hidden_delta(std::size_t l, std::size_t B, int s, const ev_list &deps) {
    const std::size_t M = m_layers[l + 1].neurons, K = m_layers[l + 2].neurons;
    T *delta = m_delta[l].data();
    const T *W_next = m_W[l + 1].data();
    const T *delta_next = m_delta[l + 1].data();
    Ev ge = submit(s, deps, Phase::Gemm, [&](cudaStream_t st) {
        m_blas.gemm(st, true, false, int(M), int(B), int(K), T(1), W_next, int(K), delta_next, int(K), T(0), delta, int(M));
    });
    const std::size_t n = M * B;
    const T *net = m_net[l].data();
    const T *out = m_act[l + 1].data();
    const ActivationType act = m_layers[l + 1].activation;
    const bool from_out = m_opts.derivative_from_output;
    const unsigned g = grid(n), blk = m_block;
    auto launch = [&](auto df) {
        return submit(s, {ge}, Phase::Delta, [&](cudaStream_t st) {
            kernels::hidden_delta_k<T, decltype(df)><<<g, blk, 0, st>>>(delta, out, net, n, df);
        });
    };
    if (m_opts.specialized_kernels) {
        return dispatch_activation(act, [&](auto tag) {
            constexpr ActivationType A = decltype(tag)::value;
            if (from_out)
                return launch(kernels::DfF<A, true>{});
            return launch(kernels::DfF<A, false>{});
        });
    }
    return launch(kernels::DfRT{act, from_out});
}

template <typename T>
void Network<T>::gradients(std::size_t l, const T *in, std::size_t B, int s, const ev_list &deps, Ev &ev_w, Ev &ev_b) {
    const std::size_t M = m_layers[l + 1].neurons, N = m_layers[l].neurons;
    const T *delta = m_delta[l].data();
    const T invB = T(1) / static_cast<T>(B);
    T *gW = m_gW[l].data();
    T *gb = m_gb[l].data();
    ev_w = submit(s, deps, Phase::Gemm, [&](cudaStream_t st) {
        m_blas.gemm(st, false, true, int(M), int(N), int(B), invB, delta, int(M), in, int(N), T(0), gW, int(M));
    });
    if (m_opts.bias_gemv) {
        const T *ones = m_ones.data();
        ev_b = submit(s, deps, Phase::BiasGrad, [&](cudaStream_t st) {
            m_blas.gemv(st, false, int(M), int(B), invB, delta, int(M), ones, 1, T(0), gb, 1);
        });
    } else {
        const unsigned g = grid(M), blk = m_block;
        ev_b = submit(s, deps, Phase::BiasGrad, [&](cudaStream_t st) {
            kernels::bias_grad_k<T><<<g, blk, 0, st>>>(delta, gb, M, B);
        });
    }
}

template <typename T>
detail::Ev Network<T>::update(std::size_t l, const kernels::StepScalars<T> *sp, int s, const ev_list &deps) {
    kernels::UpdateParams<T> u;
    u.strategy = static_cast<int>(m_adapt.strategy);
    u.momentum = m_mom.type == MomentumConfig<T>::Type::Classical;
    u.l1 = m_reg.uses_l1();
    u.l2 = m_reg.uses_l2();
    u.host_corr = m_opts.host_adam_correction;
    u.mu = m_mom.momentum_rate;
    u.lr = m_lr;
    u.eps = m_adapt.epsilon;
    u.beta1 = m_adapt.beta1;
    u.beta2 = m_adapt.beta2;
    u.lambda1 = m_reg.lambda1;
    u.lambda2 = m_reg.lambda2;
    const std::size_t nw = m_W[l].size(), nb = m_b[l].size();
    T *W = m_W[l].data(), *b = m_b[l].data();
    const T *gW = m_gW[l].data(), *gb = m_gb[l].data();
    T *mW = m_mW[l].data(), *vW = m_vW[l].data(), *mb = m_mb[l].data(), *vb = m_vb[l].data();
    const unsigned g = grid(nw + nb), blk = m_block;
    return submit(s, deps, Phase::Update, [&](cudaStream_t st) {
        kernels::update_k<T><<<g, blk, 0, st>>>(W, gW, mW, vW, nw, b, gb, mb, vb, nb, u, sp);
    });
}

template <typename T> detail::Ev Network<T>::penalty(const ev_list &deps) {
    const bool l1 = m_reg.uses_l1(), l2 = m_reg.uses_l2();
    ev_list evs;
    T *tmp = m_reg_tmp.data();
    for (std::size_t l = 0; l < m_L; ++l) {
        const int n = int(m_W[l].size());
        const T *W = m_W[l].data();
        if (l1)
            evs.push_back(submit(0, {deps[l]}, Phase::Reg, [&](cudaStream_t st) { m_blas.asum(st, n, W, 1, tmp + 2 * l); }));
        if (l2)
            evs.push_back(submit(0, {deps[l]}, Phase::Reg, [&](cudaStream_t st) { m_blas.nrm2(st, n, W, 1, tmp + 2 * l + 1); }));
    }
    const std::size_t L = m_L;
    const T lambda1 = m_reg.lambda1, lambda2 = m_reg.lambda2;
    double *acc = m_loss_acc.data();
    return submit(0, evs, Phase::Reg, [&](cudaStream_t st) {
        kernels::penalty_k<T><<<1, 32, 0, st>>>(tmp, L, l1, l2, lambda1, lambda2, acc);
    });
}

/// One mini-batch: forward, loss/delta, hidden deltas, gradients, updates.
template <typename T>
void Network<T>::run_batch(const T *x_batch, const T *targets, std::size_t cur, const kernels::StepScalars<T> *sp,
                           const ev_list &first_deps, bool fine, bool direct, ev_list &ev_upd, ev_list &ev_act,
                           ev_list &ev_delta, ev_list &ev_gw, ev_list &ev_gb, ev_list &chain_next) {
    const std::size_t n_in = m_layers.front().neurons;
    if (fine) {
        const T *in = x_batch;
        ev_list deps0 = first_deps;
        if (!direct) {
            ev_list cdeps = deps0;
            cdeps.push_back(ev_upd[0]);
            T *act0 = m_act[0].data();
            Ev ce = submit(0, cdeps, Phase::Other, [&](cudaStream_t st) {
                check(cudaMemcpyAsync(act0, x_batch, cur * n_in * sizeof(T), cudaMemcpyDeviceToDevice, st), "memcpy D2D");
            });
            deps0 = {ce};
            in = act0;
        }
        for (std::size_t l = 0; l < m_L; ++l) {
            ev_list deps = (l == 0) ? deps0 : ev_list{ev_act[l]};
            deps.push_back(ev_upd[l]);
            if (l + 1 < m_L)
                deps.push_back(ev_upd[l + 1]);
            ev_act[l + 1] = forward_layer(l, (l == 0) ? in : m_act[l].data(), cur, stream_for_layer(l), deps);
        }
        {
            ev_list deps{ev_act[m_L]};
            deps.insert(deps.end(), first_deps.begin(), first_deps.end());
            ev_delta[m_L - 1] = output_delta_loss(targets, cur, stream_for_layer(m_L - 1), deps);
        }
        for (std::size_t l = m_L - 1; l-- > 0;)
            ev_delta[l] = hidden_delta(l, cur, stream_for_layer(l), {ev_delta[l + 1]});
        for (std::size_t l = 0; l < m_L; ++l) {
            gradients(l, (l == 0) ? in : m_act[l].data(), cur, stream_for_layer(l), {ev_delta[l]}, ev_gw[l], ev_gb[l]);
            ev_list deps{ev_gw[l], ev_gb[l]};
            if (l >= 1)
                deps.push_back(ev_delta[l - 1]); // the hidden-delta GEMM of layer l reads W_l
            ev_upd[l] = update(l, sp, stream_for_layer(l), deps);
        }
    } else {
        // 0.1-style coarse graph, everything on stream 0
        ev_list chain = chain_next.empty() ? first_deps : chain_next;
        const T *in = x_batch;
        if (!direct) {
            T *act0 = m_act[0].data();
            Ev ce = submit(0, chain, Phase::Other, [&](cudaStream_t st) {
                check(cudaMemcpyAsync(act0, x_batch, cur * n_in * sizeof(T), cudaMemcpyDeviceToDevice, st), "memcpy D2D");
            });
            chain = {ce};
            in = act0;
        }
        for (std::size_t l = 0; l < m_L; ++l)
            chain = {forward_layer(l, (l == 0) ? in : m_act[l].data(), cur, 0, chain)};
        chain = {output_delta_loss(targets, cur, 0, chain)};
        ev_delta[m_L - 1] = chain.front();
        for (std::size_t l = m_L - 1; l-- > 0;) {
            chain = {hidden_delta(l, cur, 0, chain)};
            ev_delta[l] = chain.front();
        }
        ev_list grads;
        for (std::size_t l = 0; l < m_L; ++l) {
            gradients(l, (l == 0) ? in : m_act[l].data(), cur, 0, chain, ev_gw[l], ev_gb[l]);
            grads.push_back(ev_gw[l]);
            grads.push_back(ev_gb[l]);
        }
        ev_list updates;
        for (std::size_t l = 0; l < m_L; ++l) {
            ev_upd[l] = update(l, sp, 0, grads);
            updates.push_back(ev_upd[l]);
        }
        chain_next = updates;
    }
}

// ---------------------------------------------------------------- training

template <typename T>
std::vector<T> Network<T>::train(const std::vector<T> &input_samples, const std::vector<T> &target_samples,
                                 unsigned num_samples, unsigned batch_size, unsigned max_epochs) {
    const auto t_start = Profiler::clock::now();
    const std::size_t n_in = m_layers.front().neurons, n_out = m_layers.back().neurons;
    if (num_samples == 0)
        throw std::invalid_argument("cudann: number of samples cannot be zero");
    if (batch_size == 0)
        throw std::invalid_argument("cudann: batch size must be greater than 0");
    if (max_epochs == 0)
        throw std::invalid_argument("cudann: max_epochs must be greater than 0");
    if (input_samples.size() != std::size_t(num_samples) * n_in)
        throw std::invalid_argument("cudann: input has " + std::to_string(input_samples.size()) + " values, expected " +
                                    std::to_string(std::size_t(num_samples) * n_in));
    if (target_samples.size() != std::size_t(num_samples) * n_out)
        throw std::invalid_argument("cudann: targets have " + std::to_string(target_samples.size()) + " values, expected " +
                                    std::to_string(std::size_t(num_samples) * n_out));
    check(cudaSetDevice(m_ordinal), "cudaSetDevice");
    const std::size_t N = num_samples;
    const std::size_t B = std::min<std::size_t>(batch_size, N);
    const std::size_t n_batches = (N + B - 1) / B;
    const bool fine = m_opts.fine_deps;
    const bool direct = m_opts.direct_input;
    const bool use_reg = m_reg.uses_l1() || m_reg.uses_l2();
    const bool graphs = m_opts.queue == QueueOrder::Graph;
    using Step = kernels::StepScalars<T>;

    ensure_workspace(B);
    m_adam_step = 0;

    // Buffers declared before the guard: on an exception the guard drains the
    // streams before they are released.
    Buf X, Y, Xs, Ys;
    detail::DevBuffer<std::uint32_t> perm_dev;
    detail::PinnedBuffer<T> stage_x, stage_y;
    detail::PinnedBuffer<Step> step_host;
    detail::DevBuffer<Step> step_dev;
    std::vector<std::uint32_t> perm_host;
    std::vector<cudaGraphExec_t> graph_exec;
    struct Quiesce {
        Network &net;
        std::vector<cudaGraphExec_t> &execs;
        ~Quiesce() {
            for (auto s : net.m_streams)
                (void)cudaStreamSynchronize(s);
            for (auto g : execs)
                (void)cudaGraphExecDestroy(g);
            execs.clear();
            net.m_capturing = false;
            (void)cudaGetLastError();
        }
    } quiesce{*this, graph_exec};

    cudaStream_t s0 = stream(0);
    ev_list ev_ds;
    {
        X = Buf(N * n_in, m_kind);
        Y = Buf(N * n_out, m_kind);
        if (m_opts.shuffle) {
            Xs = Buf(N * n_in, m_kind);
            Ys = Buf(N * n_out, m_kind);
            perm_dev = detail::DevBuffer<std::uint32_t>(N, m_kind);
        }
        step_host = detail::PinnedBuffer<Step>(n_batches);
        step_dev = detail::DevBuffer<Step>(n_batches, MemoryKind::Device);
        const T *src_x = input_samples.data();
        const T *src_y = target_samples.data();
        if (m_opts.pinned_host) {
            const auto t0 = Profiler::clock::now();
            stage_x = detail::PinnedBuffer<T>(N * n_in);
            stage_y = detail::PinnedBuffer<T>(N * n_out);
            std::memcpy(stage_x.data(), src_x, N * n_in * sizeof(T));
            std::memcpy(stage_y.data(), src_y, N * n_out * sizeof(T));
            m_prof.profile().h2d_ns += m_prof.enabled() ? Profiler::elapsed_ns(t0) : 0;
            src_x = stage_x.data();
            src_y = stage_y.data();
        }
        T *xd = X.data(), *yd = Y.data();
        ev_ds.push_back(submit(0, {}, Phase::H2D, [&](cudaStream_t st) {
            check(cudaMemcpyAsync(xd, src_x, N * n_in * sizeof(T), cudaMemcpyHostToDevice, st), "memcpy X");
        }, N * n_in * sizeof(T)));
        ev_ds.push_back(submit(0, {}, Phase::H2D, [&](cudaStream_t st) {
            check(cudaMemcpyAsync(yd, src_y, N * n_out * sizeof(T), cudaMemcpyHostToDevice, st), "memcpy Y");
        }, N * n_out * sizeof(T)));
        if (!m_opts.pinned_host) // pageable source: the copy is synchronous with respect to the host anyway
            check(cudaStreamSynchronize(s0), "cudaStreamSynchronize");
    }
    std::mt19937 shuffle_rng(m_opts.shuffle_seed);

    std::vector<T> losses;
    losses.reserve(max_epochs);
    ev_list ev_upd(m_L), ev_act(m_L + 1), ev_delta(m_L), ev_gw(m_L), ev_gb(m_L), chain_next;
    double *acc = m_loss_acc.data();
    T *acc_t = m_loss_acc_t.data();

    for (unsigned epoch = 0; epoch < max_epochs; ++epoch) {
        const auto t_epoch = Profiler::clock::now();
        // the streams are idle here (every epoch ends with a full sync): reset accumulators
        ev_list ev_fill;
        ev_fill.push_back(submit(0, {}, Phase::Other, [&](cudaStream_t st) {
            check(cudaMemsetAsync(acc, 0, 2 * sizeof(double), st), "memset");
            if (!m_opts.loss_reduction)
                check(cudaMemsetAsync(acc_t, 0, sizeof(T), st), "memset");
        }));
        const T *Xsrc = X.data();
        const T *Ysrc = Y.data();
        ev_list ev_data = ev_ds;
        if (m_opts.shuffle) {
            perm_host = detail::shuffle_permutation(static_cast<std::uint32_t>(N), shuffle_rng);
            std::uint32_t *pd = perm_dev.data();
            Ev ep = submit(0, ev_ds, Phase::H2D, [&](cudaStream_t st) {
                check(cudaMemcpyAsync(pd, perm_host.data(), N * sizeof(std::uint32_t), cudaMemcpyHostToDevice, st), "memcpy perm");
            }, N * sizeof(std::uint32_t));
            check(cudaStreamSynchronize(s0), "cudaStreamSynchronize"); // perm_host is pageable
            const T *x0 = X.data();
            const T *y0 = Y.data();
            T *xs = Xs.data();
            T *ys = Ys.data();
            Ev gx = submit(0, {ep}, Phase::Other, [&](cudaStream_t st) {
                kernels::gather_k<T><<<grid(N * n_in), m_block, 0, st>>>(x0, xs, pd, n_in, N * n_in);
            });
            Ev gy = submit(0, {gx}, Phase::Other, [&](cudaStream_t st) {
                kernels::gather_k<T><<<grid(N * n_out), m_block, 0, st>>>(y0, ys, pd, n_out, N * n_out);
            });
            ev_data = {gy};
            Xsrc = Xs.data();
            Ysrc = Ys.data();
        }
        ev_list first_deps = ev_data;
        first_deps.insert(first_deps.end(), ev_fill.begin(), ev_fill.end());
        chain_next.clear();
        for (auto &e : ev_upd)
            e = Ev{};

        // per-step scalars (learning rate schedule, Adam bias correction)
        T lr_eff = m_lr;
        if (m_adapt.strategy == AdaptiveLearningRate<T>::Strategy::LinearDecay) {
            const T gamma = (max_epochs > 1) ? static_cast<T>(epoch) / static_cast<T>(max_epochs - 1) : T(0);
            lr_eff = m_lr * (T(1) - gamma) + gamma * m_adapt.final_lr;
        }
        const bool adam = m_adapt.strategy == AdaptiveLearningRate<T>::Strategy::Adam;

        for (std::size_t bi = 0; bi < n_batches; ++bi) {
            const std::size_t start = bi * B;
            const std::size_t cur = std::min(B, N - start);
            if (adam)
                ++m_adam_step;
            Step &sh = step_host.data()[bi];
            sh.lr_eff = lr_eff;
            sh.t = static_cast<T>(m_adam_step);
            sh.c1 = (adam && m_opts.host_adam_correction)
                        ? T(1) / (T(1) - static_cast<T>(std::pow(static_cast<double>(m_adapt.beta1), m_adam_step)))
                        : T(0);
            sh.c2 = (adam && m_opts.host_adam_correction)
                        ? T(1) / (T(1) - static_cast<T>(std::pow(static_cast<double>(m_adapt.beta2), m_adam_step)))
                        : T(0);
            const Step *sp = step_dev.data() + bi;
            const T *x_batch = Xsrc + start * n_in;
            const T *targets = Ysrc + start * n_out;

            if (graphs && epoch > 0) {
                // replay the graph captured in the first epoch (it copies step_host[bi] itself)
                m_prof.record(Phase::Other, s0, [&] { check(cudaGraphLaunch(graph_exec[bi], s0), "cudaGraphLaunch"); });
                continue;
            }
            if (graphs) {
                check(cudaStreamBeginCapture(s0, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
                m_capturing = true;
                m_prof.suspend(true);
                // fork the other streams from the capturing stream
                Ev fork = record(0);
                for (std::size_t si = 1; si < m_streams.size(); ++si)
                    check(cudaStreamWaitEvent(stream(int(si)), fork.e, 0), "fork");
            }
            // the step scalars: a tiny H2D copy every batch (a node of the graph in Graph mode).
            // Inside a capture nothing may depend on uncaptured work (the epoch-start fill /
            // gather): stream order on s0 already sequences the graph after them.
            const ev_list batch_first = (bi == 0 && !graphs) ? first_deps : ev_list{};
            Ev es = submit(0, batch_first, Phase::H2D, [&](cudaStream_t st) {
                check(cudaMemcpyAsync(step_dev.data() + bi, step_host.data() + bi, sizeof(Step), cudaMemcpyHostToDevice, st),
                      "memcpy step");
            }, sizeof(Step));
            ev_list deps_batch = batch_first;
            deps_batch.push_back(es);
            run_batch(x_batch, targets, cur, sp, deps_batch, fine, direct, ev_upd, ev_act, ev_delta, ev_gw, ev_gb, chain_next);
            if (graphs) {
                // join every stream back into the capturing stream, end the capture, instantiate
                for (std::size_t si = 1; si < m_streams.size(); ++si) {
                    Ev j = record(int(si));
                    check(cudaStreamWaitEvent(s0, j.e, 0), "join");
                }
                cudaGraph_t graph;
                check(cudaStreamEndCapture(s0, &graph), "cudaStreamEndCapture");
                m_capturing = false;
                m_prof.suspend(false);
                cudaGraphExec_t exec;
                check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "cudaGraphInstantiate");
                (void)cudaGraphDestroy(graph);
                graph_exec.push_back(exec);
                m_prof.record(Phase::Other, s0, [&] { check(cudaGraphLaunch(exec, s0), "cudaGraphLaunch"); });
                // after the launch the next batch depends on this graph through stream order on s0
                for (auto &e : ev_upd)
                    e = Ev{};
                chain_next.clear();
            }
        }

        // ---- epoch end: penalty, loss readback, synchronisation ----
        ev_list tail;
        if (use_reg) {
            ev_list deps = graphs ? ev_list(m_L, Ev{}) : ev_upd;
            tail.push_back(penalty(deps));
        }
        ev_list rb_deps = tail;
        rb_deps.insert(rb_deps.end(), ev_upd.begin(), ev_upd.end());
        if (!graphs)
            rb_deps.push_back(ev_delta[m_L - 1]);
        double *hs = m_host_scalars.data();
        submit(0, rb_deps, Phase::D2H, [&](cudaStream_t st) {
            check(cudaMemcpyAsync(hs, acc, 2 * sizeof(double), cudaMemcpyDeviceToHost, st), "memcpy loss");
        }, 2 * sizeof(double));
        if (!m_opts.loss_reduction) {
            T *ht = m_host_scalar_t.data();
            submit(0, rb_deps, Phase::D2H, [&](cudaStream_t st) {
                check(cudaMemcpyAsync(ht, acc_t, sizeof(T), cudaMemcpyDeviceToHost, st), "memcpy loss");
            }, sizeof(T));
        }
        sync_all();
        const double data_loss = m_opts.loss_reduction ? hs[0] : static_cast<double>(m_host_scalar_t.data()[0]);
        const double total = data_loss / static_cast<double>(N) + (use_reg ? hs[1] : 0.0);
        losses.push_back(static_cast<T>(total));
        if (m_opts.record_history)
            snapshot_history();
        if (m_prof.enabled()) {
            m_prof.profile().epoch_wall_ns.push_back(Profiler::elapsed_ns(t_epoch));
            ++m_prof.profile().epochs;
            m_prof.profile().batches += n_batches;
        }
        bool stop = false;
        if (m_stop.type == StopCriteria<T>::Type::MinError) {
            stop = losses.back() < m_stop.threshold;
        } else if (m_stop.type == StopCriteria<T>::Type::MinErrorChange && epoch > 0) {
            stop = std::abs(losses[epoch - 1] - losses[epoch]) < m_stop.threshold;
        }
        if (stop)
            break;
    }
    sync_all();
    if (!m_opts.record_history)
        snapshot_history();
    if (!m_opts.persistent_workspace)
        release_workspace();
    if (m_prof.enabled())
        m_prof.profile().wall_ns += Profiler::elapsed_ns(t_start);
    return losses;
}

// ---------------------------------------------------------------- inference

template <typename T>
std::vector<T> Network<T>::predict(const std::vector<T> &input_samples, unsigned num_samples, unsigned batch_size) {
    const auto t_start = Profiler::clock::now();
    const std::size_t n_in = m_layers.front().neurons, n_out = m_layers.back().neurons;
    if (num_samples == 0)
        return {};
    if (input_samples.size() != std::size_t(num_samples) * n_in)
        throw std::invalid_argument("cudann: predict input has " + std::to_string(input_samples.size()) + " values, expected " +
                                    std::to_string(std::size_t(num_samples) * n_in));
    check(cudaSetDevice(m_ordinal), "cudaSetDevice");
    const std::size_t N = num_samples;
    const std::size_t B = (batch_size == 0) ? N : std::min<std::size_t>(batch_size, N);
    ensure_workspace(B);

    std::vector<T> out(N * n_out);
    Buf X;
    detail::PinnedBuffer<T> stage_x, stage_out;
    struct Quiesce {
        Network &net;
        ~Quiesce() {
            for (auto s : net.m_streams)
                (void)cudaStreamSynchronize(s);
            (void)cudaGetLastError();
        }
    } quiesce{*this};
    cudaStream_t s0 = stream(0);
    const T *src = input_samples.data();
    X = Buf(N * n_in, m_kind);
    if (m_opts.pinned_host) {
        const auto t0 = Profiler::clock::now();
        stage_x = detail::PinnedBuffer<T>(N * n_in);
        std::memcpy(stage_x.data(), src, N * n_in * sizeof(T));
        m_prof.profile().h2d_ns += m_prof.enabled() ? Profiler::elapsed_ns(t0) : 0;
        src = stage_x.data();
        stage_out = detail::PinnedBuffer<T>(N * n_out);
    }
    T *xd = X.data();
    ev_list chain{submit(0, {}, Phase::H2D, [&](cudaStream_t st) {
        check(cudaMemcpyAsync(xd, src, N * n_in * sizeof(T), cudaMemcpyHostToDevice, st), "memcpy X");
    }, N * n_in * sizeof(T))};
    T *out_ptr = m_opts.pinned_host ? stage_out.data() : out.data();
    for (std::size_t start = 0; start < N; start += B) {
        const std::size_t cur = std::min(B, N - start);
        const T *in = X.data() + start * n_in;
        if (!m_opts.direct_input) {
            T *act0 = m_act[0].data();
            chain = {submit(0, chain, Phase::Other, [&](cudaStream_t st) {
                check(cudaMemcpyAsync(act0, in, cur * n_in * sizeof(T), cudaMemcpyDeviceToDevice, st), "memcpy D2D");
            })};
            in = act0;
        }
        for (std::size_t l = 0; l < m_L; ++l)
            chain = {forward_layer(l, (l == 0) ? in : m_act[l].data(), cur, 0, chain)};
        const T *last = m_act[m_L].data();
        T *dst = out_ptr + start * n_out;
        chain = {submit(0, chain, Phase::D2H, [&](cudaStream_t st) {
            check(cudaMemcpyAsync(dst, last, cur * n_out * sizeof(T), cudaMemcpyDeviceToHost, st), "memcpy out");
        }, cur * n_out * sizeof(T))};
    }
    sync_all();
    if (m_opts.pinned_host) {
        const auto t0 = Profiler::clock::now();
        std::memcpy(out.data(), stage_out.data(), N * n_out * sizeof(T));
        m_prof.profile().d2h_ns += m_prof.enabled() ? Profiler::elapsed_ns(t0) : 0;
    }
    if (!m_opts.persistent_workspace)
        release_workspace();
    if (m_prof.enabled())
        m_prof.profile().wall_ns += Profiler::elapsed_ns(t_start);
    (void)s0;
    return out;
}

} // namespace cudann
