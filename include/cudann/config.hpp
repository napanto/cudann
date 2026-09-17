// SPDX-License-Identifier: LGPL-3.0-only
// cudann - CUDA feed-forward neural network library (1:1 port of syclnn)
// Copyright (C) 2025-2026 Antonio Napolitano, Suqi Chen
//
// config.hpp: the public configuration types.  This file is kept *identical*
// (apart from the namespace) in syclnn, cudann and ompnn so that the three
// libraries expose the same API and the same benchmark/ablation switches.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#define CUDANN_VERSION_MAJOR 0
#define CUDANN_VERSION_MINOR 1
#define CUDANN_VERSION_PATCH 0
#define CUDANN_VERSION_STRING "0.1.0"

namespace cudann {

// ---------------------------------------------------------------------------
// Network description
// ---------------------------------------------------------------------------

/// Activation function of a layer.
enum class ActivationType {
    Disabled,  ///< identity, f(x) = x
    Sigmoid,   ///< 1 / (1 + exp(-x))
    Tanh,      ///< tanh(x)
    ReLU,      ///< max(0, x)
    LeakyReLU, ///< x > 0 ? x : 0.01 x
    ELU        ///< x > 0 ? x : exp(x) - 1
};

/// One layer: its width and the activation applied to its net input.
struct LayerDescription {
    unsigned neurons;                                    ///< number of units
    ActivationType activation = ActivationType::Disabled; ///< ignored for the input layer
};

/// Weight-decay configuration.  Only the coefficient(s) of the selected type
/// are used: L1 uses lambda1, L2 uses lambda2, ElasticNet uses both.
template <typename T> struct Regularization {
    enum class Type { Disabled, L1, L2, ElasticNet } type;
    T lambda1{};
    T lambda2{};
    Regularization(Type t = Type::Disabled, T l1 = T(0), T l2 = T(0)) : type(t), lambda1(l1), lambda2(l2) {}
    bool uses_l1() const { return type == Type::L1 || type == Type::ElasticNet; }
    bool uses_l2() const { return type == Type::L2 || type == Type::ElasticNet; }
};

/// Early-stopping rule, evaluated after every epoch.  max_epochs is always a
/// hard cap on top of it.
template <typename T> struct StopCriteria {
    enum class Type { MaxEpochs, MinError, MinErrorChange } type;
    T threshold{};
    StopCriteria(Type t = Type::MaxEpochs, T th = T(1e-4)) : type(t), threshold(th) {}
};

/// Classical momentum (only honoured by the Constant and LinearDecay strategies).
template <typename T> struct MomentumConfig {
    enum class Type { Disabled, Classical } type;
    T momentum_rate{};
    MomentumConfig(Type t = Type::Disabled, T r = T(0)) : type(t), momentum_rate(r) {}
};

/// Learning-rate strategy.
template <typename T> struct AdaptiveLearningRate {
    enum class Strategy { Constant, LinearDecay, AdaGrad, RMSProp, Adam } strategy;
    T epsilon{};  ///< AdaGrad / RMSProp / Adam
    T beta1{};    ///< Adam beta1, RMSProp rho
    T beta2{};    ///< Adam beta2
    T final_lr{}; ///< LinearDecay: learning rate reached at the last epoch
    AdaptiveLearningRate(Strategy s = Strategy::Constant, T eps = T(1e-8), T b1 = T(0.9), T b2 = T(0.999),
                         T flr = T(1e-4))
        : strategy(s), epsilon(eps), beta1(b1), beta2(b2), final_lr(flr) {}
};

/// Back-propagation flavour (only Standard exists; kept for API stability).
enum class BackPropagation { Standard };

// ---------------------------------------------------------------------------
// Runtime options (device selection, instrumentation, ablation switches)
// ---------------------------------------------------------------------------

/// Memory placement of device tensors.
enum class MemoryKind {
    Device, ///< malloc_device / cudaMalloc + explicit copies (default)
    Shared, ///< malloc_shared / cudaMallocManaged (page migration)
    Host    ///< malloc_host / pinned zero-copy host memory
};

/// Ordering of the work submitted to the device.
enum class QueueOrder {
    OutOfOrder, ///< event-driven DAG (SYCL default; several CUDA streams)
    InOrder,    ///< one in-order queue / one CUDA stream
    Graph       ///< captured graph replayed per batch (CUDA Graphs; ignored elsewhere)
};

/**
 * Every knob that changes *how* the network runs but not *what* it computes.
 * The defaults are the optimised 0.2 configuration; each switch can be turned
 * back to the 0.1 behaviour so that the benchmark can measure its effect.
 * Backends silently ignore switches that do not apply to them and report the
 * effective configuration through Network::options().
 */
struct Options {
    /// Device selector: "default", "cpu", "gpu", "accelerator", "index:N",
    /// "<backend>:N" (opencl:0, cuda:0, hip:0, level_zero:0, ...), or a substring
    /// of the device name.
    std::string device = "default";
    /// BLAS backend for the GEMM/GEMV/asum/nrm2 calls: "auto" (run-time dispatch by
    /// device type), or one of the compiled-in backends: "mklcpu", "netlib",
    /// "generic", "cublas", "rocblas" (SYCL) / "openblas", "mkl", "cublas",
    /// "rocblas" (OpenMP) / "cublas" (CUDA).
    std::string blas = "auto";

    bool profile = false;        ///< collect per-phase device timings (Profile)
    bool record_history = false; ///< snapshot weights after every epoch (0.1 behaviour: always)
    bool shuffle = false;        ///< reshuffle the samples every epoch (std::mt19937 + Fisher-Yates)
    unsigned shuffle_seed = 1;

    MemoryKind memory = MemoryKind::Device;    ///< 0.1: Shared
    QueueOrder queue = QueueOrder::OutOfOrder; ///< SYCL: queue property; CUDA: streams / graphs
    unsigned streams = 4;                      ///< CUDA: number of streams when queue == OutOfOrder
    bool pinned_host = true;                   ///< stage host<->device copies through pinned memory

    bool loss_reduction = true;         ///< loss via reduction (0.1: one atomic per element)
    bool bias_gemv = true;              ///< bias gradient via GEMV (0.1: serial loop per neuron)
    bool direct_input = true;           ///< first GEMM reads the dataset slice (0.1: memcpy per batch)
    bool fine_deps = true;              ///< fine-grained event graph (0.1: coarse phase barriers)
    bool join_kernels = false;          ///< 0.1: empty "join" kernels instead of multi-event dependencies
    bool specialized_kernels = true;    ///< kernels templated on ActivationType (0.1: switch in kernel)
    bool derivative_from_output = true; ///< sigma'/tanh'/elu' from the stored output (0.1: recompute)
    bool host_adam_correction = true;   ///< Adam bias-correction factors computed once per step on host
    unsigned workgroup_size = 0;        ///< nd_range work-group / CUDA block size for element-wise kernels (0: runtime default)
    bool persistent_workspace = true;   ///< keep batch temporaries across train()/predict() calls
    bool fast_math = false;             ///< allow the backend's fast-math intrinsics (breaks parity)
    bool sync_ops = false;              ///< wait for every launch (ompnn's synchronous execution model; ablation)
    std::string blas_queue = "auto";    ///< accepted for parity with syclnn (no meaning here)
    unsigned sync_every = 0;            ///< wait for the queue every N batches (0 = automatic: 4 on CPU devices, never on GPUs);
                                        ///< bounds the outstanding commands, which the OpenCL CPU runtime handles superlinearly
};

// ---------------------------------------------------------------------------
// Instrumentation
// ---------------------------------------------------------------------------

/**
 * Accumulated timings of one Network, filled only when Options::profile is on.
 * Device times come from event profiling (SYCL, CUDA) or from host timers around
 * synchronous regions (OpenMP); all values are nanoseconds.
 */
struct Profile {
    std::uint64_t h2d_ns = 0;      ///< host -> device copies
    std::uint64_t d2h_ns = 0;      ///< device -> host copies
    std::uint64_t gemm_ns = 0;     ///< BLAS GEMM (forward, hidden delta, weight gradient)
    std::uint64_t act_ns = 0;      ///< bias + activation kernels
    std::uint64_t delta_ns = 0;    ///< hidden-layer delta (f') kernels
    std::uint64_t biasgrad_ns = 0; ///< bias-gradient kernels / GEMV
    std::uint64_t update_ns = 0;   ///< parameter-update kernels
    std::uint64_t loss_ns = 0;     ///< output-layer delta + loss kernel
    std::uint64_t reg_ns = 0;      ///< regularisation penalty (asum/nrm2 + combine)
    std::uint64_t other_ns = 0;    ///< fills, gathers, joins
    std::uint64_t wall_ns = 0;     ///< host wall-clock inside train()/predict()
    std::uint64_t wait_ns = 0;     ///< host time blocked in synchronisation calls
    std::uint64_t launches = 0;    ///< kernels and memory operations submitted
    std::uint64_t bytes_h2d = 0;
    std::uint64_t bytes_d2h = 0;
    std::uint64_t epochs = 0;
    std::uint64_t batches = 0;
    std::uint64_t unprofiled = 0; ///< events whose profiling info was unavailable
    std::vector<std::uint64_t> epoch_wall_ns;
    std::vector<std::uint64_t> predict_wall_ns; ///< host wall-clock of every predict() call (ns)

    std::uint64_t device_total_ns() const {
        return h2d_ns + d2h_ns + gemm_ns + act_ns + delta_ns + biasgrad_ns + update_ns + loss_ns + reg_ns + other_ns;
    }
    void reset() { *this = Profile{}; }
};

} // namespace cudann
