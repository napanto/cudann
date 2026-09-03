// SPDX-License-Identifier: LGPL-3.0-only
// cudann - activation functions and derivatives, callable from CUDA kernels.
// Same definitions as syclnn/activations.hpp (sycl::exp -> expf/exp, ...).
#pragma once

#include <cuda_runtime.h>

#include <type_traits>

#include "cudann/config.hpp"

namespace cudann {

#define CUDANN_HD __host__ __device__

template <typename T> CUDANN_HD inline T dev_exp(T x) { return exp(x); }
template <> CUDANN_HD inline float dev_exp<float>(float x) { return expf(x); }
template <typename T> CUDANN_HD inline T dev_tanh(T x) { return tanh(x); }
template <> CUDANN_HD inline float dev_tanh<float>(float x) { return tanhf(x); }
template <typename T> CUDANN_HD inline T dev_sqrt(T x) { return sqrt(x); }
template <> CUDANN_HD inline float dev_sqrt<float>(float x) { return sqrtf(x); }
template <typename T> CUDANN_HD inline T dev_pow(T x, T y) { return pow(x, y); }
template <> CUDANN_HD inline float dev_pow<float>(float x, float y) { return powf(x, y); }
template <typename T> CUDANN_HD inline T dev_fmax(T a, T b) { return fmax(a, b); }
template <> CUDANN_HD inline float dev_fmax<float>(float a, float b) { return fmaxf(a, b); }

template <typename T> CUDANN_HD constexpr T leaky_relu_alpha() { return T(0.01); }
template <typename T> CUDANN_HD constexpr T elu_alpha() { return T(1); }

/// sgn(x) in {-1, 0, 1}
template <typename T> CUDANN_HD inline T sgn(T x) { return T(0) < x ? T(1) : (x < T(0) ? T(-1) : T(0)); }

/// Compile-time activation dispatch: Act<A>::f / ::df / ::df_out.
template <ActivationType A> struct Act;

template <> struct Act<ActivationType::Disabled> {
    template <typename T> CUDANN_HD static inline T f(T x) { return x; }
    template <typename T> CUDANN_HD static inline T df(T) { return T(1); }
    template <typename T> CUDANN_HD static inline T df_out(T, T) { return T(1); }
};
template <> struct Act<ActivationType::Sigmoid> {
    template <typename T> CUDANN_HD static inline T f(T x) { return T(1) / (T(1) + dev_exp(-x)); }
    template <typename T> CUDANN_HD static inline T df(T x) {
        T s = f(x);
        return s * (T(1) - s);
    }
    template <typename T> CUDANN_HD static inline T df_out(T o, T) { return o * (T(1) - o); }
};
template <> struct Act<ActivationType::Tanh> {
    template <typename T> CUDANN_HD static inline T f(T x) { return dev_tanh(x); }
    template <typename T> CUDANN_HD static inline T df(T x) {
        T t = dev_tanh(x);
        return T(1) - t * t;
    }
    template <typename T> CUDANN_HD static inline T df_out(T o, T) { return T(1) - o * o; }
};
template <> struct Act<ActivationType::ReLU> {
    template <typename T> CUDANN_HD static inline T f(T x) { return dev_fmax(T(0), x); }
    template <typename T> CUDANN_HD static inline T df(T x) { return x > T(0) ? T(1) : T(0); }
    template <typename T> CUDANN_HD static inline T df_out(T, T x) { return df(x); }
};
template <> struct Act<ActivationType::LeakyReLU> {
    template <typename T> CUDANN_HD static inline T f(T x) { return x > T(0) ? x : leaky_relu_alpha<T>() * x; }
    template <typename T> CUDANN_HD static inline T df(T x) { return x > T(0) ? T(1) : leaky_relu_alpha<T>(); }
    template <typename T> CUDANN_HD static inline T df_out(T, T x) { return df(x); }
};
template <> struct Act<ActivationType::ELU> {
    template <typename T> CUDANN_HD static inline T f(T x) { return x > T(0) ? x : elu_alpha<T>() * (dev_exp(x) - T(1)); }
    template <typename T> CUDANN_HD static inline T df(T x) { return x > T(0) ? T(1) : elu_alpha<T>() * dev_exp(x); }
    template <typename T> CUDANN_HD static inline T df_out(T o, T x) { return x > T(0) ? T(1) : o + elu_alpha<T>(); }
};

/// Run-time dispatch (the "switch inside the kernel" ablation).
template <typename T> CUDANN_HD inline T activate(ActivationType a, T x) {
    switch (a) {
    case ActivationType::Disabled: return Act<ActivationType::Disabled>::f(x);
    case ActivationType::Sigmoid: return Act<ActivationType::Sigmoid>::f(x);
    case ActivationType::Tanh: return Act<ActivationType::Tanh>::f(x);
    case ActivationType::ReLU: return Act<ActivationType::ReLU>::f(x);
    case ActivationType::LeakyReLU: return Act<ActivationType::LeakyReLU>::f(x);
    case ActivationType::ELU: return Act<ActivationType::ELU>::f(x);
    }
    return x;
}
template <typename T> CUDANN_HD inline T derivative(ActivationType a, T x) {
    switch (a) {
    case ActivationType::Disabled: return Act<ActivationType::Disabled>::df(x);
    case ActivationType::Sigmoid: return Act<ActivationType::Sigmoid>::df(x);
    case ActivationType::Tanh: return Act<ActivationType::Tanh>::df(x);
    case ActivationType::ReLU: return Act<ActivationType::ReLU>::df(x);
    case ActivationType::LeakyReLU: return Act<ActivationType::LeakyReLU>::df(x);
    case ActivationType::ELU: return Act<ActivationType::ELU>::df(x);
    }
    return T(1);
}
template <typename T> CUDANN_HD inline T derivative_from_output(ActivationType a, T o, T x) {
    switch (a) {
    case ActivationType::Disabled: return Act<ActivationType::Disabled>::df_out(o, x);
    case ActivationType::Sigmoid: return Act<ActivationType::Sigmoid>::df_out(o, x);
    case ActivationType::Tanh: return Act<ActivationType::Tanh>::df_out(o, x);
    case ActivationType::ReLU: return Act<ActivationType::ReLU>::df_out(o, x);
    case ActivationType::LeakyReLU: return Act<ActivationType::LeakyReLU>::df_out(o, x);
    case ActivationType::ELU: return Act<ActivationType::ELU>::df_out(o, x);
    }
    return T(1);
}

/// Host-side: call fn(std::integral_constant<ActivationType, A>{}) for the run-time value a.
template <typename Fn> inline decltype(auto) dispatch_activation(ActivationType a, Fn &&fn) {
    switch (a) {
    case ActivationType::Sigmoid: return fn(std::integral_constant<ActivationType, ActivationType::Sigmoid>{});
    case ActivationType::Tanh: return fn(std::integral_constant<ActivationType, ActivationType::Tanh>{});
    case ActivationType::ReLU: return fn(std::integral_constant<ActivationType, ActivationType::ReLU>{});
    case ActivationType::LeakyReLU: return fn(std::integral_constant<ActivationType, ActivationType::LeakyReLU>{});
    case ActivationType::ELU: return fn(std::integral_constant<ActivationType, ActivationType::ELU>{});
    case ActivationType::Disabled:
    default: return fn(std::integral_constant<ActivationType, ActivationType::Disabled>{});
    }
}

} // namespace cudann
