// SPDX-License-Identifier: LGPL-3.0-only
// cudann - cuBLAS wrapper (column-major, device pointers), asynchronous on a stream.
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace cudann {

inline void check_blas(cublasStatus_t st, const char *what) {
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cudann: ") + what + " failed with cuBLAS status " + std::to_string(int(st)));
}

inline std::vector<std::string> compiled_blas_backends() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    return {"rocblas", "tiled"};
#else
    return {"cublas", "tiled"};
#endif
}

/// Hand-written CUDA BLAS (Options::blas = "tiled"): the 16x16 shared-memory
/// tiled GEMM of the lectures (same tile as syclnn's and ompnn's versions), a
/// one-thread-per-row GEMV and a single-block reduction for asum / nrm2.
/// Column-major; threadIdx.x runs along the rows (coalesced loads / stores).
namespace handwritten {
constexpr int TILE = 16;

template <typename T>
__global__ void gemm_k(bool ta, bool tb, int m, int n, int k, T alpha, const T *__restrict__ A, int lda,
                       const T *__restrict__ B, int ldb, T beta, T *C, int ldc) {
    __shared__ T As[TILE][TILE];
    __shared__ T Bs[TILE][TILE];
    const int li = threadIdx.x, lj = threadIdx.y;
    const int i = blockIdx.x * TILE + li, j = blockIdx.y * TILE + lj;
    T acc = T(0);
    for (int t = 0; t < k; t += TILE) {
        int p = t + lj;
        As[li][lj] = (i < m && p < k) ? (ta ? A[p + std::size_t(i) * lda] : A[i + std::size_t(p) * lda]) : T(0);
        p = t + li;
        Bs[li][lj] = (p < k && j < n) ? (tb ? B[j + std::size_t(p) * ldb] : B[p + std::size_t(j) * ldb]) : T(0);
        __syncthreads();
        for (int kk = 0; kk < TILE; ++kk)
            acc += As[li][kk] * Bs[kk][lj];
        __syncthreads();
    }
    if (i < m && j < n) {
        T *c = C + i + std::size_t(j) * ldc;
        *c = (beta == T(0)) ? alpha * acc : alpha * acc + beta * (*c);
    }
}

template <typename T>
__global__ void gemv_k(bool ta, int m, int n, T alpha, const T *__restrict__ A, int lda, const T *__restrict__ x, T beta, T *y) {
    const int rows = ta ? n : m, inner = ta ? m : n;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        T acc = T(0);
        for (int p = 0; p < inner; ++p)
            acc += (ta ? A[p + std::size_t(i) * lda] : A[i + std::size_t(p) * lda]) * x[p];
        y[i] = (beta == T(0)) ? alpha * acc : alpha * acc + beta * y[i];
    }
}

/// one block: result = sum |x| (abs) or sqrt(sum x^2)
template <typename T> __global__ void reduce_k(bool abs, int n, const T *__restrict__ x, T *result) {
    __shared__ double sdata[1024];
    double v = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const double e = static_cast<double>(x[i]);
        v += abs ? (e < 0 ? -e : e) : e * e;
    }
    sdata[threadIdx.x] = v;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        *result = static_cast<T>(abs ? sdata[0] : sqrt(sdata[0]));
}
} // namespace handwritten

/// Options::blas accepts "auto", "cublas" (and "rocblas"/"hipblas" in the
/// HIPified build) or "tiled" = the hand-written shared-memory kernels below.
inline bool use_handwritten(const std::string &name) { return name == "tiled" || name == "handwritten"; }
inline void check_blas_option(const std::string &name) {
    if (name.empty() || name == "auto" || name == "cublas" || use_handwritten(name))
        return;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    if (name == "rocblas" || name == "hipblas")
        return;
#endif
    throw std::invalid_argument("cudann: BLAS backend '" + name + "' is not available (this build uses " +
                                compiled_blas_backends().front() + "; use \"auto\")");
}

class Blas {
  public:
    explicit Blas(bool handwritten = false) : m_handwritten(handwritten) { check_blas(cublasCreate(&m_h), "cublasCreate"); }
    bool handwritten() const { return m_handwritten; }
    ~Blas() {
        if (m_h)
            (void)cublasDestroy(m_h);
    }
    Blas(const Blas &) = delete;
    Blas &operator=(const Blas &) = delete;
    Blas(Blas &&o) noexcept : m_handwritten(o.m_handwritten), m_h(o.m_h) { o.m_h = nullptr; }
    Blas &operator=(Blas &&o) noexcept {
        if (this != &o) {
            if (m_h)
                (void)cublasDestroy(m_h);
            m_h = o.m_h;
            m_handwritten = o.m_handwritten;
            o.m_h = nullptr;
        }
        return *this;
    }
    template <typename T>
    void hw_gemm(cudaStream_t s, bool ta, bool tb, int m, int n, int k, T alpha, const T *a, int lda, const T *b, int ldb,
                 T beta, T *c, int ldc) {
        const dim3 grid((m + handwritten::TILE - 1) / handwritten::TILE, (n + handwritten::TILE - 1) / handwritten::TILE);
        handwritten::gemm_k<T><<<grid, dim3(handwritten::TILE, handwritten::TILE), 0, s>>>(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    template <typename T>
    void hw_gemv(cudaStream_t s, bool ta, int m, int n, T alpha, const T *a, int lda, const T *x, T beta, T *y) {
        const int rows = ta ? n : m;
        handwritten::gemv_k<T><<<(rows + 255) / 256, 256, 0, s>>>(ta, m, n, alpha, a, lda, x, beta, y);
    }
    template <typename T> void hw_reduce(cudaStream_t s, bool abs, int n, const T *x, T *result) {
        handwritten::reduce_k<T><<<1, 1024, 0, s>>>(abs, n, x, result);
    }
    cublasHandle_t handle() const { return m_h; }

    static cublasOperation_t op(bool trans) { return trans ? CUBLAS_OP_T : CUBLAS_OP_N; }

    void gemm(cudaStream_t s, bool ta, bool tb, int m, int n, int k, float alpha, const float *a, int lda, const float *b,
              int ldb, float beta, float *c, int ldc) {
        if (m_handwritten) {
            hw_gemm(s, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            return;
        }
        use(s, CUBLAS_POINTER_MODE_HOST);
        check_blas(cublasSgemm(m_h, op(ta), op(tb), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc), "cublasSgemm");
    }
    void gemm(cudaStream_t s, bool ta, bool tb, int m, int n, int k, double alpha, const double *a, int lda,
              const double *b, int ldb, double beta, double *c, int ldc) {
        if (m_handwritten) {
            hw_gemm(s, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            return;
        }
        use(s, CUBLAS_POINTER_MODE_HOST);
        check_blas(cublasDgemm(m_h, op(ta), op(tb), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc), "cublasDgemm");
    }
    void gemv(cudaStream_t s, bool ta, int m, int n, float alpha, const float *a, int lda, const float *x, int incx,
              float beta, float *y, int incy) {
        if (m_handwritten) {
            hw_gemv(s, ta, m, n, alpha, a, lda, x, beta, y);
            return;
        }
        use(s, CUBLAS_POINTER_MODE_HOST);
        check_blas(cublasSgemv(m_h, op(ta), m, n, &alpha, a, lda, x, incx, &beta, y, incy), "cublasSgemv");
    }
    void gemv(cudaStream_t s, bool ta, int m, int n, double alpha, const double *a, int lda, const double *x, int incx,
              double beta, double *y, int incy) {
        if (m_handwritten) {
            hw_gemv(s, ta, m, n, alpha, a, lda, x, beta, y);
            return;
        }
        use(s, CUBLAS_POINTER_MODE_HOST);
        check_blas(cublasDgemv(m_h, op(ta), m, n, &alpha, a, lda, x, incx, &beta, y, incy), "cublasDgemv");
    }
    /// result is a *device* pointer (pointer mode DEVICE), so no synchronisation is needed.
    void asum(cudaStream_t s, int n, const float *x, int incx, float *result) {
        if (m_handwritten) {
            hw_reduce(s, true, n, x, result);
            return;
        }
        use(s, CUBLAS_POINTER_MODE_DEVICE);
        check_blas(cublasSasum(m_h, n, x, incx, result), "cublasSasum");
    }
    void asum(cudaStream_t s, int n, const double *x, int incx, double *result) {
        if (m_handwritten) {
            hw_reduce(s, true, n, x, result);
            return;
        }
        use(s, CUBLAS_POINTER_MODE_DEVICE);
        check_blas(cublasDasum(m_h, n, x, incx, result), "cublasDasum");
    }
    void nrm2(cudaStream_t s, int n, const float *x, int incx, float *result) {
        if (m_handwritten) {
            hw_reduce(s, false, n, x, result);
            return;
        }
        use(s, CUBLAS_POINTER_MODE_DEVICE);
        check_blas(cublasSnrm2(m_h, n, x, incx, result), "cublasSnrm2");
    }
    void nrm2(cudaStream_t s, int n, const double *x, int incx, double *result) {
        if (m_handwritten) {
            hw_reduce(s, false, n, x, result);
            return;
        }
        use(s, CUBLAS_POINTER_MODE_DEVICE);
        check_blas(cublasDnrm2(m_h, n, x, incx, result), "cublasDnrm2");
    }

  private:
    void use(cudaStream_t s, cublasPointerMode_t mode) {
        check_blas(cublasSetStream(m_h, s), "cublasSetStream");
        check_blas(cublasSetPointerMode(m_h, mode), "cublasSetPointerMode");
    }
    bool m_handwritten = false;
    cublasHandle_t m_h = nullptr;
};

} // namespace cudann
