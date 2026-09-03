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
    return {"rocblas"};
#else
    return {"cublas"};
#endif
}

/// The only BLAS of this library; Options::blas accepts "auto", "cublas" (and
/// "rocblas"/"hipblas" in the HIPified build).
inline void check_blas_option(const std::string &name) {
    if (name.empty() || name == "auto" || name == "cublas")
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
    Blas() { check_blas(cublasCreate(&m_h), "cublasCreate"); }
    ~Blas() {
        if (m_h)
            (void)cublasDestroy(m_h);
    }
    Blas(const Blas &) = delete;
    Blas &operator=(const Blas &) = delete;
    Blas(Blas &&o) noexcept : m_h(o.m_h) { o.m_h = nullptr; }
    Blas &operator=(Blas &&o) noexcept {
        if (this != &o) {
            if (m_h)
                (void)cublasDestroy(m_h);
            m_h = o.m_h;
            o.m_h = nullptr;
        }
        return *this;
    }
    cublasHandle_t handle() const { return m_h; }

    static cublasOperation_t op(bool trans) { return trans ? CUBLAS_OP_T : CUBLAS_OP_N; }

    void gemm(cudaStream_t s, bool ta, bool tb, int m, int n, int k, float alpha, const float *a, int lda, const float *b,
              int ldb, float beta, float *c, int ldc) {
        use(s, CUBLAS_POINTER_MODE_HOST);
        check_blas(cublasSgemm(m_h, op(ta), op(tb), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc), "cublasSgemm");
    }
    void gemm(cudaStream_t s, bool ta, bool tb, int m, int n, int k, double alpha, const double *a, int lda,
              const double *b, int ldb, double beta, double *c, int ldc) {
        use(s, CUBLAS_POINTER_MODE_HOST);
        check_blas(cublasDgemm(m_h, op(ta), op(tb), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc), "cublasDgemm");
    }
    void gemv(cudaStream_t s, bool ta, int m, int n, float alpha, const float *a, int lda, const float *x, int incx,
              float beta, float *y, int incy) {
        use(s, CUBLAS_POINTER_MODE_HOST);
        check_blas(cublasSgemv(m_h, op(ta), m, n, &alpha, a, lda, x, incx, &beta, y, incy), "cublasSgemv");
    }
    void gemv(cudaStream_t s, bool ta, int m, int n, double alpha, const double *a, int lda, const double *x, int incx,
              double beta, double *y, int incy) {
        use(s, CUBLAS_POINTER_MODE_HOST);
        check_blas(cublasDgemv(m_h, op(ta), m, n, &alpha, a, lda, x, incx, &beta, y, incy), "cublasDgemv");
    }
    /// result is a *device* pointer (pointer mode DEVICE), so no synchronisation is needed.
    void asum(cudaStream_t s, int n, const float *x, int incx, float *result) {
        use(s, CUBLAS_POINTER_MODE_DEVICE);
        check_blas(cublasSasum(m_h, n, x, incx, result), "cublasSasum");
    }
    void asum(cudaStream_t s, int n, const double *x, int incx, double *result) {
        use(s, CUBLAS_POINTER_MODE_DEVICE);
        check_blas(cublasDasum(m_h, n, x, incx, result), "cublasDasum");
    }
    void nrm2(cudaStream_t s, int n, const float *x, int incx, float *result) {
        use(s, CUBLAS_POINTER_MODE_DEVICE);
        check_blas(cublasSnrm2(m_h, n, x, incx, result), "cublasSnrm2");
    }
    void nrm2(cudaStream_t s, int n, const double *x, int incx, double *result) {
        use(s, CUBLAS_POINTER_MODE_DEVICE);
        check_blas(cublasDnrm2(m_h, n, x, incx, result), "cublasDnrm2");
    }

  private:
    void use(cudaStream_t s, cublasPointerMode_t mode) {
        check_blas(cublasSetStream(m_h, s), "cublasSetStream");
        check_blas(cublasSetPointerMode(m_h, mode), "cublasSetPointerMode");
    }
    cublasHandle_t m_h = nullptr;
};

} // namespace cudann
