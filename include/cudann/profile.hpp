// SPDX-License-Identifier: LGPL-3.0-only
// cudann - per-phase profiler fed by cudaEvent pairs recorded around each launch.
#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <vector>

#include "cudann/config.hpp"
#include "cudann/device.hpp"

namespace cudann {

enum class Phase { H2D, D2H, Gemm, Act, Delta, BiasGrad, Update, Loss, Reg, Other };

/**
 * Records (start, end) event pairs on the stream of every launch and folds the
 * elapsed times into a Profile at synchronisation points.  Disabled: launches
 * run untouched.  Suspended (stream capture for CUDA Graphs): event records
 * would become graph nodes whose timing is not reliable, so nothing is recorded.
 */
class Profiler {
  public:
    explicit Profiler(bool enabled = false) : m_enabled(enabled) {}
    Profiler(const Profiler &) = delete;
    Profiler &operator=(const Profiler &) = delete;
    Profiler(Profiler &&o) noexcept { *this = std::move(o); }
    Profiler &operator=(Profiler &&o) noexcept {
        if (this != &o) {
            destroy();
            m_enabled = o.m_enabled;
            m_suspended = o.m_suspended;
            m_profile = std::move(o.m_profile);
            m_pending = std::move(o.m_pending);
            m_pool = std::move(o.m_pool);
            m_pos = o.m_pos;
            o.m_pool.clear();
            o.m_pending.clear();
        }
        return *this;
    }
    ~Profiler() { destroy(); }

    bool enabled() const { return m_enabled; }
    void suspend(bool s) { m_suspended = s; }
    Profile &profile() { return m_profile; }
    const Profile &profile() const { return m_profile; }
    void reset() {
        m_profile.reset();
        m_pending.clear();
        m_pos = 0;
    }

    /// Run `launch()` on `stream`, bracketed by timing events when enabled.
    template <typename F> void record(Phase phase, cudaStream_t stream, F &&launch, std::uint64_t bytes = 0) {
        if (!m_enabled || m_suspended) {
            launch();
            return;
        }
        ++m_profile.launches;
        if (phase == Phase::H2D)
            m_profile.bytes_h2d += bytes;
        else if (phase == Phase::D2H)
            m_profile.bytes_d2h += bytes;
        cudaEvent_t a = take(), b = take();
        (void)cudaEventRecord(a, stream);
        launch();
        (void)cudaEventRecord(b, stream);
        m_pending.push_back({phase, a, b});
    }

    /// Fold every pending pair into the profile (call after a full synchronisation).
    void flush() {
        if (!m_enabled)
            return;
        for (auto &r : m_pending) {
            float ms = 0.f;
            if (cudaEventElapsedTime(&ms, r.start, r.end) != cudaSuccess) {
                (void)cudaGetLastError();
                ++m_profile.unprofiled;
                continue;
            }
            slot(r.phase) += static_cast<std::uint64_t>(ms * 1e6f);
        }
        m_pending.clear();
        m_pos = 0;
    }

    template <typename F> void timed_wait(F &&f) {
        const auto t0 = clock::now();
        f();
        m_profile.wait_ns += elapsed_ns(t0);
    }

    using clock = std::chrono::steady_clock;
    static std::uint64_t elapsed_ns(clock::time_point since) {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - since).count());
    }

  private:
    struct Rec {
        Phase phase;
        cudaEvent_t start, end;
    };
    cudaEvent_t take() {
        if (m_pos == m_pool.size()) {
            cudaEvent_t e;
            check(cudaEventCreate(&e), "cudaEventCreate");
            m_pool.push_back(e);
        }
        return m_pool[m_pos++];
    }
    void destroy() noexcept {
        for (auto e : m_pool)
            (void)cudaEventDestroy(e);
        m_pool.clear();
        m_pending.clear();
    }
    std::uint64_t &slot(Phase p) {
        switch (p) {
        case Phase::H2D: return m_profile.h2d_ns;
        case Phase::D2H: return m_profile.d2h_ns;
        case Phase::Gemm: return m_profile.gemm_ns;
        case Phase::Act: return m_profile.act_ns;
        case Phase::Delta: return m_profile.delta_ns;
        case Phase::BiasGrad: return m_profile.biasgrad_ns;
        case Phase::Update: return m_profile.update_ns;
        case Phase::Loss: return m_profile.loss_ns;
        case Phase::Reg: return m_profile.reg_ns;
        case Phase::Other:
        default: return m_profile.other_ns;
        }
    }

    bool m_enabled = false;
    bool m_suspended = false;
    Profile m_profile;
    std::vector<Rec> m_pending;
    std::vector<cudaEvent_t> m_pool;
    std::size_t m_pos = 0;
};

} // namespace cudann
