// SPDX-License-Identifier: LGPL-3.0-only
// cudann - error checking, device enumeration and selection.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "cudann/config.hpp"

namespace cudann {

/// Throw std::runtime_error on a CUDA runtime error.
inline void check(cudaError_t err, const char *what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cudann: ") + what + ": " + cudaGetErrorString(err));
}
#define CUDANN_CHECK(call) ::cudann::check((call), #call)

struct DeviceInfo {
    int index = 0;
    std::string name;
    std::string vendor;
    std::string type = "gpu";
    std::string backend; ///< "cuda" or "hip"
    std::string platform;
    std::string driver;
    std::uint64_t global_mem_bytes = 0;
    unsigned compute_units = 0;
    bool fp64 = true;
    int cc_major = 0, cc_minor = 0;
};

inline std::string backend_name() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    return "hip";
#else
    return "cuda";
#endif
}

inline DeviceInfo describe(int ordinal) {
    cudaDeviceProp p{};
    check(cudaGetDeviceProperties(&p, ordinal), "cudaGetDeviceProperties");
    DeviceInfo d;
    d.index = ordinal;
    d.name = p.name;
    d.backend = backend_name();
    d.vendor = d.backend == "hip" ? "AMD" : "NVIDIA";
    int rt = 0, drv = 0;
    (void)cudaRuntimeGetVersion(&rt);
    (void)cudaDriverGetVersion(&drv);
    auto ver = [](int v) { return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10); };
    d.platform = (d.backend == "hip" ? "HIP runtime " : "CUDA runtime ") + ver(rt);
    d.driver = ver(drv);
    d.global_mem_bytes = static_cast<std::uint64_t>(p.totalGlobalMem);
    d.compute_units = static_cast<unsigned>(p.multiProcessorCount);
    d.cc_major = p.major;
    d.cc_minor = p.minor;
    return d;
}

inline int device_count() {
    int n = 0;
    const cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess) {
        (void)cudaGetLastError(); // clear
        return 0;
    }
    return n;
}

/// All GPUs visible to the runtime, in ordinal order.
inline std::vector<DeviceInfo> devices() {
    std::vector<DeviceInfo> out;
    const int n = device_count();
    for (int i = 0; i < n; ++i)
        out.push_back(describe(i));
    return out;
}

namespace detail {
inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}
inline bool is_number(const std::string &s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
}
} // namespace detail

/**
 * Resolve an Options::device string to a device ordinal:
 *   "default" | "gpu" | "cuda" | "hip"   -> device 0
 *   "index:N" | "N" | "cuda:N" | "hip:N" -> ordinal N
 *   "cpu"                                -> error (CUDA has no CPU device)
 *   anything else                        -> case-insensitive substring of the device name
 */
inline int select_device(const std::string &spec) {
    const std::string s = detail::lower(spec);
    const int n = device_count();
    auto no_device = [&](const std::string &what) {
        std::string names;
        for (int i = 0; i < n; ++i)
            names += "\n  index:" + std::to_string(i) + "  " + backend_name() + ":gpu  " + describe(i).name;
        return std::invalid_argument("no " + backend_name() + " device matches '" + spec + "' (" + what + "); available:" +
                                     (n ? names : std::string(" none")));
    };
    if (n == 0)
        throw no_device("no device present");
    if (s.empty() || s == "default" || s == "auto" || s == "gpu" || s == "cuda" || s == "hip")
        return 0;
    if (s == "cpu" || s == "accelerator" || s == "host")
        throw no_device("no such device type");
    std::string key = s, arg;
    const auto colon = s.find(':');
    if (colon != std::string::npos) {
        key = s.substr(0, colon);
        arg = s.substr(colon + 1);
    }
    if (detail::is_number(s) || ((key == "index" || key == "cuda" || key == "hip") && detail::is_number(arg))) {
        const int i = std::stoi(detail::is_number(s) ? s : arg);
        if (i < 0 || i >= n)
            throw no_device("index out of range");
        return i;
    }
    if (colon != std::string::npos && (key == "cuda" || key == "hip") && arg == "gpu")
        return 0;
    for (int i = 0; i < n; ++i)
        if (detail::lower(describe(i).name).find(s) != std::string::npos)
            return i;
    throw no_device("no name match");
}

} // namespace cudann
