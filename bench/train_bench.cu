// SPDX-License-Identifier: LGPL-3.0-only
// train_bench: minimal C++ driver for Network<T> (no Python in the loop).
//
//   train_bench --layers 784,1024,1024,10 --samples 8192 --batch 256 --epochs 5 \
//               --dtype float --device gpu --profile --json
//
// Prints one JSON object with the wall-clock time per epoch and the Profile, so
// that fnn-bench can check that the Python binding adds no measurable overhead.
// `--check` additionally verifies that the loss decreased (ctest smoke test).

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cudann/network.hpp"

using namespace cudann;

namespace {

struct Args {
    std::vector<unsigned> layers{8, 16, 4};
    unsigned samples = 256, batch = 32, epochs = 3, seed = 1;
    std::string dtype = "double", device = "default", blas = "auto", hidden = "tanh", output = "sigmoid";
    bool profile = false, json = false, check = false, infer = false;
    unsigned workgroup = 0;
};

ActivationType parse_act(const std::string &s) {
    if (s == "disabled" || s == "linear") return ActivationType::Disabled;
    if (s == "sigmoid") return ActivationType::Sigmoid;
    if (s == "tanh") return ActivationType::Tanh;
    if (s == "relu") return ActivationType::ReLU;
    if (s == "leaky_relu") return ActivationType::LeakyReLU;
    if (s == "elu") return ActivationType::ELU;
    throw std::invalid_argument("unknown activation " + s);
}

Args parse(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + k);
            return argv[++i];
        };
        if (k == "--layers") {
            a.layers.clear();
            std::stringstream ss(next());
            std::string tok;
            while (std::getline(ss, tok, ',')) a.layers.push_back(static_cast<unsigned>(std::stoul(tok)));
        } else if (k == "--samples") a.samples = std::stoul(next());
        else if (k == "--batch") a.batch = std::stoul(next());
        else if (k == "--epochs") a.epochs = std::stoul(next());
        else if (k == "--seed") a.seed = std::stoul(next());
        else if (k == "--dtype") a.dtype = next();
        else if (k == "--device") a.device = next();
        else if (k == "--blas") a.blas = next();
        else if (k == "--hidden") a.hidden = next();
        else if (k == "--output") a.output = next();
        else if (k == "--workgroup") a.workgroup = std::stoul(next());
        else if (k == "--profile") a.profile = true;
        else if (k == "--json") a.json = true;
        else if (k == "--check") a.check = true;
        else if (k == "--infer") a.infer = true;
        else if (k == "--help") {
            std::puts("train_bench --layers a,b,c --samples N --batch B --epochs E --dtype float|double --device D "
                      "--blas auto --hidden ACT --output ACT --workgroup W "
                      "--profile --json --check --infer");
            std::exit(0);
        } else throw std::invalid_argument("unknown argument " + k);
    }
    return a;
}

template <typename T> int run(const Args &a) {
    std::vector<LayerDescription> layers;
    for (std::size_t i = 0; i < a.layers.size(); ++i)
        layers.push_back({a.layers[i], i == 0 ? ActivationType::Disabled
                                             : (i + 1 == a.layers.size() ? parse_act(a.output) : parse_act(a.hidden))});
    const unsigned n_in = a.layers.front(), n_out = a.layers.back();
    std::mt19937 rng(a.seed);
    std::uniform_real_distribution<T> dist(T(-1), T(1));
    std::vector<T> X(std::size_t(a.samples) * n_in), Y(std::size_t(a.samples) * n_out);
    for (auto &x : X) x = dist(rng);
    for (std::size_t n = 0; n < a.samples; ++n)
        for (unsigned k = 0; k < n_out; ++k) {
            T s = T(0);
            for (unsigned j = 0; j < n_in; ++j) s += X[n * n_in + j] * T((j + k) % 7 - 3) / T(n_in);
            Y[n * n_out + k] = T(0.5) * (T(1) + std::tanh(s));
        }
    Options o;
    o.device = a.device;
    o.blas = a.blas;
    o.profile = a.profile;
    o.workgroup_size = a.workgroup;
    Network<T> net(layers, T(0.01), Regularization<T>{}, BackPropagation::Standard,
                   AdaptiveLearningRate<T>(AdaptiveLearningRate<T>::Strategy::Adam), StopCriteria<T>{}, MomentumConfig<T>{},
                   a.seed, o);
    using clock = std::chrono::steady_clock;
    std::vector<double> epoch_s;
    std::vector<T> losses;
    double total_s = 0;
    if (!a.infer) {
        for (unsigned e = 0; e < a.epochs; ++e) {
            const auto t0 = clock::now();
            auto l = net.train(X, Y, a.samples, a.batch, 1);
            const double s = std::chrono::duration<double>(clock::now() - t0).count();
            epoch_s.push_back(s);
            total_s += s;
            losses.push_back(l.front());
        }
    } else {
        for (unsigned e = 0; e < a.epochs; ++e) {
            const auto t0 = clock::now();
            auto out = net.predict(X, a.samples, a.batch);
            const double s = std::chrono::duration<double>(clock::now() - t0).count();
            epoch_s.push_back(s);
            total_s += s;
            (void)out;
        }
    }
    const Profile &p = net.profile();
    if (a.json) {
        std::printf("{\"device\":\"%s\",\"blas\":\"%s\",\"dtype\":\"%s\",\"samples\":%u,\"batch\":%u,\"epochs\":%u,"
                    "\"mode\":\"%s\",\"epoch_s\":[",
                    net.device_name().c_str(), net.blas_backend().c_str(), a.dtype.c_str(), a.samples, a.batch, a.epochs,
                    a.infer ? "infer" : "train");
        for (std::size_t i = 0; i < epoch_s.size(); ++i) std::printf("%s%.6f", i ? "," : "", epoch_s[i]);
        std::printf("],\"losses\":[");
        for (std::size_t i = 0; i < losses.size(); ++i) std::printf("%s%.9g", i ? "," : "", double(losses[i]));
        std::printf("],\"profile\":{\"h2d_ns\":%llu,\"d2h_ns\":%llu,\"gemm_ns\":%llu,\"act_ns\":%llu,\"delta_ns\":%llu,"
                    "\"biasgrad_ns\":%llu,\"update_ns\":%llu,\"loss_ns\":%llu,\"reg_ns\":%llu,\"other_ns\":%llu,"
                    "\"wall_ns\":%llu,\"wait_ns\":%llu,\"launches\":%llu,\"unprofiled\":%llu}}\n",
                    (unsigned long long)p.h2d_ns, (unsigned long long)p.d2h_ns, (unsigned long long)p.gemm_ns,
                    (unsigned long long)p.act_ns, (unsigned long long)p.delta_ns, (unsigned long long)p.biasgrad_ns,
                    (unsigned long long)p.update_ns, (unsigned long long)p.loss_ns, (unsigned long long)p.reg_ns,
                    (unsigned long long)p.other_ns, (unsigned long long)p.wall_ns, (unsigned long long)p.wait_ns,
                    (unsigned long long)p.launches, (unsigned long long)p.unprofiled);
    } else {
        std::printf("device: %s  blas: %s  dtype: %s\n", net.device_name().c_str(), net.blas_backend().c_str(),
                    a.dtype.c_str());
        for (std::size_t i = 0; i < epoch_s.size(); ++i)
            std::printf("epoch %zu: %.4f s%s%s\n", i, epoch_s[i], losses.empty() ? "" : "  loss ",
                        losses.empty() ? "" : std::to_string(double(losses[i])).c_str());
        std::printf("total %.4f s (%.1f samples/s)\n", total_s, a.samples * double(a.epochs) / total_s);
        if (a.profile)
            std::printf("profile: gemm %.3f ms, act %.3f ms, delta %.3f ms, biasgrad %.3f ms, update %.3f ms, loss %.3f ms, "
                        "h2d %.3f ms, d2h %.3f ms, wait %.3f ms, launches %llu\n",
                        p.gemm_ns / 1e6, p.act_ns / 1e6, p.delta_ns / 1e6, p.biasgrad_ns / 1e6, p.update_ns / 1e6,
                        p.loss_ns / 1e6, p.h2d_ns / 1e6, p.d2h_ns / 1e6, p.wait_ns / 1e6, (unsigned long long)p.launches);
    }
    if (a.check && !a.infer) {
        if (losses.empty() || !(losses.back() < losses.front())) {
            std::fprintf(stderr, "check failed: loss did not decrease (%g -> %g)\n", double(losses.front()),
                         double(losses.back()));
            return 1;
        }
        for (auto l : losses)
            if (!(l == l)) {
                std::fprintf(stderr, "check failed: NaN loss\n");
                return 1;
            }
    }
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    try {
        Args a = parse(argc, argv);
        if (a.dtype == "float") return run<float>(a);
        if (a.dtype == "double") return run<double>(a);
        throw std::invalid_argument("dtype must be float or double");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "train_bench: %s\n", e.what());
        return 2;
    }
}
