// SPDX-License-Identifier: LGPL-3.0-only
// cudann - pybind11 module `_cudann`.
//
// The binding layer is identical in syclnn, cudann and ompnn apart from the
// namespace / module name, so that fnn-testkit and fnn-bench can drive the three
// libraries through one adapter.  Enum and class names keep the 0.1 spelling
// (Network_double, RegularizationType_double, ...) for backward compatibility.

#include <pybind11/native_enum.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <sstream>

#if __has_include("cudann/build_info.hpp")
#include "cudann/build_info.hpp"
#endif
#include "cudann/blas.hpp"
#include "cudann/config.hpp"
#include "cudann/device.hpp"
#include "cudann/network.hpp"

namespace py = pybind11;
using namespace cudann;

#ifndef CUDANN_MODULE_NAME
#define CUDANN_MODULE_NAME _cudann
#endif
#ifndef CUDANN_PY_PACKAGE
#define CUDANN_PY_PACKAGE "cudann"
#endif
#ifndef CUDANN_COMPILER_ID
#define CUDANN_COMPILER_ID "unknown"
#endif
#ifndef CUDANN_COMPILE_FLAGS
#define CUDANN_COMPILE_FLAGS ""
#endif
#ifndef CUDANN_CUDA_ARCHS
#define CUDANN_CUDA_ARCHS ""
#endif
#ifndef CUDANN_ONEMATH_VERSION
#define CUDANN_ONEMATH_VERSION ""
#endif
#ifndef CUDANN_GIT_SHA
#define CUDANN_GIT_SHA "unknown"
#endif

namespace {

template <typename T> using arr_t = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T> std::vector<T> to_vector(const arr_t<T> &a) {
    const T *p = a.data();
    return std::vector<T>(p, p + a.size());
}

template <typename T> py::array_t<T> to_array(const std::vector<T> &v) {
    py::array_t<T> out(static_cast<py::ssize_t>(v.size()));
    if (!v.empty())
        std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(T));
    return out;
}

template <typename T> py::list to_list_of_arrays(const std::vector<std::vector<T>> &vv) {
    py::list out;
    for (const auto &v : vv)
        out.append(to_array(v));
    return out;
}

py::dict profile_to_dict(const Profile &p) {
    py::dict d;
    d["h2d_ns"] = p.h2d_ns;
    d["d2h_ns"] = p.d2h_ns;
    d["gemm_ns"] = p.gemm_ns;
    d["act_ns"] = p.act_ns;
    d["delta_ns"] = p.delta_ns;
    d["biasgrad_ns"] = p.biasgrad_ns;
    d["update_ns"] = p.update_ns;
    d["loss_ns"] = p.loss_ns;
    d["reg_ns"] = p.reg_ns;
    d["other_ns"] = p.other_ns;
    d["wall_ns"] = p.wall_ns;
    d["wait_ns"] = p.wait_ns;
    d["launches"] = p.launches;
    d["bytes_h2d"] = p.bytes_h2d;
    d["bytes_d2h"] = p.bytes_d2h;
    d["epochs"] = p.epochs;
    d["batches"] = p.batches;
    d["unprofiled"] = p.unprofiled;
    d["device_total_ns"] = p.device_total_ns();
    d["epoch_wall_ns"] = p.epoch_wall_ns;
    return d;
}

py::dict device_to_dict(const DeviceInfo &d) {
    py::dict out;
    out["index"] = d.index;
    out["name"] = d.name;
    out["vendor"] = d.vendor;
    out["type"] = d.type;
    out["backend"] = d.backend;
    out["platform"] = d.platform;
    out["driver"] = d.driver;
    out["global_mem_bytes"] = d.global_mem_bytes;
    out["compute_units"] = d.compute_units;
    out["fp64"] = d.fp64;
    out["compute_capability"] = std::to_string(d.cc_major) + "." + std::to_string(d.cc_minor);
    return out;
}

template <typename T> void declare_regularization(py::module_ &m, const std::string &suffix) {
    using Reg = Regularization<T>;
    py::native_enum<typename Reg::Type>(m, ("RegularizationType_" + suffix).c_str(), "enum.Enum")
        .value("Disabled", Reg::Type::Disabled)
        .value("L1", Reg::Type::L1)
        .value("L2", Reg::Type::L2)
        .value("ElasticNet", Reg::Type::ElasticNet)
        .finalize();
    py::class_<Reg>(m, ("Regularization_" + suffix).c_str())
        .def(py::init<typename Reg::Type, T, T>(), py::arg("type") = Reg::Type::Disabled, py::arg("lambda1") = T(0),
             py::arg("lambda2") = T(0))
        .def_readwrite("type", &Reg::type)
        .def_readwrite("lambda1", &Reg::lambda1)
        .def_readwrite("lambda2", &Reg::lambda2);
}

template <typename T> void declare_stop_criteria(py::module_ &m, const std::string &suffix) {
    using SC = StopCriteria<T>;
    py::native_enum<typename SC::Type>(m, ("StopCriteriaType_" + suffix).c_str(), "enum.Enum")
        .value("MaxEpochs", SC::Type::MaxEpochs)
        .value("MinError", SC::Type::MinError)
        .value("MinErrorChange", SC::Type::MinErrorChange)
        .finalize();
    py::class_<SC>(m, ("StopCriteria_" + suffix).c_str())
        .def(py::init<typename SC::Type, T>(), py::arg("type") = SC::Type::MaxEpochs, py::arg("threshold") = T(1e-4))
        .def_readwrite("type", &SC::type)
        .def_readwrite("threshold", &SC::threshold);
}

template <typename T> void declare_momentum(py::module_ &m, const std::string &suffix) {
    using Mom = MomentumConfig<T>;
    py::native_enum<typename Mom::Type>(m, ("MomentumType_" + suffix).c_str(), "enum.Enum")
        .value("Disabled", Mom::Type::Disabled)
        .value("Classical", Mom::Type::Classical)
        .finalize();
    py::class_<Mom>(m, ("MomentumConfig_" + suffix).c_str())
        .def(py::init<typename Mom::Type, T>(), py::arg("type") = Mom::Type::Disabled, py::arg("momentum_rate") = T(0))
        .def_readwrite("type", &Mom::type)
        .def_readwrite("momentum_rate", &Mom::momentum_rate);
}

template <typename T> void declare_adaptive_lr(py::module_ &m, const std::string &suffix) {
    using ALR = AdaptiveLearningRate<T>;
    py::native_enum<typename ALR::Strategy>(m, ("AdaptiveLearningStrategy_" + suffix).c_str(), "enum.Enum")
        .value("Constant", ALR::Strategy::Constant)
        .value("LinearDecay", ALR::Strategy::LinearDecay)
        .value("AdaGrad", ALR::Strategy::AdaGrad)
        .value("RMSProp", ALR::Strategy::RMSProp)
        .value("Adam", ALR::Strategy::Adam)
        .finalize();
    py::class_<ALR>(m, ("AdaptiveLearningRate_" + suffix).c_str())
        .def(py::init<typename ALR::Strategy, T, T, T, T>(), py::arg("strategy") = ALR::Strategy::Constant,
             py::arg("epsilon") = T(1e-8), py::arg("beta1") = T(0.9), py::arg("beta2") = T(0.999),
             py::arg("final_learning_rate") = T(1e-4))
        .def_readwrite("strategy", &ALR::strategy)
        .def_readwrite("epsilon", &ALR::epsilon)
        .def_readwrite("beta1", &ALR::beta1)
        .def_readwrite("beta2", &ALR::beta2)
        .def_readwrite("final_learning_rate", &ALR::final_lr);
}

template <typename T> void declare_network(py::module_ &m, const std::string &suffix) {
    using Net = Network<T>;
    const std::string name = "Network_" + suffix;
    py::class_<Net>(m, name.c_str())
        .def(py::init<std::vector<LayerDescription>, T, Regularization<T>, BackPropagation, AdaptiveLearningRate<T>,
                      StopCriteria<T>, MomentumConfig<T>, Options>(),
             py::arg("layers"), py::arg("learning_rate"), py::arg("regularization") = Regularization<T>{},
             py::arg("backpropagation") = BackPropagation::Standard,
             py::arg("adaptive_learning_rate") = AdaptiveLearningRate<T>{}, py::arg("stop_criteria") = StopCriteria<T>{},
             py::arg("momentum") = MomentumConfig<T>{}, py::arg("options") = Options{},
             "Random initialisation seeded from the clock.")
        .def(py::init<std::vector<LayerDescription>, T, Regularization<T>, BackPropagation, AdaptiveLearningRate<T>,
                      StopCriteria<T>, MomentumConfig<T>, unsigned, Options>(),
             py::arg("layers"), py::arg("learning_rate"), py::arg("regularization") = Regularization<T>{},
             py::arg("backpropagation") = BackPropagation::Standard,
             py::arg("adaptive_learning_rate") = AdaptiveLearningRate<T>{}, py::arg("stop_criteria") = StopCriteria<T>{},
             py::arg("momentum") = MomentumConfig<T>{}, py::arg("seed"), py::arg("options") = Options{},
             "Random initialisation in [-0.1, 0.1] from std::mt19937(seed).")
        .def(py::init([](std::vector<LayerDescription> layers, T lr, Regularization<T> reg, BackPropagation bp,
                         AdaptiveLearningRate<T> alr, StopCriteria<T> stop, MomentumConfig<T> mom,
                         const std::vector<arr_t<T>> &weights, const std::vector<arr_t<T>> &biases, Options options) {
                 std::vector<std::vector<T>> w, b;
                 for (const auto &a : weights)
                     w.push_back(to_vector(a));
                 for (const auto &a : biases)
                     b.push_back(to_vector(a));
                 return new Net(std::move(layers), lr, reg, bp, alr, stop, mom, w, b, std::move(options));
             }),
             py::arg("layers"), py::arg("learning_rate"), py::arg("regularization") = Regularization<T>{},
             py::arg("backpropagation") = BackPropagation::Standard,
             py::arg("adaptive_learning_rate") = AdaptiveLearningRate<T>{}, py::arg("stop_criteria") = StopCriteria<T>{},
             py::arg("momentum") = MomentumConfig<T>{}, py::arg("initial_weights"), py::arg("initial_biases"),
             py::arg("options") = Options{},
             "Explicit initial weights (flattened column-major (n_out, n_in)) and biases.")
        .def(
            "train",
            [](Net &self, arr_t<T> X, arr_t<T> Y, unsigned n_samples, unsigned batch_size, unsigned max_epochs) {
                auto x = to_vector(X);
                auto y = to_vector(Y);
                std::vector<T> loss;
                {
                    py::gil_scoped_release release;
                    loss = self.train(x, y, n_samples, batch_size, max_epochs);
                }
                return to_array(loss);
            },
            py::arg("x"), py::arg("y"), py::arg("n_samples"), py::arg("batch_size"), py::arg("max_epochs"),
            R"doc(Train the network.

x, y : flattened inputs (n_samples x n_in) and targets (n_samples x n_out), sample-major.
Returns the total loss (mean data loss + regularisation penalty) of every epoch run.)doc")
        .def(
            "predict",
            [](Net &self, arr_t<T> X, unsigned n_samples, unsigned batch_size) {
                auto x = to_vector(X);
                std::vector<T> out;
                {
                    py::gil_scoped_release release;
                    out = self.predict(x, n_samples, batch_size);
                }
                return to_array(out);
            },
            py::arg("x"), py::arg("n_samples"), py::arg("batch_size") = 0,
            "Forward pass; returns flattened outputs (n_samples x n_out). batch_size 0 = one batch.")
        .def_property_readonly(
            "weights_biases",
            [](const Net &self) {
                auto [w, b] = self.weights_biases();
                py::list hw, hb;
                for (const auto &snap : w)
                    hw.append(to_list_of_arrays(snap));
                for (const auto &snap : b)
                    hb.append(to_list_of_arrays(snap));
                return py::make_tuple(hw, hb);
            },
            "Snapshots (weights_history, biases_history): [snapshot][layer] flattened column-major arrays. "
            "Snapshot 0 is the initial state; one per epoch with options.record_history, else the final state.")
        .def_property_readonly("weights", [](const Net &self) { return to_list_of_arrays(self.weights()); },
                               "Current weights, one flattened column-major (n_out, n_in) array per layer.")
        .def_property_readonly("biases", [](const Net &self) { return to_list_of_arrays(self.biases()); },
                               "Current biases, one array per layer.")
        .def(
            "set_weights_biases",
            [](Net &self, const std::vector<arr_t<T>> &weights, const std::vector<arr_t<T>> &biases) {
                std::vector<std::vector<T>> w, b;
                for (const auto &a : weights)
                    w.push_back(to_vector(a));
                for (const auto &a : biases)
                    b.push_back(to_vector(a));
                self.set_weights_biases(w, b);
            },
            py::arg("weights"), py::arg("biases"), "Replace all weights / biases and reset the history.")
        .def_property_readonly("profile", [](const Net &self) { return profile_to_dict(self.profile()); },
                               "Per-phase device times (ns) and counters; all zero unless options.profile.")
        .def("reset_profile", &Net::reset_profile)
        .def_property_readonly("options", &Net::options)
        .def_property_readonly("device_name", &Net::device_name)
        .def_property_readonly("device", [](const Net &self) { return device_to_dict(self.device_info()); })
        .def_property_readonly("blas_backend", &Net::blas_backend)
        .def_property_readonly("parameter_count", &Net::parameter_count)
        .def_property_readonly("layers", &Net::layers);
}

} // namespace

PYBIND11_MODULE(CUDANN_MODULE_NAME, m) {
    m.doc() = "cudann: CUDA feed-forward neural network (cuBLAS + hand-written kernels), 1:1 port of syclnn";
    m.attr("__version__") = CUDANN_VERSION_STRING;

    py::native_enum<ActivationType>(m, "ActivationType", "enum.Enum")
        .value("Disabled", ActivationType::Disabled)
        .value("Sigmoid", ActivationType::Sigmoid)
        .value("Tanh", ActivationType::Tanh)
        .value("ReLU", ActivationType::ReLU)
        .value("LeakyReLU", ActivationType::LeakyReLU)
        .value("ELU", ActivationType::ELU)
        .export_values()
        .finalize();
    py::native_enum<BackPropagation>(m, "BackPropagation", "enum.Enum")
        .value("Standard", BackPropagation::Standard)
        .export_values()
        .finalize();
    py::native_enum<MemoryKind>(m, "MemoryKind", "enum.Enum")
        .value("Device", MemoryKind::Device)
        .value("Shared", MemoryKind::Shared)
        .value("Host", MemoryKind::Host)
        .finalize();
    py::native_enum<QueueOrder>(m, "QueueOrder", "enum.Enum")
        .value("OutOfOrder", QueueOrder::OutOfOrder)
        .value("InOrder", QueueOrder::InOrder)
        .value("Graph", QueueOrder::Graph)
        .finalize();

    py::class_<LayerDescription>(m, "LayerDescription")
        .def(py::init<unsigned, ActivationType>(), py::arg("neurons"), py::arg("activation") = ActivationType::Disabled)
        .def_readwrite("neurons", &LayerDescription::neurons)
        .def_readwrite("activation", &LayerDescription::activation)
        .def("__repr__", [](const LayerDescription &l) {
            return "LayerDescription(" + std::to_string(l.neurons) + ", " + std::to_string(int(l.activation)) + ")";
        });

    auto opts = py::class_<Options>(m, "Options", "Runtime options: device, BLAS backend, profiling, ablation switches.")
                    .def(py::init<>())
                    .def_readwrite("device", &Options::device)
                    .def_readwrite("blas", &Options::blas)
                    .def_readwrite("profile", &Options::profile)
                    .def_readwrite("record_history", &Options::record_history)
                    .def_readwrite("shuffle", &Options::shuffle)
                    .def_readwrite("shuffle_seed", &Options::shuffle_seed)
                    .def_readwrite("streams", &Options::streams)
                    .def_readwrite("pinned_host", &Options::pinned_host)
                    .def_readwrite("loss_reduction", &Options::loss_reduction)
                    .def_readwrite("bias_gemv", &Options::bias_gemv)
                    .def_readwrite("direct_input", &Options::direct_input)
                    .def_readwrite("fine_deps", &Options::fine_deps)
                    .def_readwrite("join_kernels", &Options::join_kernels)
                    .def_readwrite("specialized_kernels", &Options::specialized_kernels)
                    .def_readwrite("derivative_from_output", &Options::derivative_from_output)
                    .def_readwrite("host_adam_correction", &Options::host_adam_correction)
                    .def_readwrite("workgroup_size", &Options::workgroup_size)
                    .def_readwrite("persistent_workspace", &Options::persistent_workspace)
                    .def_readwrite("fast_math", &Options::fast_math)
                    .def_readwrite("sync_every", &Options::sync_every)
                    .def_readwrite("blas_queue", &Options::blas_queue)
                    .def_readwrite("sync_ops", &Options::sync_ops);
    // memory / queue accept the enum or its lower-case name
    opts.def_property(
        "memory", [](const Options &o) { return o.memory; },
        [](Options &o, py::object v) {
            if (py::isinstance<py::str>(v)) {
                const std::string s = v.cast<std::string>();
                if (s == "device")
                    o.memory = MemoryKind::Device;
                else if (s == "shared" || s == "managed")
                    o.memory = MemoryKind::Shared;
                else if (s == "host" || s == "pinned" || s == "zero_copy")
                    o.memory = MemoryKind::Host;
                else
                    throw py::value_error("memory must be device|shared|host");
            } else {
                o.memory = v.cast<MemoryKind>();
            }
        });
    opts.def_property(
        "queue", [](const Options &o) { return o.queue; },
        [](Options &o, py::object v) {
            if (py::isinstance<py::str>(v)) {
                const std::string s = v.cast<std::string>();
                if (s == "out_of_order" || s == "ooo")
                    o.queue = QueueOrder::OutOfOrder;
                else if (s == "in_order")
                    o.queue = QueueOrder::InOrder;
                else if (s == "graph")
                    o.queue = QueueOrder::Graph;
                else
                    throw py::value_error("queue must be out_of_order|in_order|graph");
            } else {
                o.queue = v.cast<QueueOrder>();
            }
        });
    opts.def("__repr__", [](const Options &o) {
        std::ostringstream s;
        s << "Options(device='" << o.device << "', blas='" << o.blas << "', profile=" << o.profile
          << ", record_history=" << o.record_history << ", shuffle=" << o.shuffle << ", memory=" << int(o.memory)
          << ", queue=" << int(o.queue) << ", loss_reduction=" << o.loss_reduction << ", bias_gemv=" << o.bias_gemv
          << ", direct_input=" << o.direct_input << ", fine_deps=" << o.fine_deps << ", join_kernels=" << o.join_kernels
          << ", specialized_kernels=" << o.specialized_kernels << ", derivative_from_output=" << o.derivative_from_output
          << ", host_adam_correction=" << o.host_adam_correction << ", workgroup_size=" << o.workgroup_size
          << ", persistent_workspace=" << o.persistent_workspace << ", pinned_host=" << o.pinned_host << ")";
        return s.str();
    });

    declare_regularization<float>(m, "float");
    declare_stop_criteria<float>(m, "float");
    declare_momentum<float>(m, "float");
    declare_adaptive_lr<float>(m, "float");
    declare_network<float>(m, "float");

    declare_regularization<double>(m, "double");
    declare_stop_criteria<double>(m, "double");
    declare_momentum<double>(m, "double");
    declare_adaptive_lr<double>(m, "double");
    declare_network<double>(m, "double");

    m.def(
        "devices",
        [] {
            py::list out;
            for (const auto &d : devices())
                out.append(device_to_dict(d));
            return out;
        },
        "All CUDA devices visible to this build, as dicts (index, name, type, backend, ...).");
    m.def(
        "get_cuda_devices",
        [] {
            std::vector<std::string> names;
            for (const auto &d : devices())
                names.push_back(d.name);
            return names;
        },
        "Names of all visible devices (syclnn.get_sycl_devices counterpart).");
    m.def(
        "build_info",
        [] {
            py::dict d;
            d["version"] = CUDANN_VERSION_STRING;
            d["compiler"] = CUDANN_COMPILER_ID;
            d["flags"] = CUDANN_COMPILE_FLAGS;
            d["cuda_archs"] = CUDANN_CUDA_ARCHS;
            d["onemath"] = CUDANN_ONEMATH_VERSION;
            d["git_sha"] = CUDANN_GIT_SHA;
            d["blas_backends"] = compiled_blas_backends();
            d["blas_selectable"] = compiled_blas_backends();
            {
                int rt = 0, drv = 0;
                cudaRuntimeGetVersion(&rt);
                cudaDriverGetVersion(&drv);
                d["runtime_version"] = rt;
                d["driver_version"] = drv;
            }
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
            d["platform"] = "HIP (hipified)";
#else
            d["platform"] = "CUDA";
#endif
            d["dtypes"] = std::vector<std::string>{"float", "double"};
            return d;
        },
        "How this module was built: compiler, flags, SYCL targets, oneMath backends.");
}
