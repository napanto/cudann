"""Backend-specific smoke tests; the full parity suite is collected from fnn_testkit."""

import numpy as np
import pytest

cudann = pytest.importorskip("cudann")


def test_import_and_devices():
    assert cudann.__version__.startswith("0.1")
    devs = cudann.devices()
    assert devs and all("compute_capability" in d for d in devs)
    info = cudann.build_info()
    assert info["platform"] in ("CUDA", "HIP (hipified)")
    assert "cuda_archs" in info


@pytest.mark.parametrize("queue", ["out_of_order", "in_order", "graph"])
def test_queue_modes_train(queue):
    layers = [cudann.LayerDescription(3), cudann.LayerDescription(5, cudann.ActivationType.Tanh),
              cudann.LayerDescription(2, cudann.ActivationType.Sigmoid)]
    X = np.random.default_rng(0).uniform(-1, 1, (16, 3)).astype(np.float32)
    Y = np.zeros((16, 2), dtype=np.float32)
    nets = {}
    for q in ("out_of_order", queue):
        net = cudann.Network(layers, 0.1, dtype="float", seed=1, queue=q, profile=True)
        nets[q] = net.train(X.ravel(), Y.ravel(), 16, 5, 3)  # 4 batches, remainder of 1
    np.testing.assert_allclose(nets[queue], nets["out_of_order"], rtol=1e-5)


def test_unknown_blas_rejected():
    layers = [cudann.LayerDescription(2), cudann.LayerDescription(1, cudann.ActivationType.Tanh)]
    with pytest.raises(ValueError):
        cudann.Network(layers, 0.1, blas="mklcpu", seed=1)
