"""The CUDA kernels must do what the reference loops do, bit for bit."""

import numpy as np
import pytest

import sg_cuda  # noqa: F401  sets up the CUDA paths before CuPy loads

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("no CUDA device", allow_module_level=True)
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("no CUDA device", allow_module_level=True)

from sg_cuda.net import Net, Params  # noqa: E402


@pytest.mark.parametrize("recurrent", [False, True])
def test_fused_loops_match_the_reference(recurrent):
    rng = np.random.default_rng(4)
    net = Net(30, 5, Params(hidden=40, recurrent=recurrent, gain=30.0), rng, cp)
    x = cp.asarray((rng.random((6, 50, 30)) < 0.2).astype(np.float32))
    y = cp.asarray(rng.integers(0, 5, 6))
    fused = net.forward(x)
    net.fused = False
    ref = net.forward(x)
    for k in ("mem", "syn", "spk", "out"):
        assert cp.array_equal(fused[k], ref[k]), k
    assert float(ref["spk"].sum()) > 0
    g = net.loss(ref, y)[1]
    want = net.backward(ref, g)
    net.fused = True
    got = net.backward(fused, g)
    for k in want:
        assert cp.array_equal(got[k], want[k]), k
