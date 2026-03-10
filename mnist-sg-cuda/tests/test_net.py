import numpy as np
import pytest

import sg_cuda.net as net_mod
from sg_cuda.data import Spikes
from sg_cuda.net import Net, Params, surrogate
from sg_cuda.train import Trainer


@pytest.fixture
def float64(monkeypatch):
    monkeypatch.setattr(net_mod, "F32", np.float64)


def smooth(net, k=4.0):
    """A sigmoid in place of the step, its true slope in place of the
    surrogate and the reset in the gradient: backward() is then the exact
    gradient, which finite differences can check."""
    sig = lambda v: 1.0 / (1.0 + np.exp(-k * v))  # noqa: E731
    net.spike = sig
    net.slope = lambda v: k * sig(v) * (1.0 - sig(v))
    net.detach_reset = False


@pytest.mark.parametrize("recurrent", [False, True])
def test_backward_is_the_gradient_of_the_loss(float64, recurrent):
    rng = np.random.default_rng(3)
    net = Net(5, 3, Params(hidden=4, recurrent=recurrent, gain=40.0), rng, np)
    net.w = {k: v.astype(np.float64) for k, v in net.w.items()}
    smooth(net)
    x = rng.random((2, 7, 5)) * (rng.random((2, 7, 5)) < 0.5)
    y = np.array([2, 0])

    def loss():
        return net.loss(net.forward(x), y)[0]

    state = net.forward(x)
    grads = net.backward(state, net.loss(state, y)[1])
    assert set(grads) == set(net.w)
    for name, w in net.w.items():
        num = np.zeros_like(w)
        for i in np.ndindex(w.shape):
            keep = w[i]
            w[i] = keep + 1e-6
            up = loss()
            w[i] = keep - 1e-6
            down = loss()
            w[i] = keep
            num[i] = (up - down) / 2e-6
        assert np.abs(num).max() > 1e-6, f"{name} has no gradient to check"
        assert np.allclose(grads[name], num, rtol=1e-4, atol=1e-8), name


def test_surrogate_peaks_at_threshold():
    v = np.array([-1.0, -0.01, 0.0, 0.01, 1.0])
    s = surrogate(v, 100.0)
    assert s[2] == 1.0 and s[1] == s[3] == 0.25 and s[0] < 1e-3


def test_activity_counts_deliveries():
    rng = np.random.default_rng(1)
    for recurrent, fan in ((False, 3), (True, 3 + 6)):
        net = Net(4, 3, Params(hidden=6, recurrent=recurrent, gain=60.0), rng, np)
        x = (rng.random((2, 20, 4)) < 0.3).astype(np.float32)
        state = net.forward(x)
        spikes, synops = net.activity(state)
        hidden = state["spk"].sum(axis=(0, 2))
        assert hidden.sum() > 0
        assert np.allclose(spikes, x.sum(axis=(1, 2)) + hidden)
        assert np.allclose(synops, x.sum(axis=(1, 2)) * 6 + hidden * fan)
    assert net.dense_macs(20, static=False) == 20 * (4 * 6 + 6 * 3 + 36)


def bars(n, rng, steps=20):
    """Class 0 spikes on the first half of the inputs, class 1 on the second."""
    labels = rng.integers(0, 2, n).astype(np.uint8)
    times, units = [], []
    for c in labels:
        u = rng.integers(0, 5, 12) + 5 * c
        times.append(rng.integers(0, steps, 12))
        units.append(u)
    return Spikes("bars", 10, steps, 1.0, labels, times, units, (10, 1))


def test_training_separates_two_patterns():
    rng = np.random.default_rng(0)
    train, test = bars(256, rng), bars(64, rng)
    t = Trainer(10, 2, Params(hidden=16, lr=1e-2), seed=2, batch=32)
    before = t.evaluate(test)[0]
    r = t.fit(train, test, 3)
    assert r[-1]["test_accuracy"] > 0.95 > before or r[-1]["test_accuracy"] > 0.95
    assert r[-1]["spikes"] > 0 and r[-1]["synops"] > r[-1]["spikes"]
