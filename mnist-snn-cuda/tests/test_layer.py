import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cupy as cp
import numpy as np
import pytest

from reference import present as ref_present
from snn_cuda.layer import Layer
from snn_cuda.params import Params
from snn_cuda.rng import Rng

P = Params()


def bar(vertical):
    img = np.zeros((28, 28), np.uint8)
    if vertical:
        img[4:24, 12:16] = 255
    else:
        img[12:16, 4:24] = 255
    return img.ravel()


def noisy(seed):
    r = np.random.default_rng(seed)
    img = np.zeros((28, 28), np.uint8)
    for _ in range(3):
        y, x = r.integers(2, 20, 2)
        img[y:y + 8, x:x + 3] = r.integers(120, 256)
    return img.ravel()


def run(layer, imgs, learn):
    px = cp.asarray(np.stack(imgs))
    order = cp.arange(len(imgs), dtype=cp.int32)
    return [cp.asnumpy(a) for a in layer.present(px, order, learn)]


def test_single_winner_spikes():
    w = np.full((3, 784), 0.5, np.float32)
    w[1] = 0.6
    layer = Layer(784, 3, P, weights=w)
    (win,), (sp,), _ = run(layer, [bar(True)], False)
    assert win == 1 and sp[1] > 0 and sp[0] + sp[2] == 0


def test_blank_image_never_fires():
    layer = Layer(784, 4, P, rng=Rng(1))
    (win,), (sp,), _ = run(layer, [np.zeros(784, np.uint8)], True)
    assert win == -1 and not sp.any()


def test_inference_leaves_weights_alone():
    layer = Layer(784, 4, P, rng=Rng(2))
    before = layer.weights()
    run(layer, [bar(True)], False)
    assert (layer.weights() == before).all() and not cp.asnumpy(layer.theta).any()


def test_learning_separates_two_patterns():
    layer = Layer(784, 2, P, rng=Rng(3))
    v, h = bar(True), bar(False)
    run(layer, [v, h] * 30, True)
    (wv, wh), _, _ = run(layer, [v, h], False)
    assert wv >= 0 and wh >= 0 and wv != wh
    assert layer.weights()[wv][14 * 28 + 13] > layer.weights()[wv][2 * 28 + 2]


@pytest.mark.parametrize("n_out", [5, 40])
def test_matches_numpy_reference(n_out):
    imgs = [bar(True), bar(False), noisy(1), noisy(2), noisy(3)] * 3
    layer = Layer(784, n_out, P, rng=Rng(4))
    w, theta = layer.weights(), np.zeros(n_out, np.float32)
    got_w, got_s, got_b = run(layer, imgs, True)
    for k, img in enumerate(imgs):
        win, sp, base = ref_present(w, theta, img, 28, 28, P, True)
        assert got_w[k] == win and (got_s[k] == sp).all() and got_b[k] == base, k
    np.testing.assert_allclose(layer.weights(), w, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(cp.asnumpy(layer.theta), theta, rtol=1e-5, atol=1e-6)


def test_parallel_eval_matches_sequential_blocks():
    imgs = [bar(True), bar(False), noisy(5), noisy(6)] * 8
    layer = Layer(784, 10, P, rng=Rng(5))
    run(layer, imgs[:12], True)
    full = run(layer, imgs, False)
    for k, img in enumerate(imgs):
        one = run(layer, [img], False)
        assert full[0][k] == one[0][0] and (full[1][k] == one[1][0]).all()
