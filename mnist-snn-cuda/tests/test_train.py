import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cupy as cp
import numpy as np

from snn_cuda import weights
from snn_cuda.dataset import Dataset
from snn_cuda.train import Labels, Trainer
from test_layer import bar, noisy


def bars(n):
    imgs = [bar(i % 2 == 0) if i % 3 else noisy(i) for i in range(n)]
    labels = [0 if i % 3 and i % 2 == 0 else 1 if i % 3 else 2 for i in range(n)]
    return Dataset("bars", 28, 28, np.stack(imgs), np.uint8(labels))


def test_labels_forget_and_pick_last_of_ties():
    lab = Labels(2, 3)
    assert lab.of(0) == -1
    lab.record(0, 1)
    assert lab.of(0) == 1
    for _ in range(5):
        lab.record(0, 2)
    assert lab.of(0) == 2
    lab.counts[1] = [1, 1, 0]
    assert lab.of(1) == 1


def test_learns_two_bars():
    d = bars(120)
    d.pixels, d.labels = d.pixels[d.labels < 2], d.labels[d.labels < 2]
    t = Trainer(784, 6, 2, seed=2)
    t.fit(d, None, 3)
    assert t.evaluate(d) > 0.9


def test_chunk_size_does_not_change_training():
    d = bars(40)
    ws = []
    for chunk in (3, 256):
        t = Trainer(784, 8, 3, seed=5)
        t.train_epoch(d, 0, chunk)
        ws.append(t.layer.weights())
    assert (ws[0] == ws[1]).all()


def test_weights_roundtrip(tmp_path):
    w = Trainer(784, 3, 2, seed=1).layer.weights()
    weights.save(tmp_path / "w.txt", w)
    assert (weights.load(tmp_path / "w.txt") == w).all()
