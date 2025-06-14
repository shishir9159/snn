"""Training loop: presents images, labels neurons by the classes they win, tracks accuracy."""

import time
from collections import deque

import cupy as cp
import numpy as np

from .layer import Layer
from .params import Params
from .rng import Rng

RECENT = 500  # images in the running-accuracy window
CHUNK = 256  # images per training launch
AHEAD = 2  # launches in flight before the host reads results back


class Labels:
    """Per-neuron class counts with forgetting, so labels follow the weights as they change."""

    DECAY = np.float32(0.99)

    def __init__(self, n_out, classes):
        self.counts = np.zeros((n_out, classes), np.float32)

    def record(self, neuron, cls):
        row = self.counts[neuron]
        row *= self.DECAY
        row[cls] += 1

    def of(self, neuron):
        """Class the neuron wins most, -1 if it has won nothing."""
        row = self.counts[neuron]
        best = len(row) - 1 - int(np.argmax(row[::-1]))  # last of equal maxima, like Rust
        return best if row[best] > 0 else -1

    def table(self):
        return np.array([self.of(j) for j in range(len(self.counts))], np.int32)


class Trainer:
    def __init__(self, n_in, n_out, classes, params=None, seed=1, weights=None):
        self.params = params or Params()
        self.rng = Rng(seed)
        self.layer = Layer(n_in, n_out, self.params, weights=weights, rng=self.rng)
        self.labels = Labels(n_out, classes)
        self.recent = deque(maxlen=RECENT)
        self.on_image = None  # optional callback(epoch, index, total, running accuracy)

    def running_accuracy(self):
        return sum(self.recent) / max(len(self.recent), 1)

    def _upload(self, data):
        if getattr(data, "dev", None) is None:
            data.dev = cp.asarray(data.pixels)
        return data.dev

    def train_epoch(self, data, epoch=0, chunk=CHUNK):
        px = self._upload(data)
        order = list(range(len(data)))
        self.rng.shuffle(order)
        order = np.int32(order)
        order_d = cp.asarray(order)
        pending, hits = deque(), 0
        for start in range(0, len(order), chunk):
            idx = order_d[start:start + chunk]
            pending.append((start, order[start:start + chunk],
                            self.layer.present(px, idx, True)))
            if len(pending) > AHEAD:
                hits += self._drain(epoch, len(order), data, pending.popleft())
        while pending:
            hits += self._drain(epoch, len(order), data, pending.popleft())
        return hits / max(len(order), 1)

    def _drain(self, epoch, total, data, item):
        """Replays one launch's winners on the host: labels and running accuracy."""
        start, ids, launch = item
        hits = 0
        for k, w in enumerate(cp.asnumpy(launch[0])):
            label = int(data.labels[ids[k]])
            correct = w >= 0 and self.labels.of(int(w)) == label
            if w >= 0:
                self.labels.record(int(w), label)
            self.recent.append(correct)
            hits += correct
            if self.on_image:
                self.on_image(epoch, start + k, total, self.running_accuracy())
        return hits

    def evaluate(self, data):
        """Accuracy without learning; weights and labels stay put."""
        px = self._upload(data)
        order = cp.arange(len(data), dtype=cp.int32)
        winners = cp.asnumpy(self.layer.present(px, order, False)[0])
        table = np.append(self.labels.table(), -1)  # index -1 (no winner) -> no class
        return float((table[winners] == data.labels).mean()) if len(data) else 0.0

    def assign_labels(self, data):
        """Labels neurons from a pass over `data` without touching the weights."""
        px = self._upload(data)
        order = cp.arange(len(data), dtype=cp.int32)
        for k, w in enumerate(cp.asnumpy(self.layer.present(px, order, False)[0])):
            if w >= 0:
                self.labels.record(int(w), int(data.labels[k]))

    def fit(self, train, test=None, epochs=1, on_epoch=None):
        results = []
        for epoch in range(epochs):
            t0 = time.perf_counter()
            train_acc = self.train_epoch(train, epoch)
            rate = len(train) / max(time.perf_counter() - t0, 1e-9)
            test_acc = self.evaluate(test) if test is not None and len(test) else None
            r = dict(epoch=epoch, train_accuracy=train_acc, test_accuracy=test_acc,
                     images_per_sec=rate)
            results.append(r)
            if on_epoch:
                on_epoch(r)
        return results
