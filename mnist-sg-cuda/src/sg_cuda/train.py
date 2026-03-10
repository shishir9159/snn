"""Minibatch training with Adam, evaluation and per-sample activity."""

import time

import numpy as np

from .backend import host, xp
from .net import Adam, Net

# picojoules per accumulate and per multiply-accumulate, 32-bit float in 45 nm
# (Horowitz 2014), as in snn_core::energy
AC_PJ, MAC_PJ = 0.9, 4.6


class Trainer:
    def __init__(self, n_in, classes, params, seed=1, batch=128):
        self.rng = np.random.default_rng(seed)
        self.net = Net(n_in, classes, params, self.rng)
        self.opt = Adam(self.net.w, params.lr)
        self.batch = batch
        # called after every training batch with (epoch, samples done, total, running accuracy)
        self.on_batch = None
        # called with (epoch, idx, labels, state, scores, loss) for every training batch
        self.watch = None
        self.recent = []

    def running_accuracy(self):
        return float(np.mean(self.recent)) if self.recent else 0.0

    def _batches(self, n, shuffle):
        order = self.rng.permutation(n) if shuffle else np.arange(n)
        return [order[i:i + self.batch] for i in range(0, n, self.batch)]

    def train_epoch(self, data, epoch=0):
        """One shuffled pass. Returns accuracy and mean spikes and synops per sample."""
        hits, spikes, synops, done = 0, 0.0, 0.0, 0
        for idx in self._batches(len(data), True):
            x, y = data.batch(idx), xp.asarray(data.labels[idx].astype(np.int64))
            state = self.net.forward(x)
            loss, g = self.net.loss(state, y)
            self.opt.step(self.net.w, self.net.backward(state, g))
            scores = Net.scores(state)
            right = host(scores.argmax(axis=1) == y)
            hits += int(right.sum())
            self.recent = (self.recent + right.tolist())[-500:]
            s, o = self.net.activity(state)
            spikes, synops = spikes + float(s.sum()), synops + float(o.sum())
            done += len(idx)
            if self.watch:
                self.watch(epoch, idx, data.labels[idx], state, scores, loss)
            if self.on_batch:
                self.on_batch(epoch, done, len(data), self.running_accuracy())
        n = max(len(data), 1)
        return hits / n, spikes / n, synops / n

    def evaluate(self, data):
        """Accuracy and the answer for every sample."""
        answers = []
        for idx in self._batches(len(data), False):
            answers.append(host(Net.scores(self.net.forward(data.batch(idx))).argmax(axis=1)))
        answers = np.concatenate(answers) if answers else np.zeros(0, np.int64)
        return float((answers == data.labels).mean()) if len(data) else 0.0, answers

    def fit(self, train, test, epochs, on_epoch=None):
        out = []
        for e in range(epochs):
            t0 = time.perf_counter()
            acc, spikes, synops = self.train_epoch(train, e)
            rate = len(train) / max(time.perf_counter() - t0, 1e-9)
            test_acc = self.evaluate(test)[0] if test is not None and len(test) else None
            r = dict(epoch=e, train_accuracy=acc, test_accuracy=test_acc, images_per_sec=rate,
                     spikes=spikes, synops=synops)
            if on_epoch:
                on_epoch(r)
            out.append(r)
        return out
