"""Live dashboard for the CUDA trainer through snn_live (the Rust server).

Images nobody watches are not recorded; a viewed one runs alone with per-step
buffers filled by the kernel and copied back once.
"""

import time

import cupy as cp
import numpy as np

INPUT, OUTPUT = 0, 1
FIELDS = 16  # neurons whose weight maps are sent


class LiveRun:
    def __init__(self, trainer, train, test, epochs, port=7878, speed=1.0):
        import snn_live

        self.t, self.train, self.test, self.epochs = trainer, train, test, epochs
        lay = trainer.layer
        p = lay.p
        self.steps = p.time_steps
        self.unthrottled = speed is None
        self.hits_total = 0
        meta = dict(model="pop", dataset=train.name, width=train.width, height=train.height,
                    classes=trainer.labels.counts.shape[1], n_train=len(train), n_test=len(test),
                    epochs=epochs, steps=self.steps)
        self.live = snn_live.Live(
            meta,
            [("input", train.width, train.height, "input"), ("output", lay.n_out, "output")],
            title="Neuron model · LIF · CUDA",
            params=[("Threshold", "adaptive"), ("Leak", p.lif.leak, "/ms"),
                    ("Refractory", p.lif.t_ref, "ms"), ("Floor", p.lif.floor),
                    ("Weights", f"{p.stdp.w_min}–{p.stdp.w_max}"),
                    ("Learning rate", p.stdp.sigma), ("Backend", "CuPy + CUDA")],
            series=(["output rate"], ["v"], ["spikes/step"]),
            port=port, speed=speed)
        n = self.steps
        self.rec = (cp.zeros((n, lay.n_in), cp.uint8), cp.zeros((n, lay.n_out), cp.float32),
                    cp.zeros((n, lay.n_out), cp.uint8))

    def _record(self, px, k):
        """Runs image k alone and returns what the view needs."""
        act, pot, fired = self.rec
        act.fill(0)
        lay = self.t.layer
        order = cp.asarray([k], dtype=cp.int32)
        winners, spikes, bases = lay.present(px, order, True, record=self.rec)
        return winners, spikes, bases

    def _stream(self, steps_act, steps_fired, pot, probe_unit):
        live, n_out = self.live, pot.shape[1]
        for t in range(self.steps):
            a = np.flatnonzero(steps_act[t]).astype(np.uint16)
            f = np.flatnonzero(steps_fired[t]).astype(np.uint16)
            live.step(t, [a, f], np.float32([len(f) / n_out]), np.float32([pot[t, probe_unit]]))

    def _fields(self, w, units):
        lay, labels = self.t.layer, self.t.labels
        side = (self.train.height, self.train.width)
        theta = cp.asnumpy(lay.theta)
        cls = [labels.of(int(u)) for u in units]
        self.live.fields("Weights", OUTPUT, w[units].reshape(-1, *side), units.astype(np.uint16),
                         cls, theta[units].astype(np.float32))

    def epoch(self, e):
        t, live, data = self.t, self.live, self.train
        px = t._upload(data)
        lay = t.layer
        order = list(range(len(data)))
        t.rng.shuffle(order)
        hits, t0 = 0, time.perf_counter()
        for index, k in enumerate(order):
            if live.stopped:
                break
            label = int(data.labels[k])
            info = (e, index, len(data), label, True)
            probe = live.probe_request() or (OUTPUT, 0)
            unit = probe[1] % lay.n_out if probe[0] == OUTPUT else 0
            streaming = live.begin_image(info, data.pixels[k], (OUTPUT, unit, 1.0))
            if streaming:
                winners, spikes, bases = self._record(px, k)
            else:
                winners, spikes, bases = lay.present(px, cp.asarray([k], dtype=cp.int32), True)
            win, spk = int(winners[0]), cp.asnumpy(spikes[0])
            n_in_spikes = 0
            if streaming:
                act, pot, fired = (cp.asnumpy(a) for a in self.rec)
                n_in_spikes = int(act.sum())
                self._stream(act, fired, pot, unit)
            elif not self.unthrottled:
                for step in range(self.steps):
                    live.step(step)
            correct = t.settle(win, label)
            hits += correct
            pred = t.labels.of(win) if win >= 0 else -1
            if streaming:
                w = lay.weights()
                self.live.map(0, w.mean(0).reshape(data.height, data.width))
                if live.fields_due:
                    best = np.argsort(-t.labels.counts.sum(1))[:FIELDS]
                    self._fields(w, np.sort(best))
            scores = np.zeros(t.labels.counts.shape[1], np.float32)
            for j in np.flatnonzero(spk):
                c = t.labels.of(int(j))
                if c >= 0:
                    scores[c] += spk[j]
            live.end_image(info, pred if pred >= 0 else None, bool(correct),
                           t.running_accuracy(), scores, [n_in_spikes, int(spk.sum())],
                           np.float32([spk.sum() / self.steps]))
        return hits / max(len(data), 1), len(data) / max(time.perf_counter() - t0, 1e-9)

    def fit(self):
        for e in range(self.epochs):
            train_acc, rate = self.epoch(e)
            if self.live.stopped:
                break
            test_acc = self.t.evaluate(self.test) if len(self.test) else None
            self.live.epoch_end(e, train_acc, test_acc, rate)
        self.live.hold()
