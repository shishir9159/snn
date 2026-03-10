"""Live dashboard through snn_live (the Rust server). Training runs in
batches; afterwards each sample of the batch is replayed to the view as if it
had been presented alone, and the ones the view asks for stream step by step.
"""

import numpy as np

from . import backend
from .backend import host

INPUT, HIDDEN = 0, 1
FIELDS = 16  # hidden units whose input weights are drawn


class LiveRun:
    def __init__(self, trainer, train, test, epochs, port=7878, speed=1.0):
        import snn_live

        self.t, self.train, self.test, self.epochs = trainer, train, test, epochs
        net, p = trainer.net, trainer.net.p
        w, h = train.shape
        self.unthrottled = speed is None
        meta = dict(model="pop", dataset=train.name, width=w, height=h, classes=net.classes,
                    n_train=len(train), n_test=len(test), epochs=epochs, steps=train.steps,
                    dt_ms=train.dt_ms, dense_macs=net.dense_macs(train.steps, train.static))
        params = [("Hidden", p.hidden), ("Recurrent", "yes" if p.recurrent else "no"),
                  ("τ current", p.tau_syn * train.dt_ms, "ms"), ("τ membrane", p.tau_mem * train.dt_ms, "ms"),
                  ("Surrogate", f"fast sigmoid · {p.scale:g}"), ("Adam step", p.lr),
                  ("Backend", backend.name())]
        # the readout never spikes; the decision panel shows its scores
        pops = [("input", w, h, "input"), ("hidden", p.hidden, "latent")]
        self.live = snn_live.Live(meta, pops, title="Model · surrogate gradient", params=params,
                                  series=(["hidden rate"], ["v"], ["loss"]), port=port, speed=speed)
        trainer.watch = self.batch
        self.done = 0

    def _stream(self, state, k, unit):
        """Steps of sample k: its input and hidden spikes, readout potentials."""
        x, spk, mem = host(state["x"][k]), host(state["spk"][:, k]), host(state["mem"][:, k, unit])
        hidden = spk.shape[1]
        for t in range(x.shape[0]):
            ins = np.flatnonzero(x[t]).astype(np.uint16)
            hid = np.flatnonzero(spk[t]).astype(np.uint16)
            self.live.step(t, [ins, hid], np.float32([len(hid) / hidden]),
                           np.float32([mem[t]]))

    def _fields(self, spikes):
        """Input weights of the hidden units that fired most in the batch."""
        w, h = self.train.shape
        units = np.sort(np.argsort(-spikes)[:FIELDS])
        weights = host(self.t.net.w["in"])
        grid = np.zeros((len(units), w * h), np.float32)
        grid[:, : weights.shape[0]] = weights[:, units].T
        self.live.fields("Input weights", HIDDEN, grid.reshape(len(units), h, w), units.astype(np.uint16),
                         None, spikes[units].astype(np.float32))

    def batch(self, epoch, idx, labels, state, scores, loss):
        live, net = self.live, self.t.net
        answers = host(scores.argmax(axis=1))
        s = host(scores)
        probs = np.exp(s - s.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        inputs = host(state["x"].sum(axis=(1, 2)))
        hidden = host(state["spk"].sum(axis=0))
        fan = net.classes + (net.p.hidden if "rec" in net.w else 0)
        acc = self.t.running_accuracy()
        for k, i in enumerate(idx):
            if live.stopped:
                return
            info = (epoch, self.done + k, len(self.train), int(labels[k]), True)
            probe = live.probe_request()
            unit = probe[1] % net.p.hidden if probe and probe[0] == HIDDEN else int(np.argmax(hidden[k]))
            if live.begin_image(info, self.train.picture(i), (HIDDEN, unit, 1.0)):
                self._stream(state, k, unit)
            elif not self.unthrottled:
                for t in range(self.train.steps):
                    live.step(t)
            fired = int(hidden[k].sum())
            if live.fields_due:
                self._fields(hidden.sum(axis=0))
            live.end_image(info, int(answers[k]), bool(answers[k] == labels[k]), acc,
                           probs[k].astype(np.float32), [int(inputs[k]), fired], np.float32([loss]),
                           int(inputs[k] * net.p.hidden + fired * fan))
        self.done = (self.done + len(idx)) % max(len(self.train), 1)

    def epoch(self, r):
        self.live.epoch_end(r["epoch"], r["train_accuracy"], r["test_accuracy"], r["images_per_sec"],
                            r["spikes"], r["synops"])

    def fit(self):
        self.t.fit(self.train, self.test, self.epochs, self.epoch)
        self.live.hold()
