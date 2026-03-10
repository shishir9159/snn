"""Current-based LIF network: input spikes drive a hidden spiking layer
(optionally recurrent), which drives leaky non-spiking readout units; a class
scores the peak of its readout potential over the presentation.

Gradients come from backpropagation through time written out by hand. The
spike's step function gets the fast-sigmoid surrogate derivative of SuperSpike
(Zenke & Ganguli 2018), and the reset stays out of the gradient, as in Zenke
& Vogels 2021. On CuPy the time loops run as the kernels of kernels.cu; the
loops here are the reference they match.
"""

import math
from dataclasses import dataclass

import numpy as np

from . import backend

F32 = np.float32


@dataclass
class Params:
    hidden: int = 128
    recurrent: bool = False
    # time constants of the synaptic current and the membrane, steps
    tau_syn: float = 5.0
    tau_mem: float = 10.0
    # steepness of the surrogate derivative
    scale: float = 100.0
    # initial weights have std gain * (1 - beta) / sqrt(fan in)
    gain: float = 7.0
    lr: float = 1e-3


def surrogate(v, scale):
    """d spike / d v for a spike at v > 0: 1 / (scale |v| + 1)^2."""
    d = scale * abs(v) + 1.0
    return 1.0 / (d * d)


class Net:
    def __init__(self, n_in, classes, p, rng, xp=None):
        self.p, self.n_in, self.classes = p, n_in, classes
        self.xp = xp or backend.xp
        self.alpha = F32(math.exp(-1.0 / p.tau_syn))
        self.beta = F32(math.exp(-1.0 / p.tau_mem))
        # the spike nonlinearity and whether the reset is in the gradient; tests swap them
        self.spike = lambda v: (v > 0).astype(F32)
        self.slope = lambda v: surrogate(v, p.scale)
        self.detach_reset = True
        # the CUDA kernels, which know only the step, its surrogate and a detached reset
        self.fused = self.xp is not np
        if self.fused:
            from . import kernels

            self.k = kernels
        g = p.gain * (1.0 - float(self.beta))
        init = lambda fan, shape: self.xp.asarray(rng.normal(0.0, g / math.sqrt(fan), shape), F32)  # noqa: E731
        self.w = {"in": init(n_in, (n_in, p.hidden)), "out": init(p.hidden, (p.hidden, classes))}
        if p.recurrent:
            self.w["rec"] = init(p.hidden, (p.hidden, p.hidden))

    def forward(self, x):
        """x is (batch, steps, inputs). Returns the state backward() needs:
        membrane, current and spikes per step (steps, batch, hidden), and the
        readout potential (steps + 1, batch, classes), starting at 0."""
        xp = self.xp
        b, steps, _ = x.shape
        h, a, be = self.p.hidden, self.alpha, self.beta
        drive = (x.reshape(b * steps, -1) @ self.w["in"]).reshape(b, steps, h)
        rec = self.w.get("rec")
        if self.fused and rec is None:
            mems, syns, spk = self.k.lif_forward(drive, a, be)
        else:
            mems, syns, spk = (xp.empty((steps, b, h), F32) for _ in range(3))
            mem, syn = xp.zeros((b, h), F32), xp.zeros((b, h), F32)
            for t in range(steps):
                if self.fused:
                    rec_in = self.spike(mem - 1.0) @ rec
                    self.k.lif_step(mem, syn, drive[:, t], rec_in, mems[t], syns[t], spk[t], a, be)
                    continue
                s = self.spike(mem - 1.0)
                mems[t], syns[t], spk[t] = mem, syn, s
                inp = drive[:, t] if rec is None else drive[:, t] + s @ rec
                mem = (be * mem + syn) * (1.0 - s)
                syn = a * syn + inp
        h2 = (spk.reshape(steps * b, h) @ self.w["out"]).reshape(steps, b, -1)
        if self.fused:
            outs = self.k.readout_forward(h2, a, be)
        else:
            flt, out = xp.zeros_like(h2[0]), xp.zeros_like(h2[0])
            outs = xp.zeros((steps + 1,) + h2[0].shape, F32)
            for t in range(steps):
                out = be * out + flt
                flt = a * flt + h2[t]
                outs[t + 1] = out
        return {"x": x, "mem": mems, "syn": syns, "spk": spk, "out": outs}

    @staticmethod
    def scores(state):
        """Peak readout potential per class over the presentation."""
        return state["out"].max(axis=0)

    def loss(self, state, labels):
        """Mean cross-entropy of the scores, and its gradient on the readout."""
        xp = self.xp
        outs = state["out"]
        m = outs.max(axis=0)
        z = m - m.max(axis=1, keepdims=True)
        p = xp.exp(z) / xp.exp(z).sum(axis=1, keepdims=True)
        b = m.shape[0]
        rows = xp.arange(b)
        loss = float(-xp.log(p[rows, labels] + 1e-12).mean())
        dm = p
        dm[rows, labels] -= 1.0
        dm /= b
        # the max passes its gradient to the step that reached it
        g = xp.zeros_like(outs)
        peak = outs.argmax(axis=0)
        cols = xp.arange(m.shape[1])
        g[peak, rows[:, None], cols[None, :]] = dm
        return loss, g

    def backward(self, state, g_out):
        """Gradients of every weight from the gradient on the readout potential."""
        xp = self.xp
        x, mems, syns, spk = state["x"], state["mem"], state["syn"], state["spk"]
        steps, b, h = spk.shape
        a, be, w = self.alpha, self.beta, self.w
        if self.fused:
            g_h2 = self.k.readout_backward(g_out, a, be)
        else:
            # readout: out[t+1] = beta out[t] + flt[t], flt[t+1] = alpha flt[t] + h2[t]
            g_h2 = xp.empty((steps, b, self.classes), F32)
            gu, gf = g_out[steps].copy(), xp.zeros((b, self.classes), F32)
            for t in range(steps - 1, -1, -1):
                g_h2[t] = gf
                gu, gf = g_out[t] + be * gu, gu + a * gf
        grads = {"out": spk.reshape(steps * b, h).T @ g_h2.reshape(steps * b, -1)}
        g_spk = (g_h2.reshape(steps * b, -1) @ w["out"].T).reshape(steps, b, h)
        # hidden: mem[t+1] = (beta mem[t] + syn[t])(1 - s[t]), syn[t+1] = alpha syn[t] + in[t]
        rec = w.get("rec")
        if self.fused and rec is None:
            g_drive = self.k.lif_backward(g_spk, mems, spk, a, be, self.p.scale)
        else:
            g_mem, g_syn = xp.zeros((b, h), F32), xp.zeros((b, h), F32)
            g_drive = xp.empty((steps, b, h), F32)
            g_rec = xp.zeros_like(rec) if rec is not None else None
            for t in range(steps - 1, -1, -1):
                s, keep = spk[t], 1.0 - spk[t]
                g_s = g_spk[t]
                if rec is not None:
                    g_s = g_s + g_syn @ rec.T
                    g_rec += s.T @ g_syn
                if self.fused:
                    self.k.lif_step_back(g_mem, g_syn, g_s, mems[t], s, g_drive[t], a, be, self.p.scale)
                    continue
                g_drive[t] = g_syn
                slope = self.slope(mems[t] - 1.0)
                g_mem_t = g_s * slope + g_mem * be * keep
                if not self.detach_reset:
                    g_mem_t -= g_mem * (be * mems[t] + syns[t]) * slope
                g_syn, g_mem = g_mem * keep + a * g_syn, g_mem_t
            if rec is not None:
                grads["rec"] = g_rec
        xs = x.reshape(b * steps, -1)
        grads["in"] = xs.T @ g_drive.transpose(1, 0, 2).reshape(b * steps, h)
        return grads

    def activity(self, state):
        """Spikes and synaptic operations per sample: an input spike reaches
        every hidden unit, a hidden spike every readout unit (and every hidden
        unit when recurrent)."""
        x, spk = state["x"], state["spk"]
        fan = self.classes + (self.p.hidden if "rec" in self.w else 0)
        inputs, hidden = x.sum(axis=(1, 2)), spk.sum(axis=(0, 2))
        return inputs + hidden, inputs * self.p.hidden + hidden * fan

    def dense_macs(self, steps, static):
        """MACs per sample of a non-spiking network with the same layers: once
        for a static image, every step for a time series."""
        per_step = self.n_in * self.p.hidden + self.p.hidden * self.classes
        per_step += self.p.hidden**2 if "rec" in self.w else 0
        return per_step * (1 if static else steps)


class Adam:
    def __init__(self, weights, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: v * 0 for k, v in weights.items()}
        self.v = {k: v * 0 for k, v in weights.items()}

    def step(self, weights, grads):
        self.t += 1
        c1, c2 = 1 - self.b1**self.t, 1 - self.b2**self.t
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            weights[k] -= (self.lr * (self.m[k] / c1) / ((self.v[k] / c2) ** 0.5 + self.eps)).astype(F32)
