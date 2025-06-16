"""GPU state of the winner-take-all layer and the launches that drive it."""

from pathlib import Path

import cupy as cp
import numpy as np

from .params import Params
from .rng import Rng

MAX_THREADS = 384  # launch bound of the kernel, which caps its registers
PRODUCERS = 256  # threads that build the next image's spike schedule
SHARED_MAX = 60 * 1024  # of 64 KiB, the rest is static
KERNEL_SRC = Path(__file__).with_name("kernels.cu")
_module = None


def _present_fn():
    global _module
    if _module is None:
        _module = cp.RawModule(
            code=KERNEL_SRC.read_text(encoding="utf-8"), options=("--fmad=false", "--std=c++14")
        )
    return _module.get_function("present")


def stdp_table(p):
    """Weight change per pre-minus-post spike time, dt in -t_back..=t_fore."""
    s = p.stdp
    dt = np.arange(-s.t_back, s.t_fore + 1, dtype=np.float32)
    pos = -np.float32(s.a_plus) * np.exp(-dt / np.float32(s.tau_plus))
    neg = np.float32(s.a_minus) * np.exp(dt / np.float32(s.tau_minus))
    return np.where(dt > 0, pos, neg).astype(np.float32)


def normalize_rows(w, norm, w_max):
    """Scale each row to L2 norm `norm`, capped at w_max (sequential f32 sums like Rust)."""
    if norm <= 0:
        return w
    ln = np.sqrt(np.cumsum(w * w, axis=1, dtype=np.float32)[:, -1])
    k = np.where(ln > 0, np.float32(norm) / np.where(ln > 0, ln, 1), 1).astype(np.float32)
    return np.minimum(w * k[:, None], np.float32(w_max)).astype(np.float32)


class Layer:
    def __init__(self, n_in, n_out, params=None, weights=None, rng=None):
        if n_out > MAX_THREADS - 64:
            raise ValueError(f"at most {MAX_THREADS - 64} output neurons")
        p = self.p = params or Params()
        self.n_in, self.n_out = n_in, n_out
        if weights is None:
            lo, hi = p.w_init
            w = (rng or Rng(1)).uniform(n_in * n_out, lo, hi).reshape(n_out, n_in)
            weights = normalize_rows(w, p.w_norm, p.stdp.w_max)
        weights = np.ascontiguousarray(weights, np.float32)
        assert weights.shape == (n_out, n_in)
        # input-major, so a spike adds one contiguous run
        self.w = cp.asarray(np.ascontiguousarray(weights.T))
        self.theta = cp.zeros(n_out, cp.float32)
        nn = -(-n_out // 32) * 32
        self.block = nn + min(PRODUCERS, MAX_THREADS - nn)
        self._config()
        self.fn = _present_fn()
        self.fn.max_dynamic_shared_size_bytes = self.shared
        self.sms = cp.cuda.Device().attributes["MultiProcessorCount"]

    def _config(self, width=None, height=None):
        p, s = self.p, self.p.stdp
        e, lif = p.encode, p.lif
        f = [lif.rest, lif.floor, lif.leak, e.pot_min, e.pot_max, e.freq_min, e.freq_max,
             e.period_scale, s.sigma, s.w_min, s.w_max, p.threshold_div, p.silent_penalty,
             p.w_norm, p.theta_rate, 0.0]
        self.fcfg = cp.asarray(np.concatenate([np.float32(f), stdp_table(p)]))
        side = int(round(self.n_in ** 0.5))
        self.width, self.height = width or side, height or self.n_in // (width or side)
        min_period = max(int(np.ceil(np.float32(e.period_scale) / np.float32(e.freq_max))), 1)
        if min_period < 2 or p.time_steps >= 0xFFFF:
            raise ValueError("kernel needs spike periods of at least 2 steps and under 65536 steps")
        max_spikes = self.n_in * max(p.time_steps // min_period, 1)
        n_in, steps = self.n_in, p.time_steps
        al4 = lambda n: (n + 3) & ~3
        # two schedules (period, lit, divisors, step starts, slices, spikes), then the pixel
        # plane, the gather stage, the winner's column and drives, and the STDP table
        nn = -(-self.n_out // 32) * 32
        cap = min(32, 8192 // nn)
        sched = (2 * al4(2 * n_in) + 4 * n_in + 4 * (steps + 2) + al4(2 * (steps + 2))
                 + al4(2 * max_spikes))
        self.shared = (2 * sched + 4 * (self.width + 4) * (self.height + 4) + 4 * cap * nn
                       + 4 * n_in + 4 * (steps + 2) + 4 * (s.t_back + s.t_fore + 1))
        if self.shared > SHARED_MAX:
            raise ValueError(f"needs {self.shared} bytes of shared memory per block")
        i = [self.n_in, self.n_out, self.width, self.height, p.time_steps, lif.t_ref,
             s.t_back, s.t_fore, max_spikes]
        self.icfg = cp.asarray(np.int32(i))

    def shape(self, width, height):
        assert width * height == self.n_in
        self._config(width, height)
        self.fn.max_dynamic_shared_size_bytes = self.shared

    def weights(self):
        """Row-major (n_out, n_in) copy on the host."""
        return cp.asnumpy(self.w).T.copy()

    def present(self, pixels, order, learn, record=None):
        """Runs images `pixels[order]` through the layer.

        `pixels` is a device (n, n_in) uint8 array, `order` a device int32 index array.
        Learning walks them in order in one block; evaluation spreads images over blocks.
        Returns device arrays (winners, spikes, base thresholds). `record` is an optional
        (act, pot, fired) tuple of device buffers for a single image.
        """
        n = len(order)
        winners = cp.empty(n, cp.int32)
        spikes = cp.empty((n, self.n_out), cp.uint16)
        bases = cp.empty(n, cp.float32)
        grid = 1 if learn else min(n, self.sms)
        null = np.uint64(0)
        rec = record or (null, null, null)
        self.fn((grid,), (self.block,), (
            pixels, order, np.int32(n), np.int32(learn), self.fcfg, self.icfg,
            self.w, self.theta, winners, spikes, bases, *rec), shared_mem=self.shared)
        return winners, spikes, bases
