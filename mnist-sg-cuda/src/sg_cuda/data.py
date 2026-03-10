"""Labelled spike recordings binned into steps: MNIST as one latency-coded
spike per pixel, or the Spiking Heidelberg Digits."""

import gzip
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .backend import xp


@dataclass
class Spikes:
    name: str
    n_in: int
    steps: int
    # milliseconds a step stands for
    dt_ms: float
    labels: np.ndarray
    # per sample, the step and the input of every spike
    times: list = field(repr=False)
    units: list = field(repr=False)
    # what the live view shows of a sample: width, height and a byte per cell
    shape: tuple = (0, 0)
    pictures: np.ndarray | None = field(default=None, repr=False)
    static: bool = True

    def __len__(self):
        return len(self.labels)

    @property
    def classes(self):
        return int(self.labels.max()) + 1 if len(self) else 0

    def truncate(self, n):
        self.labels, self.times, self.units = self.labels[:n], self.times[:n], self.units[:n]
        if self.pictures is not None:
            self.pictures = self.pictures[:n]

    def batch(self, idx):
        """Samples `idx` as spike counts (batch, steps, inputs), built on the
        device from the spikes alone."""
        rows = np.repeat(np.arange(len(idx)), [len(self.times[i]) for i in idx])
        t = np.concatenate([self.times[i] for i in idx]) if len(idx) else rows
        u = np.concatenate([self.units[i] for i in idx]) if len(idx) else rows
        flat = xp.asarray((rows * self.steps + t) * self.n_in + u)
        x = xp.zeros(len(idx) * self.steps * self.n_in, np.float32)
        xp.add.at(x, flat, np.float32(1))
        return x.reshape(len(idx), self.steps, self.n_in)

    def picture(self, i):
        if self.pictures is not None:
            return self.pictures[i]
        # spikes per input, scaled to bytes
        count = np.bincount(self.units[i], minlength=self.n_in).astype(np.float32)
        w, h = self.shape
        pic = np.zeros(w * h, np.float32)
        pic[: self.n_in] = count
        return (pic / max(pic.max(), 1.0) * 255).astype(np.uint8)


def _idx(folder, stem):
    for ext in ("", ".gz"):
        p = Path(folder) / (stem + ext)
        if p.is_file():
            raw = p.read_bytes()
            return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    raise FileNotFoundError(f"{stem} not found in {folder}")


def mnist(folder, train=True, steps=100, threshold=0.2):
    """One spike per pixel brighter than `threshold`, earlier the brighter:
    step (1 - brightness) * steps."""
    s = "train" if train else "t10k"
    img = _idx(folder, f"{s}-images-idx3-ubyte")
    n, rows, cols = (int(v) for v in np.frombuffer(img, ">u4", 3, 4))
    px = np.frombuffer(img, np.uint8, n * rows * cols, 16).reshape(n, rows * cols)
    labels = np.frombuffer(_idx(folder, f"{s}-labels-idx1-ubyte"), np.uint8, n, 8).copy()
    bright = px / np.float32(255)
    times, units = [], []
    for b in bright:
        lit = np.flatnonzero(b > threshold)
        times.append(np.minimum(((1 - b[lit]) * steps).astype(np.int64), steps - 1))
        units.append(lit)
    name = f"{Path(folder).name}/{s}"
    return Spikes(name, rows * cols, steps, 1.0, labels, times, units, (cols, rows), px.copy())
