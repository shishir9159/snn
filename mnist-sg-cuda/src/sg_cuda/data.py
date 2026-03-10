"""Labelled spike recordings binned into steps: MNIST as one latency-coded
spike per pixel, or the Spiking Heidelberg Digits."""

import gzip
import io
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


def shd(folder, train=True, steps=100, max_ms=1400.0):
    """The Spiking Heidelberg Digits (Cramer et al. 2020): the digits 0-9
    spoken in English and German, 20 classes, as spikes of 700 cochlea
    channels. Spikes in the first `max_ms` are binned into `steps` steps."""
    import h5py

    stem = "shd_train.h5" if train else "shd_test.h5"
    path = Path(folder) / stem
    if path.is_file():
        f = h5py.File(path, "r")
    elif path.with_suffix(".h5.gz").is_file():
        f = h5py.File(io.BytesIO(gzip.decompress(path.with_suffix(".h5.gz").read_bytes())), "r")
    else:
        raise FileNotFoundError(f"{stem} not found in {folder}")
    with f:
        seconds, channels = f["spikes"]["times"][:], f["spikes"]["units"][:]
        labels = f["labels"][:].astype(np.uint8)
    width = max_ms / steps
    times, units = [], []
    for sec, ch in zip(seconds, channels):
        step = (sec.astype(np.float64) * 1000.0 / width).astype(np.int64)
        keep = step < steps
        times.append(step[keep])
        units.append(ch[keep].astype(np.int64))
    name = f"shd/{'train' if train else 'test'}"
    # the channels drawn as a 28 x 25 grid
    return Spikes(name, 700, steps, width, labels, times, units, (28, 25), static=False)
