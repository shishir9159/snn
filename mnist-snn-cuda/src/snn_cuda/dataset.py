"""MNIST IDX files, plain or gzipped."""

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Dataset:
    name: str
    width: int
    height: int
    pixels: np.ndarray  # (n, width*height) uint8
    labels: np.ndarray  # (n,) uint8

    def __len__(self):
        return len(self.labels)

    @property
    def classes(self):
        return int(self.labels.max()) + 1 if len(self) else 0

    def truncate(self, n):
        self.pixels, self.labels = self.pixels[:n], self.labels[:n]


def _read(path):
    raw = Path(path).read_bytes()
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def _find(folder, stems):
    for stem in stems:
        for ext in ("", ".gz"):
            p = Path(folder) / (stem + ext)
            if p.is_file():
                return p
    raise FileNotFoundError(f"{stems[0]} not found in {folder}")


def parse_images(b):
    magic, n, rows, cols = np.frombuffer(b, ">u4", 4)
    if magic != 0x0803:
        raise ValueError("not an idx3 image file")
    body = np.frombuffer(b, np.uint8, n * rows * cols, 16)
    return int(rows), int(cols), body.reshape(n, rows * cols)


def parse_labels(b):
    magic, n = np.frombuffer(b, ">u4", 2)
    if magic != 0x0801:
        raise ValueError("not an idx1 label file")
    return np.frombuffer(b, np.uint8, n, 8)


def mnist(folder, train=True):
    s = "train" if train else "t10k"
    names = lambda kind, idx: [f"{s}-{kind}-{idx}-ubyte", f"{s}-{kind}.{idx}-ubyte"]
    rows, cols, px = parse_images(_read(_find(folder, names("images", "idx3"))))
    labels = parse_labels(_read(_find(folder, names("labels", "idx1"))))
    if len(labels) != len(px):
        raise ValueError("mnist image and label counts differ")
    return Dataset(f"mnist/{s}", cols, rows, px, labels)
