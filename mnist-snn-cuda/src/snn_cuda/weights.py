"""Plain-text weights, the same format mnist-snn-rs reads and writes."""

import numpy as np


def save(path, w):
    rows, cols = w.shape
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{rows} {cols}\n")
        for row in w:
            f.write(" ".join(repr(float(v)) for v in row) + "\n")


def load(path):
    with open(path, encoding="utf-8") as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    rows, cols = map(int, lines[0].split())
    w = np.array([l.split() for l in lines[1:]], np.float32)
    if w.shape != (rows, cols):
        raise ValueError(f"{path}: expected {rows}x{cols}, got {w.shape}")
    return w
