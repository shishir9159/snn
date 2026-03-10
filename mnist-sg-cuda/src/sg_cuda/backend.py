"""CuPy on a CUDA device, NumPy otherwise; SG_CPU=1 forces NumPy."""

import os

import numpy as np

try:
    if os.environ.get("SG_CPU"):
        raise ImportError
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() == 0:
        raise ImportError
    xp = cp
except Exception:
    xp = np


def host(a):
    """`a` as a NumPy array."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


def name():
    return "CuPy + CUDA" if xp is not np else "NumPy"
