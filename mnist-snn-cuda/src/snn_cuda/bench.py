"""Interleaved benchmark: Rust CPU profile vs this CUDA port, best of N rounds.

Build the Rust profiler first (`cargo build --release -p mnist-snn-rs --examples`);
both sides run on the same images in every round, so clock drift hits them alike.
Rust numbers add receptive field and encode to `present`, as CUDA does them on device.
"""

import argparse
import re
import subprocess
import time
from pathlib import Path

import cupy as cp
import numpy as np

from . import dataset
from .train import Trainer


def rust(exe, mnist, n):
    out = subprocess.run([str(Path(exe).resolve()), mnist, str(n)], check=True, capture_output=True, text=True).stdout
    us = {m[1].strip(): float(m[2]) for m in re.finditer(r"^(.+?)\s+([\d.]+) us/img", out, re.M)}
    enc = us["receptive field"] + us["encode"]
    return 1e6 / (enc + us["present"]), 1e6 / (enc + us["present + learning"])


def cuda(data, n_out):
    t = Trainer(784, n_out, 10, seed=1)
    t.layer.shape(data.width, data.height)
    t0 = time.perf_counter()
    t.train_epoch(data)
    cp.cuda.Stream.null.synchronize()
    train = len(data) / (time.perf_counter() - t0)
    t0 = time.perf_counter()
    t.evaluate(data)
    return len(data) / (time.perf_counter() - t0), train


def main(argv=None):
    a = argparse.ArgumentParser(prog="snn-cuda-bench")
    a.add_argument("--mnist", required=True)
    a.add_argument("--exe", default="target/release/examples/profile.exe")
    a.add_argument("--images", type=int, default=6000)
    a.add_argument("--outputs", type=int, default=40)
    a.add_argument("--rounds", type=int, default=5)
    a = a.parse_args(argv)
    if not Path(a.exe).is_file():
        raise SystemExit(f"{a.exe} not found; build the Rust profiler first")
    data = dataset.mnist(a.mnist, True)
    data.truncate(a.images)
    cuda(data, a.outputs)  # warm-up: compile, upload
    best = np.zeros((2, 2))  # (rust, cuda) x (eval, train)
    for r in range(a.rounds):
        rs, cs = rust(a.exe, a.mnist, a.images), cuda(data, a.outputs)
        best[0] = np.maximum(best[0], rs)
        best[1] = np.maximum(best[1], cs)
        print(f"round {r + 1}: rust {rs[0]:.0f}/{rs[1]:.0f}  cuda {cs[0]:.0f}/{cs[1]:.0f} img/s", flush=True)
    print(f"\nbest of {a.rounds} (img/s)      eval      train")
    for name, row in zip(("rust cpu", "cuda"), best):
        print(f"{name:20} {row[0]:9.0f} {row[1]:9.0f}")
    print(f"{'cuda / rust':20} {best[1][0] / best[0][0]:9.2f} {best[1][1] / best[0][1]:9.2f}")


if __name__ == "__main__":
    main()
