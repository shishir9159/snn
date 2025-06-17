"""Where a presentation spends its time on the GPU, from the kernel's -DPROFILE counters.

usage: python -m snn_cuda.profile --mnist DIR [--images N] [--outputs N]
Cycles are the winner thread's (neuron phases) and the first producer's (build); the
presentation takes as long as the slower of the two.
"""

import argparse

import cupy as cp
import numpy as np

from . import dataset, layer

PHASES = [
    (0, "slices up to the first spike"),
    (1, "winner agreement and column copy"),
    (7, "winner: drives, spikes, STDP"),
    (2, "other neurons: bound, replay"),
    (3, "normalise, theta"),
    (4, "wait for the others"),
    (5, "producer: build schedule"),
    (6, "producer: wait"),
]


def main(argv=None):
    a = argparse.ArgumentParser(prog="snn-cuda-profile")
    a.add_argument("--mnist", required=True)
    a.add_argument("--images", type=int, default=1000)
    a.add_argument("--outputs", type=int, default=40)
    a = a.parse_args(argv)

    data = dataset.mnist(a.mnist, True)
    data.truncate(a.images)
    mod = cp.RawModule(
        code=layer.KERNEL_SRC.read_text(encoding="utf-8"),
        options=("--fmad=false", "--std=c++14", "-DPROFILE"),
    )
    layer._module = mod
    lay = layer.Layer(data.width * data.height, a.outputs)
    lay.shape(data.width, data.height)
    px = cp.asarray(data.pixels)
    order = cp.arange(len(data), dtype=cp.int32)
    counters = cp.ndarray((8,), cp.uint64, mod.get_global("prof"))

    lay.present(px, order, True)  # compile, warm caches
    for learn in (True, False):
        counters[:] = 0
        cp.cuda.Stream.null.synchronize()
        t0, t1 = cp.cuda.Event(), cp.cuda.Event()
        t0.record()
        lay.present(px, order, learn)
        t1.record()
        t1.synchronize()
        us = cp.cuda.get_elapsed_time(t0, t1) * 1000 / len(data)
        print(f"{'learning' if learn else 'inference'}: {us:.1f} us/img")
        if learn:
            c = cp.asnumpy(counters).astype(float) / len(data)
            for slot, name in PHASES:
                print(f"  {name:34} {c[slot] / 1000:8.1f} kcycles/img")


if __name__ == "__main__":
    main()
