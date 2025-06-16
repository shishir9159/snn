"""Random population model on the live view: shows how a Python trainer drives
snn_live. Run with `uv run --package snn-live python crates/snn-live-py/demo.py`, then open
http://127.0.0.1:7878."""

import sys

import numpy as np

import snn_live

STEPS, N_IN, N_HID, CLASSES = 100, 28 * 28, 100, 10


def main(images=2000):
    rng = np.random.default_rng(1)
    meta = dict(model="pop", dataset="noise", width=28, height=28, classes=CLASSES,
                n_train=images, n_test=0, epochs=1, steps=STEPS)
    live = snn_live.Live(
        meta,
        [("input", 28, 28, "input"), ("hidden", N_HID, "latent"), ("output", CLASSES, "output")],
        title="Model · demo",
        params=[("Steps", STEPS, "/image"), ("Backend", "numpy")],
        series=(["rate"], ["v"], ["spikes/step"]),
    )
    w = rng.normal(0, 0.05, (N_HID, N_IN)).astype(np.float32)
    hits = 0
    for i in range(images):
        label = int(rng.integers(CLASSES))
        pixels = (rng.random(N_IN) < 0.15).astype(np.uint8) * 255
        info = (0, i, images, label, True)
        probe = live.probe_request() or (1, 0)
        live.begin_image(info, pixels, (*probe, 1.0))
        v = np.zeros(N_HID, np.float32)
        counts = np.zeros(3, np.int64)
        for t in range(STEPS):
            x = rng.random(N_IN) < pixels / 255 * 0.05
            v = 0.9 * v + w @ x
            h = v > 1.0
            v[h] = 0.0
            out = rng.random(CLASSES) < 0.02
            counts += [x.sum(), h.sum(), out.sum()]
            if live.streaming:
                spikes = [np.flatnonzero(s).astype(np.uint16) for s in (x, h, out)]
                live.step(t, spikes, np.float32([h.mean()]), np.float32([v[probe[1] % N_HID]]))
            else:
                live.step(t)
        if live.streaming:
            live.map(0, w.mean(0).reshape(28, 28))
        if live.fields_due:
            units = np.arange(16, dtype=np.uint16)
            live.fields("Weights", 1, w[:16].reshape(16, 28, 28), units)
        scores = rng.random(CLASSES).astype(np.float32)
        predicted = int(scores.argmax())
        hits += predicted == label
        live.end_image(info, predicted, predicted == label, hits / (i + 1), scores,
                       counts.tolist(), np.float32([counts.sum() / STEPS]))
    live.epoch_end(0, hits / images)
    live.hold()


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
