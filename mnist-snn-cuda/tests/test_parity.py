"""The CUDA model must reproduce the Rust one: same seed and data, same learning.

Runs both command lines on a slice of MNIST and compares what they print and
the weights they save. Needs data/mnist and `cargo build --release -p mnist-snn-rs`.
"""

import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

from snn_cuda import weights

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "mnist"
SUFFIX = ".exe" if sys.platform == "win32" else ""
RUST = ROOT / "target" / "release" / f"mnist-snn-rs{SUFFIX}"

pytestmark = pytest.mark.skipif(
    not (DATA / "train-images-idx3-ubyte.gz").exists() or not RUST.exists(),
    reason="needs data/mnist and a release build of mnist-snn-rs",
)

EPOCH = re.compile(r"epoch (\d+) done: train ([\d.]+)%\s+test ([\d.]+)%")


def run(cmd):
    done = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT)
    return [m.groups() for m in EPOCH.finditer(done.stdout)]


@pytest.mark.parametrize("outputs,images,epochs", [(12, 400, 1), (40, 300, 2)])
def test_same_accuracy_and_weights(tmp_path, outputs, images, epochs):
    shared = ["--mnist", str(DATA), "--train", str(images), "--test", "100",
              "--outputs", str(outputs), "--epochs", str(epochs)]
    rust_file, cuda_file = tmp_path / "rust.txt", tmp_path / "cuda.txt"
    rust = run([str(RUST), "--headless", *shared, "--save", str(rust_file)])
    cuda = run([sys.executable, "-m", "snn_cuda", *shared, "--save", str(cuda_file)])

    assert len(rust) == epochs, "the Rust run printed no epoch summaries"
    assert rust == cuda, "accuracy per epoch differs"
    a, b = weights.load(rust_file), weights.load(cuda_file)
    assert a.shape == b.shape == (outputs, 784)
    # same arithmetic up to the order floats are added in
    assert np.abs(a - b).max() < 1e-4
