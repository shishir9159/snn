import gzip
import struct

import numpy as np

from snn_cuda.dataset import mnist
from snn_cuda.rng import Rng


def test_same_seed_same_stream():
    a, b = Rng(7), Rng(7)
    assert [a.next_u64() for _ in range(50)] == [b.next_u64() for _ in range(50)]
    assert Rng(1).next_u64() != Rng(2).next_u64()


def test_rng_matches_rust_reference():
    # values printed by snn-core's Rng
    r = Rng(1)
    assert [r.next_u64() for _ in range(3)] == [
        5424204624148110235, 15555979849632202484, 6851360858507811590]
    v = list(range(10))
    Rng(5).shuffle(v)
    assert v == [7, 3, 9, 6, 2, 1, 4, 5, 0, 8]
    assert Rng(9).uniform(1, 0.0, 1.0)[0] == np.float32(0.66262805)


def test_uniform_range_and_shuffle_is_permutation():
    r = Rng(3)
    u = r.uniform(1000, 0.0, 0.4)
    assert u.dtype == np.float32 and u.min() >= 0.0 and u.max() < 0.4
    v = list(range(100))
    r.shuffle(v)
    assert sorted(v) == list(range(100)) and v != list(range(100))


def test_mnist_idx_gz(tmp_path):
    px = np.arange(2 * 4 * 3, dtype=np.uint8)
    img = struct.pack(">IIII", 0x0803, 2, 4, 3) + px.tobytes()
    lab = struct.pack(">II", 0x0801, 2) + bytes([3, 7])
    (tmp_path / "train-images-idx3-ubyte.gz").write_bytes(gzip.compress(img))
    (tmp_path / "train-labels-idx1-ubyte").write_bytes(lab)
    d = mnist(tmp_path, True)
    assert (d.width, d.height, len(d), d.classes) == (3, 4, 2, 8)
    assert d.pixels.shape == (2, 12) and list(d.labels) == [3, 7]
    d.truncate(1)
    assert len(d) == 1
