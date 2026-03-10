import gzip

import numpy as np
import pytest

from sg_cuda import data


def idx(tmp_path, stem, header, body):
    (tmp_path / stem).write_bytes(np.array(header, ">u4").tobytes() + bytes(body))


def test_mnist_pixels_spike_once_brighter_earlier(tmp_path):
    idx(tmp_path, "t10k-images-idx3-ubyte", [0x0803, 1, 2, 2], [255, 0, 128, 40])
    idx(tmp_path, "t10k-labels-idx1-ubyte", [0x0801, 1], [6])
    d = data.mnist(tmp_path, train=False, steps=10)
    assert (len(d), d.n_in, d.labels[0], d.static) == (1, 4, 6, True)
    # 40 is below the 20% threshold and stays silent
    assert list(d.units[0]) == [0, 2] and list(d.times[0]) == [0, 4]
    x = np.asarray(d.batch([0]))
    assert x.shape == (1, 10, 4) and x.sum() == 2 and x[0, 4, 2] == 1


def test_shd_bins_spike_times_and_reads_gzip(tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "shd_test.h5", "w") as f:
        g = f.create_group("spikes")
        times = g.create_dataset("times", (2,), dtype=h5py.vlen_dtype(np.float16))
        units = g.create_dataset("units", (2,), dtype=h5py.vlen_dtype(np.uint16))
        # the last spike falls past the 1.4 s the steps cover
        times[0] = np.array([0.0, 0.0135, 0.0145, 1.5], np.float16)
        units[0] = np.array([3, 699, 3, 5], np.uint16)
        times[1] = np.array([0.5], np.float16)
        units[1] = np.array([0], np.uint16)
        f.create_dataset("labels", data=np.array([7, 19], np.uint16))
    d = data.shd(tmp_path, train=False)
    assert (len(d), d.n_in, d.classes, d.dt_ms, d.static) == (2, 700, 20, 14.0, False)
    assert list(d.times[0]) == [0, 0, 1] and list(d.units[0]) == [3, 699, 3]
    x = np.asarray(d.batch([0, 1]))
    assert x[0, 0, 3] == x[0, 0, 699] == x[0, 1, 3] == 1 and x[0].sum() == 3
    assert x[1, 35, 0] == 1
    assert d.picture(0).shape == (28 * 25,) and d.picture(0).max() == 255
    raw = (tmp_path / "shd_test.h5").read_bytes()
    (tmp_path / "shd_test.h5").unlink()
    (tmp_path / "shd_test.h5.gz").write_bytes(gzip.compress(raw))
    assert list(data.shd(tmp_path, train=False).times[0]) == [0, 0, 1]
    with pytest.raises(FileNotFoundError):
        data.shd(tmp_path, train=True)
