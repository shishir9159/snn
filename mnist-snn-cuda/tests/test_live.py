import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))

import pytest

pytest.importorskip("snn_live")

from snn_cuda import live
from snn_cuda.train import Trainer
from test_train import bars


def test_epoch_runs_unwatched():
    d = bars(30)
    t = Trainer(784, 6, 3, seed=1)
    t.layer.shape(28, 28)
    run = live.LiveRun(t, d, d, 1, port=7893, speed=None)
    acc, rate = run.epoch(0)
    assert 0.0 <= acc <= 1.0 and rate > 0
