"""The kernel tests need CuPy and a CUDA device; the rest runs anywhere."""

import importlib.util

GPU_TESTS = ["test_layer.py", "test_train.py", "test_live.py"]


def _has_gpu():
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


collect_ignore = [] if _has_gpu() else GPU_TESTS
