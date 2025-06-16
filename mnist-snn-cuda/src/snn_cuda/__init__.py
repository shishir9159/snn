"""Winner-take-all STDP network (port of mnist-snn-rs) with CUDA kernels."""

import os
import sys
from pathlib import Path


def _wheel_dlls():
    # the CUDA libraries in the nvidia-* wheels are not on the Windows DLL path
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return
    for root in nvidia.__path__:
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            # nvrtc loads its builtins library with a plain LoadLibrary
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


_wheel_dlls()
