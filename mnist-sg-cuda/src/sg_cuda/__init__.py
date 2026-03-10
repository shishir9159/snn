"""Spiking network trained by backpropagation through time with surrogate
gradients (Neftci, Mostafa & Zenke 2019), on CuPy or NumPy."""

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
        # CuPy's own kernels include CUDA headers; a system toolkit of another
        # major version has headers this nvrtc cannot parse
        runtime = Path(root) / "cuda_runtime"
        if (runtime / "include" / "cuda_fp16.h").is_file():
            os.environ["CUDA_PATH"] = str(runtime)


_wheel_dlls()
