"""The time loops of net.py as CUDA kernels (kernels.cu), one launch per loop
instead of several per step."""

from pathlib import Path

import cupy as cp
import numpy as np

_module = cp.RawModule(code=Path(__file__).with_name("kernels.cu").read_text(encoding="utf-8"),
                       options=("--fmad=false", "--std=c++14"))
THREADS = 128


def _run(name, count, *args):
    _module.get_function(name)(((count + THREADS - 1) // THREADS,), (THREADS,), args)


def lif_forward(drive, alpha, beta):
    """(membrane, current, spikes) per step of a layer without recurrence."""
    b, steps, n = drive.shape
    mem, syn, spk = (cp.empty((steps, b, n), cp.float32) for _ in range(3))
    _run("lif_forward", b * n, cp.ascontiguousarray(drive), mem, syn, spk,
         np.int32(b), np.int32(steps), np.int32(n), alpha, beta)
    return mem, syn, spk


def lif_backward(g_spk, mem, spk, alpha, beta, scale):
    steps, b, n = spk.shape
    g_drive = cp.empty_like(spk)
    _run("lif_backward", b * n, cp.ascontiguousarray(g_spk), mem, spk, g_drive,
         np.int32(b), np.int32(steps), np.int32(n), alpha, beta, np.float32(scale))
    return g_drive


def readout_forward(h2, alpha, beta):
    steps, b, n = h2.shape
    outs = cp.empty((steps + 1, b, n), cp.float32)
    _run("readout_forward", b * n, cp.ascontiguousarray(h2), outs,
         np.int32(b), np.int32(steps), np.int32(n), alpha, beta)
    return outs


def readout_backward(g_out, alpha, beta):
    steps, b, n = g_out.shape[0] - 1, g_out.shape[1], g_out.shape[2]
    g_h2 = cp.empty((steps, b, n), cp.float32)
    _run("readout_backward", b * n, cp.ascontiguousarray(g_out), g_h2,
         np.int32(b), np.int32(steps), np.int32(n), alpha, beta)
    return g_h2


def lif_step(mem, syn, drive_t, rec_in, mem_out, syn_out, spk_out, alpha, beta):
    """Records this step's state and spikes, then advances `mem` and `syn`."""
    _run("lif_step", mem.size, mem, syn, cp.ascontiguousarray(drive_t), rec_in,
         mem_out, syn_out, spk_out, np.int32(mem.size), alpha, beta)


def lif_step_back(g_mem, g_syn, g_s, mem, spk, g_drive_t, alpha, beta, scale):
    _run("lif_step_back", g_mem.size, g_mem, g_syn, g_s, mem, spk, g_drive_t,
         np.int32(g_mem.size), alpha, beta, np.float32(scale))
