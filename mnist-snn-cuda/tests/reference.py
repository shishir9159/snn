"""Straight NumPy port of Layer::present, the oracle for the CUDA kernels."""

import numpy as np

from snn_cuda.layer import normalize_rows, stdp_table

f32 = np.float32
KERNEL = np.array([[-0.5, -0.125, 0.25, -0.125, -0.5],
                   [-0.125, 0.25, 0.625, 0.25, -0.125],
                   [0.25, 0.625, 1.0, 0.625, 0.25],
                   [-0.125, 0.25, 0.625, 0.25, -0.125],
                   [-0.5, -0.125, 0.25, -0.125, -0.5]], f32)


def receptive_field(img, w, h):
    out = np.zeros((h, w), f32)
    for at, px in enumerate(img):
        if px == 0:
            continue
        y, x, v = at // w, at % w, f32(px) / f32(255.0)
        for ky in range(5):
            for kx in range(5):
                oy, ox = y + 2 - ky, x + 2 - kx
                if 0 <= oy < h and 0 <= ox < w:
                    out[oy, ox] += KERNEL[ky, kx] * v
    return out.ravel()


def period(pot, e, steps):
    if not pot > 0:
        return 0
    r = min(max((f32(pot) - f32(e.pot_min)) / (f32(e.pot_max) - f32(e.pot_min)), f32(0)), f32(1))
    freq = f32(e.freq_min) + r * (f32(e.freq_max) - f32(e.freq_min))
    per = max(np.ceil(f32(e.period_scale) / freq), f32(1))
    return int(per) if per <= steps else 0


def present(w, theta, img, width, height, p, learn):
    """w: (n_out, n_in) row-major, updated in place when learning."""
    n_out, n_in = w.shape
    steps, lif, s = p.time_steps, p.lif, p.stdp
    pot = receptive_field(img, width, height)
    per = [period(v, p.encode, steps) for v in pot]
    times = [list(range(pr, steps + 1, pr)) if pr else [] for pr in per]
    by_step = [[] for _ in range(steps + 2)]
    for i, ts in enumerate(times):
        for t in ts:
            by_step[t].append(i)
    base = max(f32(max(len(a) for a in by_step)) / f32(p.threshold_div), f32(1))
    thr = (base * (f32(1) + theta)).astype(f32)
    table = stdp_table(p)
    pv = np.full(n_out, lif.rest, f32)
    t_rest = np.full(n_out, -1)
    spikes = np.zeros(n_out, np.uint16)
    winner = -1
    for t in range(1, steps + 1):
        drive = np.zeros(n_out, f32)
        for i in by_step[t]:
            drive = drive + w[:, i]
        free = t > t_rest
        pv = np.where(free, pv + drive, pv).astype(f32)
        leaked = np.maximum(pv - f32(lif.leak), f32(lif.rest))
        pv = np.where(free & (pv > lif.rest), leaked, pv).astype(f32)
        pv = np.where(free & (pv < lif.floor), f32(lif.floor), pv).astype(f32)
        if winner < 0:
            best, margin = -1, f32(0)
            for j in range(n_out):
                if t > t_rest[j] and pv[j] - thr[j] >= margin:
                    margin, best = pv[j] - thr[j], j
            winner = best
            if winner >= 0:
                keep = pv[winner]
                pv[:] = lif.floor
                pv[winner] = keep
        for j in np.flatnonzero(pv >= thr):
            pv[j] = lif.rest
            t_rest[j] = t + lif.t_ref
            spikes[j] += 1
            if learn:
                for u in range(max(t - s.t_back, 1), min(t + s.t_fore, steps) + 1):
                    dw = table[u + s.t_back - t]
                    for i in by_step[u]:
                        x = w[j, i]
                        w[j, i] = (x + f32(s.sigma) * dw * (f32(s.w_max) - x) if dw > 0
                                   else x + f32(s.sigma) * dw * (x - f32(s.w_min)))
    if learn:
        if winner >= 0:
            row = w[winner]
            for i in range(n_in):
                if not times[i]:
                    row[i] = max(row[i] - f32(p.silent_penalty), f32(s.w_min))
            w[winner:winner + 1] = normalize_rows(row[None], p.w_norm, s.w_max)
        share = f32(1) / f32(n_out)
        won = (np.arange(n_out) == winner).astype(f32)
        theta[:] = np.maximum(theta + f32(p.theta_rate) * (won - share), f32(-0.9))
    return winner, spikes, base
