"""Replays the integer test vectors `mnist-snn-rs --vectors DIR` writes and
checks every step against them, from the files alone.

config.txt holds `name value` lines; weights.txt a row of input levels per
output neuron; gains.txt a threshold gain per output neuron. Per image NNNN:
.in has `label L` then `t i i ...` for each step with input spikes, .out has
`winner W` (-1 for none) then `t j j ...` for each step with output spikes,
.trace has every potential after each step.

Thresholds: base = max(most inputs spiking at one step * div, 1 << base_bits),
threshold[j] = (base * gain[j] + half) >> (base_bits + gain_bits). Each step t:
  1. drive[j] = offset * n + (levels of neuron j over the n inputs spiking at t)
     << frac_bits
  2. a neuron past its refractory period adds its drive, leaks toward rest
     (p = max(p - leak, rest) when above rest) and is held at the floor
  3. while there is no winner, the free neuron with the largest
     p - threshold >= 0 wins (the last one on ties); the others drop to the floor
  4. a neuron at or above its threshold fires: p = rest, refractory until
     t + t_ref
"""

import argparse
import sys
from pathlib import Path


def ints(line):
    return [int(x) for x in line.split()]


def load(directory):
    d = Path(directory)
    cfg = {}
    for line in (d / "config.txt").read_text().splitlines():
        name, value = line.split()
        cfg[name] = int(value)
    weights = [ints(line) for line in (d / "weights.txt").read_text().splitlines()]
    gains = ints((d / "gains.txt").read_text())
    return cfg, weights, gains


def steps(lines):
    """`t a b ...` lines -> {t: [a, b, ...]}"""
    out = {}
    for line in lines:
        t, *ids = ints(line)
        out[t] = ids
    return out


def simulate(cfg, weights, gains, inputs):
    """(winner or -1, {t: fired}, potentials per step) for one image."""
    n, rest, floor = cfg["outputs"], cfg["rest"], cfg["floor"]
    shift = cfg["base_bits"] + cfg["gain_bits"]
    peak = max((len(v) for v in inputs.values()), default=0)
    base = max(peak * cfg["div"], 1 << cfg["base_bits"])
    thr = [min((base * g + ((1 << shift) >> 1)) >> shift, 2**31 - 1) for g in gains]
    p, free_after = [rest] * n, [-1] * n
    winner, fired_at, trace = None, {}, []
    for t in range(1, cfg["steps"] + 1):
        active = inputs.get(t, [])
        for j in range(n):
            if t <= free_after[j]:
                continue
            drive = cfg["offset"] * len(active) + (sum(weights[j][i] for i in active) << cfg["frac_bits"])
            v = p[j] + drive
            if v > rest:
                v = max(v - cfg["leak"], rest)
            p[j] = max(v, floor)
        if winner is None:
            best, margin = None, 0
            for j in range(n):
                if t > free_after[j] and p[j] - thr[j] >= margin:
                    best, margin = j, p[j] - thr[j]
            winner = best
            if winner is not None:
                p = [v if j == winner else floor for j, v in enumerate(p)]
        fired = [j for j in range(n) if p[j] >= thr[j]]
        for j in fired:
            p[j], free_after[j] = rest, t + cfg["t_ref"]
        if fired:
            fired_at[t] = fired
        trace.append(list(p))
    return (-1 if winner is None else winner), fired_at, trace


def check(directory):
    """Problems found, one string each; empty when every image replays."""
    cfg, weights, gains = load(directory)
    problems = []
    images = sorted(Path(directory).glob("*.in"))
    for path in images:
        lines = path.read_text().splitlines()
        winner, fired, trace = simulate(cfg, weights, gains, steps(lines[1:]))
        out = path.with_suffix(".out").read_text().splitlines()
        want = [ints(line) for line in path.with_suffix(".trace").read_text().splitlines()]
        if out[0] != f"winner {winner}":
            problems.append(f"{path.stem}: {out[0]}, replay says winner {winner}")
        if steps(out[1:]) != fired:
            problems.append(f"{path.stem}: output spikes differ")
        bad = next((t for t, (a, b) in enumerate(zip(trace, want), 1) if a != b), None)
        if bad or len(trace) != len(want):
            problems.append(f"{path.stem}: potentials differ from step {bad or min(len(trace), len(want)) + 1}")
    if not images:
        problems.append(f"no *.in files in {directory}")
    return problems


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("dir", help="directory --vectors wrote")
    a = a.parse_args(argv)
    problems = check(a.dir)
    for p in problems:
        print(p)
    if problems:
        sys.exit(1)
    n = len(list(Path(a.dir).glob("*.in")))
    print(f"{n} images replay exactly: winners, output spikes and every potential")


if __name__ == "__main__":
    main()
