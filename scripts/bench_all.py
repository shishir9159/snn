"""Benchmarks every Rust model and prints one report.

Builds the profile examples and model binaries, then
  * times each model's phases, best of N rounds, and
  * trains each model on a slice of MNIST for accuracy, spikes, synaptic
    operations and energy per image, once per seed, since accuracy after a
    few thousand images moves by several points from seed to seed.
`--save DIR` keeps the built executables; later `--against DIR` runs them and
the fresh build interleaved in one session, so CPU frequency drift hits both
alike. Results also go to bench-results/ (not tracked) for comparing runs.
"""

import argparse
import csv
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ".exe" if sys.platform == "win32" else ""
# name, crate, profile example, images the profile runs over, epochs of the
# quality run (PC-SNN learns from the labels over several passes)
MODELS = [
    ("wta", "mnist-snn-rs", "profile", 3000, 1),
    ("spncn", "mnist-spncn-rs", "spncn-profile", 300, 1),
    ("pcsnn", "mnist-pcsnn-rs", "pcsnn-profile", 300, 8),
    ("pcl", "mnist-pcl-rs", "pcl-profile", 600, 1),
]
PHASE = re.compile(r"^(.+?)\s+([\d.]+) us/img", re.M)
EPOCH = re.compile(
    r"train ([\d.]+)%\s+test ([\d.]+)%\s+\(([\d.]+) img/s\)"
    r"(?:\s+([\d.]+[KMG]?) spikes ([\d.]+[KMG]?) synops(?: ([\d.]+[KMG]?) MACs)? / img"
    r"\s+([\d.]+ [nµm]J)(?: vs ([\d.]+ [nµm]J) dense)?)?"
)


def expand(count):
    """'7.7K' -> 7700.0"""
    scale = {"K": 1e3, "M": 1e6, "G": 1e9}.get(count[-1:], 1)
    return float(count.rstrip("KMG")) * scale


def nanojoules(text):
    """'1.50 µJ' -> 1500.0"""
    value, unit = text.split()
    return float(value) * {"nJ": 1.0, "µJ": 1e3, "mJ": 1e6}[unit]


def parse_phases(text):
    """Phase name -> microseconds per image from a profile example's output."""
    return {name.strip(): float(us) for name, us in PHASE.findall(text)}


def parse_epoch(text):
    """The last epoch summary of a headless run, or None."""
    found = EPOCH.findall(text)
    if not found:
        return None
    train, test, rate, spikes, synops, macs, energy, dense = found[-1]
    count = lambda c: expand(c) if c else 0.0  # noqa: E731
    return {
        "train %": float(train),
        "test %": float(test),
        "img/s": float(rate),
        "spikes/img": count(spikes),
        "synops/img": count(synops),
        "MACs/img": count(macs),
        "nJ/img": nanojoules(energy) if energy else 0.0,
        "dense nJ": nanojoules(dense) if dense else 0.0,
    }


def run(cmd):
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if done.returncode:
        sys.exit(f"{' '.join(map(str, cmd))} failed:\n{done.stderr[-2000:]}")
    return done.stdout


def build(models):
    crates = [m[1] for m in models]
    pick = [arg for c in crates for arg in ("-p", c)]
    run(["cargo", "build", "--release", "--examples", "--bins", *pick])


def example(name, directory=None):
    return Path(directory or ROOT / "target" / "release" / "examples") / f"{name}{EXE}"


def best_of(rounds, runs):
    """Fastest time per phase over rounds; `runs` maps a label to a function."""
    best = {label: {} for label in runs}
    for _ in range(rounds):
        for label, go in runs.items():
            for phase, us in go().items():
                best[label][phase] = min(us, best[label].get(phase, us))
    return best


def timings(models, mnist, rounds, against):
    """(model, phase, us/img, base us/img or None) rows."""
    rows = []
    for name, _, prof, images, _ in models:
        def go(exe, images=images):
            return lambda: parse_phases(run([exe, mnist, str(images)]))

        runs = {"new": go(example(prof))}
        if against:
            runs = {"base": go(example(prof, against)), **runs}
        best = best_of(rounds, runs)
        for phase, us in best["new"].items():
            rows.append((name, phase, us, best.get("base", {}).get(phase)))
    return rows


def quality(models, mnist, train, test, seeds, jobs, epochs=None):
    """(model, epochs, last epoch's figures per seed) per model. The runs go in
    parallel: they measure learning, the phases measure speed."""
    runs = []
    for name, crate, _, _, own in models:
        exe = ROOT / "target" / "release" / f"{crate}{EXE}"
        n = epochs or own
        cmd = [exe, "--headless", "--mnist", mnist, "--train", str(train), "--test", str(test)]
        runs += [(name, n, [*cmd, "--epochs", str(n), "--seed", str(s)]) for s in seeds]
    with ThreadPoolExecutor(jobs) as pool:
        figures = list(pool.map(lambda r: parse_epoch(run(r[2])), runs))
    rows = {}
    for (name, n, cmd), f in zip(runs, figures):
        if f is None:
            sys.exit(f"{' '.join(map(str, cmd))} printed no epoch summary")
        rows.setdefault((name, n), []).append(f)
    return [(name, n, fs) for (name, n), fs in rows.items()]


def summary(figures):
    """Figure -> (mean, sample standard deviation or None) over the seeds."""
    out = {}
    for k in figures[0]:
        vs = [f[k] for f in figures]
        out[k] = (statistics.fmean(vs), statistics.stdev(vs) if len(vs) > 1 else None)
    return out


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def report(time_rows, quality_rows, rounds, train, test, seeds):
    print(f"\nphases, us per image, best of {rounds} (lower is better)")
    for name, phase, us, base in time_rows:
        change = f"  {100 * (us - base) / base:+6.1f}% vs {base:.1f}" if base else ""
        print(f"  {name:<6}{phase:<22}{us:>11.1f}{change}")
    span = f"seed {seeds[0]}" if len(seeds) == 1 else f"mean ± std over seeds {seeds[0]}-{seeds[-1]}"
    print(f"\n{train} training and {test} test images, last epoch, {span}")
    print("energy: 0.9 pJ per synop, 4.6 pJ per MAC; x dense is against an ANN with the same layers")
    head = ("model", "epochs", "train %", "test %", "spikes/img", "synops/img", "MACs/img",
            "nJ/img", "x dense")
    widths = [8, 8, 14, 14, 12, 12, 12, 10, 8]
    print("  " + "".join(f"{h:<{w}}" for h, w in zip(head, widths)))
    for name, epochs, figures in quality_rows:
        q = summary(figures)
        dense = q["dense nJ"][0]
        q["x dense"] = (q["nJ/img"][0] / dense if dense else None, None)
        cells = [name, epochs, *(cell(k, *q[k]) for k in head[2:])]
        print("  " + "".join(f"{c:<{w}}" for c, w in zip(cells, widths)))


def cell(key, mean, std=None):
    if mean is None:
        return "-"
    if "%" in key:
        return f"{mean:.1f}" + (f" ± {std:.1f}" if std is not None else "")
    return f"{mean:.2f}" if key == "x dense" else f"{mean:,.0f}"


def save_csv(time_rows, quality_rows, seeds):
    out = ROOT / "bench-results"
    out.mkdir(exist_ok=True)
    commit = git("rev-parse", "--short", "HEAD") + ("+dirty" if git("status", "--porcelain") else "")
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{commit}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["commit", "section", "model", "metric", "value", "seed"])
        for name, phase, us, _ in time_rows:
            w.writerow([commit, "phase us/img", name, phase, f"{us:.2f}", ""])
        for name, epochs, figures in quality_rows:
            w.writerow([commit, "quality", name, "epochs", epochs, ""])
            for seed, q in zip(seeds, figures):
                for metric, v in q.items():
                    w.writerow([commit, "quality", name, metric, f"{v:.2f}", seed])
            for metric, (mean, std) in summary(figures).items():
                w.writerow([commit, "quality", name, metric, f"{mean:.2f}", "mean"])
                if std is not None:
                    w.writerow([commit, "quality", name, metric, f"{std:.2f}", "std"])
    return path


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--mnist", default="data/mnist", help="MNIST idx files")
    a.add_argument("--rounds", type=int, default=5, help="timing rounds, best is kept (default 5)")
    a.add_argument("--train", type=int, default=3000, help="training images for the quality run")
    a.add_argument("--test", type=int, default=500, help="test images for the quality run")
    a.add_argument("--epochs", type=int, help="epochs of the quality run (default: per model)")
    a.add_argument("--seeds", type=int, default=5, help="quality runs per model, seeds 1..N (default 5)")
    a.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="quality runs at once (default: half the CPUs)")
    a.add_argument("--only", help="comma separated models: " + ", ".join(m[0] for m in MODELS))
    a.add_argument("--save", metavar="DIR", help="copy the built profile examples to DIR and stop")
    a.add_argument("--against", metavar="DIR", help="also time the examples saved in DIR, interleaved")
    a = a.parse_args(argv)
    if a.rounds < 5:
        print("note: fewer than 5 rounds, treat the timings as rough", file=sys.stderr)

    models = [m for m in MODELS if not a.only or m[0] in a.only.split(",")]
    if not models:
        sys.exit(f"--only names none of {[m[0] for m in MODELS]}")
    build(models)
    if a.save:
        Path(a.save).mkdir(parents=True, exist_ok=True)
        for prof in (m[2] for m in models):
            shutil.copy(example(prof), example(prof, a.save))
        print(f"saved {len(models)} examples to {a.save}")
        return
    if a.against:
        missing = [m[2] for m in models if not example(m[2], a.against).is_file()]
        if missing:
            sys.exit(f"{a.against} lacks {missing}; make it with --save")

    mnist = str(Path(a.mnist))
    time_rows = timings(models, mnist, a.rounds, a.against)
    seeds = list(range(1, max(1, a.seeds) + 1))
    quality_rows = quality(models, mnist, a.train, a.test, seeds, a.jobs, a.epochs)
    report(time_rows, quality_rows, a.rounds, a.train, a.test, seeds)
    print(f"\nwritten to {save_csv(time_rows, quality_rows, seeds).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
