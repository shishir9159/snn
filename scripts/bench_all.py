"""Benchmarks every Rust model and prints one report.

Builds the profile examples and model binaries, then
  * times each model's phases, best of N rounds, and
  * trains each model on a slice of MNIST for accuracy, spikes, synaptic
    operations and energy per image, once per seed, since accuracy after a
    few thousand images moves by several points from seed to seed.
`--robustness` also tests each trained model under pixel noise, dropped and
jittered spikes, and `--early-exit` measures accuracy against the spikes spent
when answers are cut short; both draw their curves. `--save DIR` keeps the built executables;
later `--against DIR` runs them and the fresh build interleaved in one session,
so CPU frequency drift hits both alike. Results also go to bench-results/ (not
tracked) for comparing runs.
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


ROBUST = re.compile(r"^robustness: (.+?) ((?:[\d.]+ [\d.]+%\s*)+)$", re.M)
CLEAN = re.compile(r"^robustness: clean ([\d.]+)%", re.M)
SURE = re.compile(r"^early exit: sure ([\d.]+)% at ([\d.]+)% of the steps and ([\d.]+)% of the spikes", re.M)


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


def parse_early_exit(text):
    """{'exit acc 1'..'exit acc 10', 'exit spikes 1'.., 'exit sure acc/steps/spikes': %}."""
    out = {}
    for what in ("accuracy", "spikes"):
        m = re.search(rf"^early exit: {what} ((?:[\d.]+% ?)+)$", text, re.M)
        if m:
            key = "acc" if what == "accuracy" else what
            for k, v in enumerate(re.findall(r"([\d.]+)%", m.group(1)), 1):
                out[f"exit {key} {k}"] = float(v)
    m = SURE.search(text)
    if m:
        out.update(zip(("exit sure acc", "exit sure steps", "exit sure spikes"), map(float, m.groups())))
    return out


def parse_robustness(text):
    """{'robust clean': %, 'robust <kind> <level>': %, ...} from --robustness lines."""
    out = {}
    clean = CLEAN.search(text)
    if clean:
        out["robust clean"] = float(clean.group(1))
    for kind, cells in ROBUST.findall(text):
        for level, acc in re.findall(r"([\d.]+) ([\d.]+)%", cells):
            out[f"robust {kind} {float(level):g}"] = float(acc)
    return out


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


def quality(models, mnist, train, test, seeds, jobs, epochs=None, robust=False, exits=False):
    """(model, epochs, last epoch's figures per seed) per model. The runs go in
    parallel: they measure learning, the phases measure speed."""
    runs = []
    for name, crate, _, _, own in models:
        exe = ROOT / "target" / "release" / f"{crate}{EXE}"
        n = epochs or own
        cmd = [exe, "--headless", "--mnist", mnist, "--train", str(train), "--test", str(test)]
        cmd += ["--robustness"] if robust else []
        cmd += ["--early-exit"] if exits else []
        runs += [(name, n, [*cmd, "--epochs", str(n), "--seed", str(s)]) for s in seeds]

    def go(r):
        text = run(r[2])
        f = parse_epoch(text)
        return f and {**f, **parse_robustness(text), **parse_early_exit(text)}

    with ThreadPoolExecutor(jobs) as pool:
        figures = list(pool.map(go, runs))
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


def robustness_curves(quality_rows):
    """{model: {kind: [(level, mean %, std % or None)]}}, clean as level 0."""
    out = {}
    for name, _, figures in quality_rows:
        s = summary(figures)
        if "robust clean" not in s:
            continue
        curves = {}
        for key, (mean, std) in s.items():
            if key.startswith("robust ") and key != "robust clean":
                kind, level = key[len("robust "):].rsplit(" ", 1)
                curves.setdefault(kind, [(0.0, *s["robust clean"])]).append((float(level), mean, std))
        out[name] = curves
    return out


def report_robustness(curves):
    print("\nrobustness: test accuracy %, mean over seeds")
    kinds = list(dict.fromkeys(k for c in curves.values() for k in c))
    for kind in kinds:
        levels = next(c[kind] for c in curves.values() if kind in c)
        print(f"  {kind:<16}" + "".join(f"{lv:>8g}" for lv, _, _ in levels))
        for name, c in curves.items():
            cells = [f"{m:>8.1f}" for _, m, _ in c.get(kind, [])]
            print(f"    {name:<14}" + "".join(cells))


# validated categorical slots 1-4, light and dark (dataviz reference palette)
SERIES = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"), ("#eda100", "#c98500")]
AXES = {"pixel noise": "noise σ, share of full brightness", "event dropout": "share of spikes dropped",
        "timing jitter": "jitter σ, ms"}


def lines_svg(title, note, panels, path):
    """Line charts side by side, sharing a 0-100% y axis and a legend. A panel is
    (name, x axis label, [(x 0-1, tick label)], {series: [(x 0-1, y %, tooltip)]},
    {series: [(x, y, tooltip)]} drawn as diamonds)."""
    models = list(dict.fromkeys(m for panel in panels for m in panel[3]))[: len(SERIES)]
    pw, ph, gap, left, top, bottom = 280 if len(panels) > 1 else 520, 200, 36, 40, 112, 48
    width = left + len(panels) * (pw + gap)
    height = top + ph + bottom
    var = lambda mode: "".join(f"--s{i}:{c[mode]};" for i, c in enumerate(SERIES))  # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}">',
           "<style>",
           f"svg{{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;{var(0)}}}",
           "@media (prefers-color-scheme: dark){svg{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;"
           f"--grid:#2c2c2a;--axis:#383835;{var(1)}}}}}",
           'text{font:12px system-ui,-apple-system,"Segoe UI",sans-serif;fill:var(--ink2)}',
           ".t{fill:var(--ink);font-size:14px;font-weight:600}.m{fill:var(--muted);font-size:11px}",
           ".g{stroke:var(--grid);stroke-width:1}.b{stroke:var(--axis);stroke-width:1}",
           ".l{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}",
           ".d{stroke:var(--surface);stroke-width:2}.h{fill:transparent}",
           "</style>",
           f'<rect width="{width}" height="{height}" fill="var(--surface)"/>',
           f'<text class="t" x="16" y="24">{title}</text>',
           f'<text class="m" x="16" y="42">{note}</text>']
    x = 16
    for i, name in enumerate(models):
        out.append(f'<line x1="{x}" y1="64" x2="{x + 18}" y2="64" class="l" stroke="var(--s{i})"/>')
        out.append(f'<text x="{x + 24}" y="68">{name}</text>')
        x += 24 + 8 * len(name) + 24
    y = lambda acc: top + ph * (1 - acc / 100)  # noqa: E731

    def point(i, px, py, tip, r=4, diamond=False):
        mark = (f'<rect class="d" fill="var(--s{i})" x="{px - r:.1f}" y="{py - r:.1f}" width="{2 * r}" '
                f'height="{2 * r}" transform="rotate(45 {px:.1f} {py:.1f})"/>' if diamond else
                f'<circle class="d" fill="var(--s{i})" cx="{px:.1f}" cy="{py:.1f}" r="{r}"/>')
        return f'<g><title>{tip}</title><circle class="h" cx="{px:.1f}" cy="{py:.1f}" r="10"/>{mark}</g>'

    for k, (name, axis, ticks, series, marks) in enumerate(panels):
        x0 = left + k * (pw + gap)
        px = lambda f: x0 + f * pw  # noqa: E731
        out.append(f'<text class="t" x="{x0}" y="{top - 14}">{name}</text>')
        for a in (0, 25, 50, 75, 100):
            out.append(f'<line class="{"b" if a == 0 else "g"}" x1="{x0}" x2="{x0 + pw}" y1="{y(a)}" y2="{y(a)}"/>')
            out.append(f'<text class="m" x="{x0 - 6}" y="{y(a) + 4}" text-anchor="end">{a}%</text>')
        for f, label in ticks:
            out.append(f'<text class="m" x="{px(f):.1f}" y="{top + ph + 18}" text-anchor="middle">{label}</text>')
        out.append(f'<text class="m" x="{x0 + pw / 2}" y="{top + ph + 38}" text-anchor="middle">{axis}</text>')
        for i, model in enumerate(models):
            pts = series.get(model)
            if pts:
                line = " ".join(f"{px(f):.1f},{y(v):.1f}" for f, v, _ in pts)
                out.append(f'<polyline class="l" stroke="var(--s{i})" points="{line}"/>')
                out += [point(i, px(f), y(v), tip) for f, v, tip in pts]
            out += [point(i, px(f), y(v), tip, 5, True) for f, v, tip in marks.get(model, [])]
    out.append("</svg>")
    path.write_text("\n".join(out), encoding="utf-8")


def robustness_svg(curves, path):
    """Small multiples, one panel per kind of damage, a line per model."""
    panels = []
    for kind in dict.fromkeys(k for c in curves.values() for k in c):
        levels = [lv for lv, _, _ in next(c[kind] for c in curves.values() if kind in c)]
        at = {lv: j / max(1, len(levels) - 1) for j, lv in enumerate(levels)}
        series = {}
        for model, c in curves.items():
            spread = lambda sd: f" ± {sd:.1f}" if sd is not None else ""  # noqa: E731
            series[model] = [(at[lv], m, f"{model} · {kind} {lv:g}: {m:.1f}%{spread(sd)}")
                             for lv, m, sd in c.get(kind, [])]
        ticks = [(f, f"{lv:g}") for lv, f in at.items()]
        panels.append((kind, AXES.get(kind, kind), ticks, series, {}))
    lines_svg("Test accuracy under input damage", "mean over seeds; hover a point for its spread", panels, path)


def early_exit_curves(quality_rows):
    """{model: ([(spikes %, accuracy %)] per checkpoint, (sure accuracy, steps %, spikes %))}."""
    out = {}
    for name, _, figures in quality_rows:
        s = summary(figures)
        if "exit sure acc" not in s:
            continue
        points = [(s[f"exit spikes {k}"][0], s[f"exit acc {k}"][0]) for k in range(1, 11)]
        out[name] = (points, tuple(s[f"exit sure {k}"][0] for k in ("acc", "steps", "spikes")))
    return out


def report_early_exit(curves):
    print("\nearly exit: accuracy % and spikes spent % at each tenth of a test presentation, mean over seeds")
    print(f"  {'steps %':<16}" + "".join(f"{10 * k:>7}" for k in range(1, 11)) + "   sure exit")
    for name, (points, (acc, steps, spikes)) in curves.items():
        print(f"  {name + ' acc':<16}" + "".join(f"{a:>7.1f}" for _, a in points)
              + f"   {acc:.1f} at {steps:.0f}% of steps")
        print(f"  {'  spikes':<16}" + "".join(f"{sp:>7.1f}" for sp, _ in points)
              + f"   {spikes:.0f}% of spikes")


def early_exit_svg(curves, path):
    series, marks = {}, {}
    for name, (points, (acc, steps, spikes)) in curves.items():
        series[name] = [(sp / 100, a, f"{name} · {10 * k}% of the steps: {a:.1f}% with {sp:.0f}% of the spikes")
                        for k, (sp, a) in enumerate(points, 1)]
        marks[name] = [(spikes / 100, acc, f"{name} · sure exit at {steps:.0f}% of the steps: "
                                           f"{acc:.1f}% with {spikes:.0f}% of the spikes")]
    ticks = [(f / 100, f"{f}%") for f in (0, 25, 50, 75, 100)]
    panel = ("accuracy against spikes spent", "share of the full presentation's spikes", ticks, series, marks)
    lines_svg("Answering early", "a point per tenth of the steps, ◆ where the model is sure; mean over seeds",
              [panel], path)


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
    a.add_argument("--robustness", action="store_true",
                   help="also test under input damage and draw the curves")
    a.add_argument("--early-exit", action="store_true",
                   help="also measure answers cut short and draw accuracy against spikes")
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
    quality_rows = quality(models, mnist, a.train, a.test, seeds, a.jobs, a.epochs, a.robustness,
                           a.early_exit)
    report(time_rows, quality_rows, a.rounds, a.train, a.test, seeds)
    path = save_csv(time_rows, quality_rows, seeds)
    print(f"\nwritten to {path.relative_to(ROOT)}")
    if a.robustness:
        curves = robustness_curves(quality_rows)
        report_robustness(curves)
        chart = path.with_name(path.stem + "-robustness.svg")
        robustness_svg(curves, chart)
        print(f"curves drawn in {chart.relative_to(ROOT)}")
    if a.early_exit:
        curves = early_exit_curves(quality_rows)
        report_early_exit(curves)
        chart = path.with_name(path.stem + "-early-exit.svg")
        early_exit_svg(curves, chart)
        print(f"curves drawn in {chart.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
