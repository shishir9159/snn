import argparse
import sys

from . import backend, data
from .net import Params
from .train import AC_PJ, MAC_PJ, Trainer


def parse(argv):
    a = argparse.ArgumentParser(prog="mnist-sg-cuda", description="surrogate-gradient SNN")
    src = a.add_mutually_exclusive_group(required=True)
    src.add_argument("--mnist", metavar="DIR", help="MNIST idx files (plain or .gz) in DIR")
    src.add_argument("--shd", metavar="DIR", help="Spiking Heidelberg Digits (shd_*.h5[.gz]) in DIR")
    a.add_argument("--train", type=int, help="training samples")
    a.add_argument("--test", type=int, help="test samples")
    a.add_argument("--epochs", type=int, default=1, help="passes over the training set")
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--hidden", type=int, help="hidden units (default 128)")
    a.add_argument("--recurrent", action="store_true", help="recurrent hidden layer")
    a.add_argument("--batch", type=int, default=128, help="samples per gradient step")
    a.add_argument("--lr", type=float, help="Adam step size (default 1e-3)")
    a.add_argument("--steps", type=int, default=100, help="steps a sample is binned into")
    a.add_argument("--live", action="store_true", help="serve the live view (needs snn_live)")
    a.add_argument("--port", type=int, default=7878, help="live view port")
    a.add_argument("--speed", default="1", help="multiple of real time, or max (live only)")
    return a, a.parse_args(argv)


def compact(n):
    """12.3K, 4.5M, like snn_core::energy::compact."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}G"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e4:
        return f"{n / 1e3:.0f}K"
    return f"{n / 1e3:.1f}K" if n >= 1e3 else f"{n:.0f}"


def joules(nj):
    v, unit = (nj / 1e6, "mJ") if nj >= 1e6 else (nj / 1e3, "µJ") if nj >= 1e3 else (nj, "nJ")
    return f"{v:.0f} {unit}" if v >= 10 else f"{v:.2f} {unit}"


class Printer:
    """Progress every tenth of an epoch and a summary per epoch, in the Rust
    models' format."""

    def __init__(self, total, dense_macs):
        self.every, self.mark, self.dense = max(total // 10, 1), 0, dense_macs

    def batch(self, epoch, done, total, acc):
        if done // self.every > self.mark:
            self.mark = done // self.every
            print(f"epoch {epoch + 1}  {done:>6}/{total}  running acc {acc * 100:5.1f}%", flush=True)

    def epoch(self, r):
        self.mark = 0
        test = "-" if r["test_accuracy"] is None else f"{r['test_accuracy'] * 100:.1f}%"
        nj = r["synops"] * AC_PJ / 1e3
        print(f"epoch {r['epoch'] + 1} done: train {r['train_accuracy'] * 100:.1f}%  test {test}  "
              f"({r['images_per_sec']:.0f} img/s)  {compact(r['spikes'])} spikes "
              f"{compact(r['synops'])} synops / img  {joules(nj)} vs "
              f"{joules(self.dense * MAC_PJ / 1e3)} dense", flush=True)


def load(a):
    if a.shd:
        train, test = data.shd(a.shd, True, a.steps), data.shd(a.shd, False, a.steps)
    else:
        train, test = data.mnist(a.mnist, True, a.steps), data.mnist(a.mnist, False, a.steps)
    for d, n in ((train, a.train), (test, a.test)):
        if n is not None:
            d.truncate(n)
    return train, test


def main(argv=None):
    _, a = parse(sys.argv[1:] if argv is None else argv)
    train, test = load(a)
    p = Params(recurrent=a.recurrent)
    if a.hidden:
        p.hidden = a.hidden
    if a.lr:
        p.lr = a.lr
    classes = max(train.classes, test.classes)
    print(f"{train.name}: {len(train)} train, {len(test)} test, {train.n_in} inputs, "
          f"{train.steps} steps of {train.dt_ms:g} ms, {classes} classes, on {backend.name()}")
    t = Trainer(train.n_in, classes, p, a.seed, a.batch)
    if a.live:
        from .live import LiveRun

        speed = None if a.speed == "max" else float(a.speed)
        LiveRun(t, train, test, a.epochs, a.port, speed).fit()
        return
    printer = Printer(len(train), t.net.dense_macs(train.steps, train.static))
    t.on_batch = printer.batch
    t.fit(train, test, a.epochs, printer.epoch)


if __name__ == "__main__":
    main()
