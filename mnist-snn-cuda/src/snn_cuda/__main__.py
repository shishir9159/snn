import argparse
import sys
import time

from . import dataset, weights
from .params import Params
from .train import Trainer


def parse(argv):
    a = argparse.ArgumentParser(prog="mnist-snn-cuda", description="WTA STDP on CUDA")
    a.add_argument("--mnist", required=True, help="MNIST idx files (plain or .gz) in DIR")
    a.add_argument("--train", type=int, help="training images")
    a.add_argument("--test", type=int, help="test images")
    a.add_argument("--epochs", type=int, default=1, help="passes over the training set")
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--outputs", type=int, default=40, help="output neurons")
    a.add_argument("--load", help="start from saved weights")
    a.add_argument("--save", help="save weights after training")
    a.add_argument("--live", action="store_true", help="serve the live view (needs snn_live)")
    a.add_argument("--port", type=int, default=7878, help="live view port")
    a.add_argument("--speed", default="1", help="multiple of real time, or max (live only)")
    return a.parse_args(argv)


class Printer:
    """A progress line every tenth of an epoch and a summary per epoch."""

    def __init__(self, total):
        self.every = max(total // 10, 1)
        self.t0 = time.perf_counter()

    def image(self, epoch, index, total, acc):
        if (index + 1) % self.every == 0:
            now = time.perf_counter()
            print(f"epoch {epoch + 1}  {index + 1:>6}/{total}  running acc {acc * 100:5.1f}%  "
                  f"{self.every / (now - self.t0):6.0f} img/s", flush=True)
            self.t0 = now

    def epoch(self, r):
        test = "-" if r["test_accuracy"] is None else f"{r['test_accuracy'] * 100:.1f}%"
        print(f"epoch {r['epoch'] + 1} done: train {r['train_accuracy'] * 100:.1f}%  "
              f"test {test}  ({r['images_per_sec']:.0f} img/s)", flush=True)
        self.t0 = time.perf_counter()


def load_data(a):
    train = dataset.mnist(a.mnist, True)
    test = dataset.mnist(a.mnist, False)
    if a.train is not None:
        train.truncate(a.train)
    if a.test is not None:
        test.truncate(a.test)
    print(f"{train.name}: {len(train)} train, {len(test)} test, {train.width}x{train.height}, "
          f"{max(train.classes, test.classes)} classes")
    return train, test


def main(argv=None):
    a = parse(argv if argv is not None else sys.argv[1:])
    train, test = load_data(a)
    classes = max(train.classes, test.classes)
    n_in = train.width * train.height
    w = weights.load(a.load) if a.load else None
    t = Trainer(n_in, w.shape[0] if w is not None else a.outputs, classes,
                Params(), a.seed, weights=w)
    t.layer.shape(train.width, train.height)
    if w is not None:
        t.assign_labels(train)
    if a.live:
        from .live import LiveRun

        speed = None if a.speed == "max" else float(a.speed)
        LiveRun(t, train, test, a.epochs, a.port, speed).fit()
        return
    p = Printer(len(train))
    t.on_image = p.image
    t.fit(train, test if len(test) else None, a.epochs, p.epoch)
    if a.save:
        weights.save(a.save, t.layer.weights())
        print(f"weights saved to {a.save}")


if __name__ == "__main__":
    main()
