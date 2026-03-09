"""Lays out SLAYER's NMNISTsmall, 1000 training and 100 test recordings of
N-MNIST in its own 5-byte event format, the way `--nmnist DIR` reads it:
DIR/Train/<digit>/<sample>.bin and DIR/Test/<digit>/<sample>.bin. A sample's
number is its MNIST index plus one; test samples count on from 60001.

usage: nmnist_small.py DIR   (DIR holds NMNISTsmall.zip)
"""

import sys
import zipfile
from pathlib import Path


def lay_out(root):
    """Writes the recordings under `root`; returns how many went to each set."""
    counts = {}
    with zipfile.ZipFile(root / "NMNISTsmall.zip") as z:
        for listing, split in (("train1K.txt", "Train"), ("test100.txt", "Test")):
            for line in z.read(f"NMNISTsmall/{listing}").decode().splitlines():
                if not line.strip() or line.startswith("#"):
                    continue
                sample, digit = line.split()
                out = root / split / digit / f"{int(sample):05d}.bin"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(f"NMNISTsmall/{sample}.bs2"))
                counts[split] = counts.get(split, 0) + 1
    return counts


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        sys.exit(__doc__)
    counts = lay_out(Path(args[0]))
    print(", ".join(f"{n} {split}" for split, n in counts.items()), "recordings in", args[0])


if __name__ == "__main__":
    main()
