"""Seeded xorshift64*; same stream as snn-core's Rng, so a seed gives the Rust run's order."""

import numpy as np

M64 = (1 << 64) - 1


class Rng:
    def __init__(self, seed):
        z = (seed + 0x9E3779B97F4A7C15) & M64
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        self.s = (z ^ (z >> 31)) | 1

    def next_u64(self):
        x = self.s
        x ^= x >> 12
        x ^= (x << 25) & M64
        x ^= x >> 27
        self.s = x
        return (x * 0x2545F4914F6CDD1D) & M64

    def below(self, n):
        return (((self.next_u64() >> 32) * n) >> 32)

    def shuffle(self, v):
        for i in range(len(v) - 1, 0, -1):
            j = self.below(i + 1)
            v[i], v[j] = v[j], v[i]

    def uniform(self, n, lo, hi):
        """n floats in [lo, hi), rounded as f32 like the Rust model."""
        raw = np.fromiter((self.next_u64() >> 40 for _ in range(n)), np.uint64, n)
        f = raw.astype(np.float32) / np.float32(1 << 24)
        lo, hi = np.float32(lo), np.float32(hi)
        return lo + (hi - lo) * f
