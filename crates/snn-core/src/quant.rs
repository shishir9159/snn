//! Weights on a fixed-point grid, the way an on-chip synapse memory stores them.

use crate::rng::Rng;

/// A grid of 2^bits levels spread evenly over [lo, hi], both ends included.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantizer {
    lo: f32,
    step: f32,
    /// the top level's index
    top: f32,
    stochastic: bool,
}

impl Quantizer {
    /// `None` for 0 bits, which means full precision. Learning updates are
    /// usually smaller than a coarse step; with `stochastic` rounding a weight
    /// still moves up with a probability equal to the part of a step it gained.
    pub fn new(bits: u32, lo: f32, hi: f32, stochastic: bool) -> Option<Self> {
        if bits == 0 {
            return None;
        }
        let top = ((1u32 << bits.min(16)) - 1) as f32;
        Some(Self {
            lo,
            step: (hi - lo) / top,
            top,
            stochastic,
        })
    }

    /// Number of levels.
    pub fn levels(&self) -> u32 {
        self.top as u32 + 1
    }

    pub fn step(&self) -> f32 {
        self.step
    }

    // position of `w` on the grid, in steps, within [0, top]
    fn at(&self, w: f32) -> f32 {
        ((w - self.lo) / self.step).clamp(0.0, self.top)
    }

    fn level(&self, k: f32) -> f32 {
        self.lo + k * self.step
    }

    /// Rounds each weight to the nearest level.
    pub fn nearest(&self, row: &mut [f32]) {
        for w in row {
            *w = self.level(self.at(*w).round());
        }
    }

    /// Rounds one weight as configured; a weight already on the grid stays put and
    /// draws no random number.
    pub fn snap(&self, w: f32, rng: &mut Rng) -> f32 {
        let t = self.at(w);
        let k = t.round();
        if (t - k).abs() < 1e-4 {
            return self.level(k);
        }
        let up = if self.stochastic {
            rng.f32() < t - t.floor()
        } else {
            t - t.floor() >= 0.5
        };
        self.level(t.floor() + up as u8 as f32)
    }

    pub fn apply(&self, row: &mut [f32], rng: &mut Rng) {
        for w in row {
            *w = self.snap(*w, rng);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(q: &Quantizer) -> Vec<f32> {
        (0..q.levels()).map(|k| q.level(k as f32)).collect()
    }

    #[test]
    fn zero_bits_means_full_precision() {
        assert!(Quantizer::new(0, 0.0, 2.0, true).is_none());
    }

    #[test]
    fn levels_span_the_range_evenly() {
        let q = Quantizer::new(2, 0.0, 2.0, false).unwrap();
        let g = grid(&q);
        let want = [0.0, 2.0 / 3.0, 4.0 / 3.0, 2.0];
        assert!(
            g.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-6),
            "{g:?}"
        );
        assert_eq!(Quantizer::new(8, -1.0, 1.0, false).unwrap().levels(), 256);
    }

    #[test]
    fn nearest_snaps_and_clamps() {
        let q = Quantizer::new(2, 0.0, 2.0, false).unwrap();
        let mut row = [-1.0, 0.3, 0.4, 1.0, 1.1, 5.0];
        q.nearest(&mut row);
        let g = grid(&q);
        assert_eq!(row, [g[0], g[0], g[1], g[2], g[2], g[3]]);
    }

    #[test]
    fn weights_on_the_grid_stay_and_draw_nothing() {
        let q = Quantizer::new(4, 0.0, 2.0, true).unwrap();
        let mut row = grid(&q);
        let (mut rng, mut spare) = (Rng::new(1), Rng::new(1));
        q.apply(&mut row, &mut rng);
        assert_eq!(row, grid(&q));
        assert_eq!(rng.next_u64(), spare.next_u64(), "no number was drawn");
    }

    #[test]
    fn stochastic_rounding_is_unbiased_and_on_grid() {
        let q = Quantizer::new(2, 0.0, 2.0, true).unwrap();
        let mut rng = Rng::new(7);
        let g = grid(&q);
        let (mut sum, n) = (0.0, 20_000);
        for _ in 0..n {
            let mut w = [0.4];
            q.apply(&mut w, &mut rng);
            assert!(w[0] == g[0] || w[0] == g[1], "{}", w[0]);
            sum += w[0];
        }
        assert!(
            (sum / n as f32 - 0.4).abs() < 0.01,
            "mean {}",
            sum / n as f32
        );
    }

    #[test]
    fn plain_rounding_loses_a_small_update_that_stochastic_keeps() {
        let nearest = Quantizer::new(6, 0.0, 2.0, false).unwrap();
        let stochastic = Quantizer::new(6, 0.0, 2.0, true).unwrap();
        let (mut a, mut b, mut rng) = (1.0f32, 1.0f32, Rng::new(3));
        nearest.nearest(std::slice::from_mut(&mut a));
        stochastic.nearest(std::slice::from_mut(&mut b));
        let start = a;
        // a hundred nudges of a tenth of a step
        for _ in 0..100 {
            a += nearest.step() * 0.1;
            nearest.apply(std::slice::from_mut(&mut a), &mut rng);
            b += stochastic.step() * 0.1;
            stochastic.apply(std::slice::from_mut(&mut b), &mut rng);
        }
        assert_eq!(a, start, "round to nearest never moves");
        // the expected gain is ten steps
        assert!(b > start + 3.0 * stochastic.step(), "{b} from {start}");
    }
}
