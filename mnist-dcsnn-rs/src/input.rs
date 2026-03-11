//! S1: difference-of-Gaussians filters at three scales, ON and OFF centre, then
//! the rank-order code. Strong responses spike first; the responses are sorted
//! and cut into equal bins, one per time step, as SpykeTorch's
//! Intensity2Latency does.

use crate::wave::{Shape, Wave, NEVER};

/// (window, centre sigma, surround sigma); an OFF filter swaps the sigmas.
pub const DOG: [(usize, f64, f64); 6] = [
    (3, 3.0 / 9.0, 6.0 / 9.0),
    (3, 6.0 / 9.0, 3.0 / 9.0),
    (7, 7.0 / 9.0, 14.0 / 9.0),
    (7, 14.0 / 9.0, 7.0 / 9.0),
    (13, 13.0 / 9.0, 26.0 / 9.0),
    (13, 26.0 / 9.0, 13.0 / 9.0),
];

/// A DoG kernel with its mean removed and its peak scaled to 1, row-major.
pub fn dog(size: usize, s1: f64, s2: f64) -> Vec<f32> {
    let r = (size / 2) as i64;
    let g = |s: f64, d: f64| (1.0 / (s * s)) * (-0.5 / (s * s) * d).exp();
    let mut k: Vec<f64> = (-r..=r)
        .flat_map(|y| (-r..=r).map(move |x| (x * x + y * y) as f64))
        .map(|d| (g(s1, d) - g(s2, d)) / std::f64::consts::TAU)
        .collect();
    let mean = k.iter().sum::<f64>() / k.len() as f64;
    k.iter_mut().for_each(|v| *v -= mean);
    let max = k.iter().copied().fold(f64::MIN, f64::max);
    k.iter().map(|v| (v / max) as f32).collect()
}

/// Filters an image and codes it as a spike wave.
#[derive(Debug, Clone)]
pub struct S1 {
    kernels: Vec<(usize, Vec<f32>)>,
    pub shape: Shape,
    /// responses below this are dropped
    threshold: f32,
    radius: usize,
    steps: usize,
    /// filter responses, later divided by their local mean; channel-major
    pub response: Vec<f32>,
    /// running sums of one channel, (h + 1) by (w + 1)
    sums: Vec<f64>,
    ranked: Vec<u64>,
    /// multiply-accumulates the filtering took
    pub macs: u64,
}

impl S1 {
    pub fn new(width: usize, height: usize, threshold: f32, radius: usize, steps: usize) -> Self {
        let shape = Shape::new([DOG.len(), height, width]);
        Self {
            kernels: DOG.iter().map(|&(n, a, b)| (n, dog(n, a, b))).collect(),
            shape,
            threshold,
            radius,
            steps,
            response: vec![0.0; shape.len()],
            sums: vec![0.0; (height + 1) * (width + 1)],
            ranked: Vec::new(),
            macs: 0,
        }
    }

    /// MACs of filtering a whole image densely.
    pub fn dense_macs(&self) -> u64 {
        let taps: usize = self.kernels.iter().map(|(n, _)| n * n).sum();
        (taps * self.shape.h * self.shape.w) as u64
    }

    /// Codes `pixels` into `out`.
    pub fn encode(&mut self, pixels: &[u8], out: &mut Wave) {
        self.filter(pixels);
        self.normalise();
        self.rank(out);
    }

    // correlates the image with each kernel, zero padded, scattering from lit
    // pixels; the kernels are symmetric, so a row of outputs reads a kernel row
    // forwards
    fn filter(&mut self, pixels: &[u8]) {
        let s = self.shape;
        let hw = s.h * s.w;
        assert_eq!(pixels.len(), hw);
        self.response.fill(0.0);
        self.macs = 0;
        for (at, &px) in pixels.iter().enumerate() {
            if px == 0 {
                continue;
            }
            let (y, x, v) = (at / s.w, at % s.w, px as f32);
            for (plane, (n, k)) in self.response.chunks_exact_mut(hw).zip(&self.kernels) {
                let (n, r) = (*n, n / 2);
                // outputs (y + r - ky, x + r - kx) that fall inside
                let (kx_lo, kx_hi) = ((x + r + 1).saturating_sub(s.w), (x + r).min(n - 1));
                let (ox, len) = (x + r - kx_hi, kx_hi - kx_lo + 1);
                for ky in (y + r + 1).saturating_sub(s.h)..=(y + r).min(n - 1) {
                    let krow = &k[ky * n + n - 1 - kx_hi..][..len];
                    let orow = &mut plane[(y + r - ky) * s.w + ox..][..len];
                    orow.iter_mut().zip(krow).for_each(|(o, &w)| *o += w * v);
                    self.macs += len as u64;
                }
            }
        }
        let th = self.threshold;
        self.response
            .iter_mut()
            .for_each(|v| *v = if *v < th { 0.0 } else { *v });
    }

    // divides each response by the mean of its channel over a (2r+1)² window
    fn normalise(&mut self) {
        let (s, r) = (self.shape, self.radius);
        let (w1, area) = (s.w + 1, ((2 * r + 1) * (2 * r + 1)) as f64);
        for plane in self.response.chunks_exact_mut(s.h * s.w) {
            for y in 0..s.h {
                let mut row = 0.0;
                for x in 0..s.w {
                    row += plane[y * s.w + x] as f64;
                    self.sums[(y + 1) * w1 + x + 1] = self.sums[y * w1 + x + 1] + row;
                }
            }
            for y in 0..s.h {
                let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(s.h));
                for x in 0..s.w {
                    let v = &mut plane[y * s.w + x];
                    if *v == 0.0 {
                        continue;
                    }
                    let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(s.w));
                    let sum =
                        self.sums[y1 * w1 + x1] - self.sums[y0 * w1 + x1] - self.sums[y1 * w1 + x0]
                            + self.sums[y0 * w1 + x0];
                    *v = (*v as f64 / (sum / area + 1e-12)) as f32;
                }
            }
        }
    }

    // strongest first, in equal bins of nonzero responses; the remainder never spikes
    fn rank(&mut self, out: &mut Wave) {
        let (hw, nc) = (self.shape.h * self.shape.w, self.shape.maps);
        self.ranked.clear();
        // positive floats order as their bits: largest first, then lowest unit
        let nonzero = self.response.iter().enumerate().filter(|(_, &v)| v > 0.0);
        self.ranked.extend(nonzero.map(|(i, &v)| {
            let unit = ((i % hw) * nc + i / hw) as u64;
            ((!v.to_bits() as u64) << 32) | unit
        }));
        self.ranked.sort_unstable();
        let bin = (self.ranked.len() / self.steps).max(1);
        out.time.fill(NEVER);
        for (rank, &key) in self.ranked.iter().enumerate().take(bin * self.steps) {
            out.time[key as u32 as usize] = (rank / bin) as u8;
        }
        out.index(self.steps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dog_kernels_are_centred_and_scaled() {
        for &(n, a, b) in &DOG {
            let k = dog(n, a, b);
            assert_eq!(k.len(), n * n);
            assert!(k.iter().sum::<f32>().abs() < 1e-4);
            assert_eq!(k.iter().copied().fold(f32::MIN, f32::max), 1.0);
            // symmetric, ON centre positive when the centre sigma is smaller
            let c = k[n * n / 2];
            assert_eq!(c > 0.0, a < b);
            assert_eq!(k[0], k[n * n - 1]);
            assert_eq!(k[1], k[n]);
        }
    }

    #[test]
    fn scatter_matches_direct_correlation() {
        let mut s1 = S1::new(9, 8, f32::MIN, 1, 15);
        let mut rng = snn_core::Rng::new(5);
        let img: Vec<u8> = (0..72)
            .map(|_| {
                if rng.f32() < 0.3 {
                    rng.below(256) as u8
                } else {
                    0
                }
            })
            .collect();
        s1.filter(&img);
        for c in 0..6 {
            let (n, k) = &s1.kernels[c];
            let r = (n / 2) as i64;
            for y in 0..8i64 {
                for x in 0..9i64 {
                    let mut sum = 0.0;
                    for ky in 0..*n as i64 {
                        for kx in 0..*n as i64 {
                            let (yy, xx) = (y + ky - r, x + kx - r);
                            if (0..8).contains(&yy) && (0..9).contains(&xx) {
                                sum += k[(ky * *n as i64 + kx) as usize]
                                    * img[(yy * 9 + xx) as usize] as f32;
                            }
                        }
                    }
                    let got = s1.response[(c as i64 * 72 + y * 9 + x) as usize];
                    assert!((got - sum).abs() < 1e-2, "c{c} ({y},{x}) {got} vs {sum}");
                }
            }
        }
    }

    #[test]
    fn normalising_divides_by_the_zero_padded_local_mean() {
        let mut s1 = S1::new(5, 5, 0.0, 1, 15);
        s1.response.fill(0.0);
        // channel 1, so the planes do not mix
        let at = |y: usize, x: usize| 25 + y * 5 + x;
        s1.response[at(2, 2)] = 9.0;
        s1.response[at(2, 3)] = 9.0;
        s1.response[at(0, 0)] = 3.0;
        s1.normalise();
        // 3x3 means: 18 / 9 at both, and the corner sees itself only
        assert!((s1.response[at(2, 2)] - 4.5).abs() < 1e-5);
        assert!((s1.response[at(2, 3)] - 4.5).abs() < 1e-5);
        assert!((s1.response[at(0, 0)] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn strongest_responses_spike_first_in_equal_bins() {
        let mut s1 = S1::new(4, 4, 0.0, 1, 3);
        let mut out = Wave::new(s1.shape);
        s1.response.fill(0.0);
        // seven responses of channel 0, bins of 7 / 3 = 2: the weakest never spikes
        for (pos, v) in [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0].into_iter().enumerate() {
            s1.response[pos] = v;
        }
        s1.rank(&mut out);
        assert_eq!(out.at(0), &[0, 6]);
        assert_eq!(out.at(1), &[12, 18]);
        assert_eq!(out.at(2), &[24, 30]);
        assert_eq!(out.time[36], NEVER);
        // fewer responses than steps: one per step; channel 2 at position 1 is unit 8
        s1.response.fill(0.0);
        s1.response[2 * 16 + 1] = 1.0;
        s1.response[9] = 2.0;
        s1.rank(&mut out);
        assert_eq!((out.time[54], out.time[8], out.spikes()), (0, 1, 2));
    }

    #[test]
    fn a_digit_spikes_along_its_edges() {
        let mut img = vec![0u8; 784];
        snn_core::dataset::render_digit(7, 28, &mut snn_core::Rng::new(2), &mut img);
        let mut s1 = S1::new(28, 28, 50.0, 8, 15);
        let mut out = Wave::new(s1.shape);
        s1.encode(&img, &mut out);
        assert!(out.spikes() > 100, "{}", out.spikes());
        assert!((0..15).all(|t| out.at(t).len() == out.at(0).len()));
        let blank = [0u8; 784];
        s1.encode(&blank, &mut out);
        assert_eq!(out.spikes(), 0);
    }
}
