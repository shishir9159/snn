//! Image → receptive-field potentials → input spike trains.

use crate::params::{Coding, EncodeParams};
use crate::rng::Rng;

/// 5x5 on-centre, off-surround kernel.
pub const KERNEL: [[f32; 5]; 5] = [
    [-0.5, -0.125, 0.25, -0.125, -0.5],
    [-0.125, 0.25, 0.625, 0.25, -0.125],
    [0.25, 0.625, 1.0, 0.625, 0.25],
    [-0.125, 0.25, 0.625, 0.25, -0.125],
    [-0.5, -0.125, 0.25, -0.125, -0.5],
];

/// Convolves a row-major grayscale image with [`KERNEL`], zero padded, pixels scaled to [0, 1].
pub fn receptive_field(pixels: &[u8], width: usize, height: usize, out: &mut Vec<f32>) {
    assert_eq!(pixels.len(), width * height);
    out.clear();
    out.resize(width * height, 0.0);
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (ky, row) in KERNEL.iter().enumerate() {
                let yy = (y + ky).wrapping_sub(2);
                if yy >= height {
                    continue;
                }
                for (kx, &k) in row.iter().enumerate() {
                    let xx = (x + kx).wrapping_sub(2);
                    if xx < width {
                        sum += k * pixels[yy * width + xx] as f32;
                    }
                }
            }
            out[y * width + x] = sum / 255.0;
        }
    }
}

/// Maps `v` from [in_lo, in_hi] onto [out_lo, out_hi], clamped at both ends.
pub fn interp(v: f32, in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32) -> f32 {
    if in_hi <= in_lo {
        return out_lo;
    }
    let r = ((v - in_lo) / (in_hi - in_lo)).clamp(0.0, 1.0);
    out_lo + r * (out_hi - out_lo)
}

/// Firing frequency for a potential, or None for potentials that stay silent.
pub fn frequency(pot: f32, p: &EncodeParams) -> Option<f32> {
    if pot > 0.0 {
        Some(interp(pot, p.pot_min, p.pot_max, p.freq_min, p.freq_max))
    } else {
        None
    }
}

/// Spike period in steps for a potential.
pub fn period(pot: f32, p: &EncodeParams) -> Option<usize> {
    frequency(pot, p).map(|f| (p.period_scale / f).ceil().max(1.0) as usize)
}

/// Input spikes of one presentation, indexed both by step and by input.
/// Steps run 1..=steps; buffers are reused between images.
#[derive(Debug, Clone, Default)]
pub struct SpikeTrain {
    n_inputs: usize,
    steps: usize,
    step_start: Vec<u32>,
    step_inputs: Vec<u16>,
    input_start: Vec<u32>,
    input_times: Vec<u16>,
}

impl SpikeTrain {
    pub fn new() -> Self {
        Self::default()
    }

    /// Encodes potentials into spikes over `steps` steps.
    pub fn encode(&mut self, pot: &[f32], steps: usize, p: &EncodeParams, rng: &mut Rng) {
        assert!(pot.len() <= u16::MAX as usize + 1 && steps <= u16::MAX as usize);
        self.n_inputs = pot.len();
        self.steps = steps;
        self.input_start.clear();
        self.input_times.clear();
        for &v in pot {
            self.input_start.push(self.input_times.len() as u32);
            match p.coding {
                Coding::Regular => {
                    if let Some(per) = period(v, p) {
                        self.input_times
                            .extend((per..=steps).step_by(per).map(|t| t as u16));
                    }
                }
                Coding::Poisson => {
                    if let Some(f) = frequency(v, p) {
                        let rate = f / p.period_scale;
                        for t in 1..=steps {
                            if rng.f32() < rate {
                                self.input_times.push(t as u16);
                            }
                        }
                    }
                }
            }
        }
        self.input_start.push(self.input_times.len() as u32);
        self.index_by_step();
    }

    // counting sort of (input, time) pairs into per-step lists
    fn index_by_step(&mut self) {
        self.step_start.clear();
        self.step_start.resize(self.steps + 2, 0);
        for &t in &self.input_times {
            self.step_start[t as usize + 1] += 1;
        }
        for t in 1..self.step_start.len() {
            self.step_start[t] += self.step_start[t - 1];
        }
        self.step_inputs.clear();
        self.step_inputs.resize(self.input_times.len(), 0);
        let mut fill: Vec<u32> = self.step_start[..self.steps + 1].to_vec();
        for (i, span) in self.input_start.windows(2).enumerate() {
            for &t in &self.input_times[span[0] as usize..span[1] as usize] {
                let slot = &mut fill[t as usize];
                self.step_inputs[*slot as usize] = i as u16;
                *slot += 1;
            }
        }
    }

    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Inputs spiking at step `t`, ascending.
    pub fn active(&self, t: usize) -> &[u16] {
        if t > self.steps {
            return &[];
        }
        &self.step_inputs[self.step_start[t] as usize..self.step_start[t + 1] as usize]
    }

    /// Spike times of input `i`, ascending.
    pub fn times(&self, i: usize) -> &[u16] {
        &self.input_times[self.input_start[i] as usize..self.input_start[i + 1] as usize]
    }

    pub fn total(&self) -> usize {
        self.input_times.len()
    }

    pub fn max_simultaneous(&self) -> usize {
        self.step_start
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regular() -> EncodeParams {
        EncodeParams::default()
    }

    #[test]
    fn blank_image_has_zero_potential() {
        let mut out = Vec::new();
        receptive_field(&[0; 28 * 28], 28, 28, &mut out);
        assert_eq!(out.len(), 784);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn single_pixel_spreads_the_kernel() {
        let mut img = [0u8; 81];
        img[4 * 9 + 4] = 255;
        let mut out = Vec::new();
        receptive_field(&img, 9, 9, &mut out);
        for ky in 0..5 {
            for kx in 0..5 {
                // kernel is symmetric, so the response equals the kernel itself
                assert_eq!(out[(2 + ky) * 9 + 2 + kx], KERNEL[ky][kx]);
            }
        }
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn corner_pixel_is_zero_padded() {
        let mut img = [0u8; 25];
        img[0] = 255;
        let mut out = Vec::new();
        receptive_field(&img, 5, 5, &mut out);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 0.625);
        assert_eq!(out[24], 0.0);
    }

    #[test]
    fn interp_clamps() {
        assert_eq!(interp(0.0, -1.0, 1.0, 0.0, 10.0), 5.0);
        assert_eq!(interp(-5.0, -1.0, 1.0, 0.0, 10.0), 0.0);
        assert_eq!(interp(5.0, -1.0, 1.0, 0.0, 10.0), 10.0);
        assert_eq!(interp(3.0, 1.0, 1.0, 2.0, 10.0), 2.0);
    }

    #[test]
    fn only_positive_potentials_spike() {
        let p = regular();
        assert_eq!(period(0.0, &p), None);
        assert_eq!(period(-0.5, &p), None);
        assert_eq!(period(10.0, &p), Some(30));
        assert!(period(0.01, &p).unwrap() > 30);
    }

    #[test]
    fn regular_train_is_periodic_and_indexed_both_ways() {
        let p = regular();
        let pot = [10.0, 0.0, 10.0, -1.0, 1.0];
        let mut tr = SpikeTrain::new();
        tr.encode(&pot, 200, &p, &mut Rng::new(0));
        assert_eq!(tr.times(0), &[30, 60, 90, 120, 150, 180]);
        assert!(tr.times(1).is_empty());
        assert!(tr.times(3).is_empty());
        assert_eq!(tr.active(30), &[0, 2]);
        assert_eq!(tr.max_simultaneous(), 2);

        let by_step: usize = (0..=200).map(|t| tr.active(t).len()).sum();
        assert_eq!(by_step, tr.total());
        for i in 0..pot.len() {
            for &t in tr.times(i) {
                assert!(tr.active(t as usize).contains(&(i as u16)));
            }
        }
    }

    #[test]
    fn encode_reuses_buffers() {
        let p = regular();
        let mut tr = SpikeTrain::new();
        tr.encode(&[10.0; 4], 200, &p, &mut Rng::new(0));
        tr.encode(&[0.0; 3], 50, &p, &mut Rng::new(0));
        assert_eq!(tr.total(), 0);
        assert_eq!(tr.n_inputs(), 3);
        assert_eq!(tr.max_simultaneous(), 0);
        assert!(tr.active(30).is_empty());
    }

    #[test]
    fn poisson_matches_regular_rate() {
        let p = EncodeParams {
            coding: Coding::Poisson,
            ..regular()
        };
        let mut tr = SpikeTrain::new();
        let pot = vec![10.0; 2000];
        tr.encode(&pot, 300, &p, &mut Rng::new(5));
        let mean = tr.total() as f32 / pot.len() as f32;
        assert!((mean - 10.0).abs() < 0.5, "mean {mean}");
    }
}
