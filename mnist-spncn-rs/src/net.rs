//! Spiking neural coding network. Each latent layer predicts the layer below
//! through W; error units hold activity minus prediction; error synapses E carry
//! the mismatch back up into the latent currents. Learning is ST-LRA: every
//! spike moves its unit's W column and E row along the error below.
//!
//! Departures from the paper, each needed for stable learning here: predictions
//! come straight from spike traces (`W·z`, no LIF prediction units), so a spike
//! explains its part of the input at once; weight vectors are held at a fixed
//! norm rather than bounded; E starts as Wᵀ; thresholds adapt toward a target
//! rate.

use crate::lif::Lif;
use crate::params::Params;
use snn_core::{weights, Rng, SpikeTrain};

#[derive(Debug, Clone)]
pub struct Net {
    pub p: Params,
    /// input size, then each latent layer's
    pub sizes: Vec<usize>,
    /// W of layer l+1, column-major: unit j predicts layer l with `w[l][j*n..(j+1)*n]`
    pub w: Vec<Vec<f32>>,
    /// E of layer l+1, row-major: unit j hears layer l's errors through `e[l][j*n..(j+1)*n]`
    pub e: Vec<Vec<f32>>,
    /// input spike trace
    pub zx: Vec<f32>,
    /// input spikes this step
    pub input: Vec<u16>,
    pub lat: Vec<Lif>,
    /// prediction of each layer below the top
    pub pred: Vec<Vec<f32>>,
    /// error of each layer below the top: activity minus prediction
    pub err: Vec<Vec<f32>>,
    /// mean squared error of each layer in `err` this step
    pub energy: Vec<f32>,
    /// mean squared input trace this step, what the input error would be unexplained
    pub input_energy: f32,
    drive: Vec<f32>,
}

impl Net {
    pub fn new(n_in: usize, p: Params, rng: &mut Rng) -> Self {
        let mut sizes = vec![n_in];
        sizes.extend(&p.layers);
        let w: Vec<Vec<f32>> = sizes
            .windows(2)
            .map(|s| {
                let mut w = weights::random(s[0] * s[1], -1.0, 1.0, rng);
                w.chunks_exact_mut(s[0]).for_each(|c| set_norm(c, p.norm));
                w
            })
            .collect();
        let below = &sizes[..sizes.len() - 1];
        let zeros = || below.iter().map(|&n| vec![0.0; n]).collect::<Vec<_>>();
        Self {
            // same layout, so a copy is the transpose
            e: w.clone(),
            w,
            zx: vec![0.0; n_in],
            input: Vec::new(),
            lat: p.layers.iter().map(|&n| Lif::new(n, p.v_thr)).collect(),
            pred: zeros(),
            err: zeros(),
            energy: vec![0.0; below.len()],
            input_energy: 0.0,
            drive: vec![0.0; sizes.iter().copied().max().unwrap()],
            sizes,
            p,
        }
    }

    /// Clears all neuron state before an image.
    pub fn reset(&mut self) {
        self.zx.fill(0.0);
        self.input.clear();
        self.lat.iter_mut().for_each(Lif::reset);
        self.pred.iter_mut().for_each(|x| x.fill(0.0));
        self.err.iter_mut().for_each(|x| x.fill(0.0));
    }

    /// Simulates step `t` of a presentation; with `learn`, applies ST-LRA.
    pub fn step(&mut self, t: usize, train: &SpikeTrain, learn: bool) {
        let p = &self.p;
        let decay = p.trace_decay();
        self.input.clear();
        self.input.extend_from_slice(train.active(t));
        self.zx.iter_mut().for_each(|z| *z *= decay);
        for &i in &self.input {
            self.zx[i as usize] += 1.0;
        }

        // latents: last step's errors from below through E, minus their own
        for l in 0..self.lat.len() {
            let (n, below) = (self.sizes[l + 1], self.sizes[l]);
            let drive = &mut self.drive[..n];
            for (d, row) in drive.iter_mut().zip(self.e[l].chunks_exact(below)) {
                *d = dot(row, &self.err[l]);
            }
            if let Some(own) = self.err.get(l + 1) {
                drive.iter_mut().zip(own).for_each(|(d, &e)| *d -= e);
            }
            self.lat[l].step(drive, p);
        }

        self.errors();
        if learn {
            self.learn();
        }
    }

    fn errors(&mut self) {
        for l in 0..self.err.len() {
            let n = self.sizes[l];
            let pred = &mut self.pred[l];
            pred.fill(0.0);
            for (j, &z) in self.lat[l].z.iter().enumerate() {
                if z != 0.0 {
                    let col = &self.w[l][j * n..][..n];
                    pred.iter_mut().zip(col).for_each(|(m, &w)| *m += w * z);
                }
            }
            let actual = if l == 0 { &self.zx } else { &self.lat[l - 1].z };
            for (e, (&a, &m)) in self.err[l].iter_mut().zip(actual.iter().zip(pred.iter())) {
                *e = a - m;
            }
            self.energy[l] = dot(&self.err[l], &self.err[l]) / n as f32;
        }
        self.input_energy = dot(&self.zx, &self.zx) / self.zx.len() as f32;
    }

    // each spike pulls its unit's prediction and error weights along the error below
    fn learn(&mut self) {
        let (a, ab, norm) = (self.p.alpha, self.p.alpha * self.p.beta, self.p.norm);
        for l in 0..self.lat.len() {
            let n = self.sizes[l];
            for &j in &self.lat[l].spikes {
                let j = j as usize;
                let col = &mut self.w[l][j * n..][..n];
                col.iter_mut()
                    .zip(&self.err[l])
                    .for_each(|(w, &e)| *w += a * e);
                set_norm(col, norm);
                let row = &mut self.e[l][j * n..][..n];
                row.iter_mut()
                    .zip(&self.err[l])
                    .for_each(|(w, &e)| *w += ab * e);
                set_norm(row, norm);
            }
        }
    }

    /// Ends a presentation; while learning, thresholds drift so each unit
    /// fires near the target rate on average.
    pub fn end(&mut self, learn: bool) {
        let (target, eta) = (self.p.target_rate, self.p.theta_rate);
        if !learn || target <= 0.0 {
            return;
        }
        let secs = self.p.present / 1000.0;
        for l in &mut self.lat {
            for (th, &c) in l.thr.iter_mut().zip(&l.counts) {
                let r = c as f32 / secs;
                let k = (eta * (r - target) / target).clamp(-1.0, 1.0).exp();
                *th = (*th * k).clamp(1e-4, 1e4);
            }
        }
    }

    /// Runs a whole presentation.
    pub fn present(&mut self, train: &SpikeTrain, learn: bool) {
        self.reset();
        for t in 1..=train.steps() {
            self.step(t, train, learn);
        }
        self.end(learn);
    }

    /// Each latent unit's spike count this image, square-rooted to tame bursts.
    pub fn code(&self, out: &mut Vec<f32>) {
        out.clear();
        for l in &self.lat {
            out.extend(l.counts.iter().map(|&c| (c as f32).sqrt()));
        }
    }

    /// Prediction weights of layer-1 unit `j`: what it expects to see.
    pub fn field(&self, j: usize) -> &[f32] {
        let n = self.sizes[0];
        &self.w[0][j * n..(j + 1) * n]
    }

    /// Cosine between unit `j`'s prediction weights and its error weights.
    pub fn alignment(&self, j: usize) -> f32 {
        let n = self.sizes[0];
        let (w, e) = (self.field(j), &self.e[0][j * n..(j + 1) * n]);
        dot(w, e) / (dot(w, w) * dot(e, e)).sqrt().max(f32::MIN_POSITIVE)
    }
}

// eight running sums, so the loop vectorises despite float ordering
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ca
        .remainder()
        .iter()
        .zip(cb.remainder())
        .map(|(x, y)| x * y)
        .sum();
    let mut acc = [0.0f32; 8];
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    acc.iter().sum::<f32>() + tail
}

fn set_norm(v: &mut [f32], norm: f32) {
    let k = norm / dot(v, v).sqrt().max(f32::MIN_POSITIVE);
    v.iter_mut().for_each(|x| *x *= k);
}

#[cfg(test)]
mod tests {
    use super::*;
    use snn_core::encode::{Coding, EncodeParams};

    fn encode(img: &[u8], p: &Params, seed: u64) -> SpikeTrain {
        let pot: Vec<f32> = img.iter().map(|&v| v as f32 / 255.0).collect();
        let ep = EncodeParams {
            coding: Coding::Poisson,
            pot_min: 0.0,
            pot_max: 1.0,
            freq_min: 0.0,
            freq_max: p.max_rate,
            period_scale: 1000.0 / p.dt,
        };
        let mut tr = SpikeTrain::new();
        tr.encode(&pot, p.steps(), &ep, &mut Rng::new(seed));
        tr
    }

    fn small() -> Params {
        Params {
            layers: vec![20, 10],
            ..Params::default()
        }
    }

    fn bar(vertical: bool) -> Vec<u8> {
        let mut img = vec![0u8; 784];
        for a in 4..24 {
            for b in 12..16 {
                let (x, y) = if vertical { (b, a) } else { (a, b) };
                img[y * 28 + x] = 255;
            }
        }
        img
    }

    #[test]
    fn silent_input_stays_silent() {
        let mut net = Net::new(784, small(), &mut Rng::new(1));
        let tr = encode(&[0; 784], &net.p, 1);
        net.reset();
        for t in 1..=tr.steps() {
            net.step(t, &tr, true);
            assert!(net.lat.iter().all(|l| l.spikes.is_empty()));
        }
    }

    #[test]
    fn error_is_the_input_trace_until_something_spikes() {
        let mut net = Net::new(784, small(), &mut Rng::new(1));
        let tr = encode(&bar(true), &net.p, 2);
        net.reset();
        for t in 1..=tr.steps() {
            net.step(t, &tr, false);
            if net.lat[0].counts.iter().any(|&c| c > 0) {
                break;
            }
            assert_eq!(net.err[0], net.zx);
        }
    }

    #[test]
    fn learning_moves_only_spiking_units_and_keeps_norms() {
        let mut net = Net::new(784, small(), &mut Rng::new(2));
        let tr = encode(&bar(true), &net.p, 3);
        net.reset();
        let mut moved_any = false;
        for t in 1..=tr.steps() {
            let before = net.w[0].clone();
            net.step(t, &tr, true);
            for j in 0..20 {
                let moved = net.field(j) != &before[j * 784..(j + 1) * 784];
                assert_eq!(moved, net.lat[0].spikes.contains(&(j as u16)));
                moved_any |= moved;
                let n = dot(net.field(j), net.field(j)).sqrt();
                assert!((n - net.p.norm).abs() < 1e-3);
            }
        }
        assert!(moved_any);
        // error weights got the same updates, so they still mirror W
        assert!((0..20).all(|j| net.alignment(j) > 0.999));
    }

    #[test]
    fn homeostasis_lowers_thresholds_of_silent_units() {
        let mut net = Net::new(784, small(), &mut Rng::new(3));
        let tr = encode(&[0; 784], &net.p, 4);
        net.present(&tr, true);
        assert!(net.lat[0].thr.iter().all(|&t| t < net.p.v_thr));
    }

    #[test]
    fn learning_improves_predictions() {
        let mut net = Net::new(784, small(), &mut Rng::new(4));
        let (v, h) = (
            encode(&bar(true), &net.p, 5),
            encode(&bar(false), &net.p, 6),
        );
        // error left in the second half of the image, relative to the bare input
        let settle = |net: &mut Net, tr: &SpikeTrain| {
            net.reset();
            let (mut e, mut x) = (0.0, 0.0);
            for t in 1..=tr.steps() {
                net.step(t, tr, false);
                if t > tr.steps() / 2 {
                    e += net.energy[0];
                    x += dot(&net.zx, &net.zx) / 784.0;
                }
            }
            e / x
        };
        let before = settle(&mut net, &v) + settle(&mut net, &h);
        for _ in 0..40 {
            net.present(&v, true);
            net.present(&h, true);
        }
        let after = settle(&mut net, &v) + settle(&mut net, &h);
        assert!(after < 0.9 * before, "{before} -> {after}");
    }
}
