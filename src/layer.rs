//! Winner-take-all output layer fully connected to the input spike train.

use crate::encode::SpikeTrain;
use crate::neuron::Neuron;
use crate::params::Params;
use crate::rng::Rng;
use crate::stdp::Stdp;
use crate::weights;

/// State after one simulated step, handed to an [`Observer`].
pub struct Step<'a> {
    pub t: usize,
    pub active: &'a [u16],
    pub neurons: &'a [Neuron],
    pub thresholds: &'a [f32],
    /// output neurons that spiked this step
    pub fired: &'a [u16],
    pub winner: Option<usize>,
}

/// Watches a presentation step by step.
pub trait Observer {
    fn step(&mut self, _s: &Step) {}
}

impl Observer for () {}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Outcome {
    /// first neuron over threshold; it inhibited all others
    pub winner: Option<usize>,
    /// spikes per output neuron
    pub spikes: Vec<u32>,
    /// threshold before homeostatic scaling
    pub base_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub n_in: usize,
    pub n_out: usize,
    /// row-major, one row of `n_in` weights per output neuron
    pub weights: Vec<f32>,
    pub neurons: Vec<Neuron>,
    /// homeostatic threshold scale: threshold = base * (1 + theta)
    pub theta: Vec<f32>,
    stdp: Stdp,
    thresholds: Vec<f32>,
    fired: Vec<u16>,
}

impl Layer {
    pub fn new(n_in: usize, n_out: usize, p: &Params, rng: &mut Rng) -> Self {
        let (lo, hi) = p.w_init;
        let mut w = weights::random(n_in * n_out, lo, hi, rng);
        for row in w.chunks_exact_mut(n_in) {
            normalize(row, p.w_norm, p.stdp.w_max);
        }
        Self::with_weights(n_in, n_out, w, p)
    }

    pub fn with_weights(n_in: usize, n_out: usize, weights: Vec<f32>, p: &Params) -> Self {
        assert_eq!(weights.len(), n_in * n_out);
        Self {
            n_in,
            n_out,
            weights,
            neurons: vec![Neuron::new(&p.lif); n_out],
            theta: vec![0.0; n_out],
            stdp: Stdp::new(&p.stdp),
            thresholds: vec![0.0; n_out],
            fired: Vec::with_capacity(n_out),
        }
    }

    pub fn row(&self, j: usize) -> &[f32] {
        &self.weights[j * self.n_in..(j + 1) * self.n_in]
    }

    /// Runs one image through the layer. With `learn`, applies STDP on every
    /// output spike and updates homeostasis afterwards.
    pub fn present<O: Observer>(
        &mut self,
        train: &SpikeTrain,
        p: &Params,
        learn: bool,
        obs: &mut O,
    ) -> Outcome {
        assert_eq!(train.n_inputs(), self.n_in);
        let lif = &p.lif;
        // a blank image gives no drive; keep the threshold above rest
        let base = (train.max_simultaneous() as f32 / p.threshold_div).max(1.0);
        for (th, &theta) in self.thresholds.iter_mut().zip(&self.theta) {
            *th = base * (1.0 + theta);
        }
        for n in &mut self.neurons {
            n.reset(lif);
        }
        let mut spikes = vec![0u32; self.n_out];
        let mut winner = None;
        let mut fired = std::mem::take(&mut self.fired);

        for t in 1..=train.steps() {
            let ti = t as i32;
            let active = train.active(t);
            for (n, row) in self
                .neurons
                .iter_mut()
                .zip(self.weights.chunks_exact(self.n_in))
            {
                if !n.is_refractory(ti) {
                    let input: f32 = active.iter().map(|&i| row[i as usize]).sum();
                    n.integrate(ti, input, lif);
                }
            }

            if winner.is_none() {
                winner = self.pick_winner(ti);
                if let Some(w) = winner {
                    for (j, n) in self.neurons.iter_mut().enumerate() {
                        if j != w {
                            n.inhibit(lif);
                        }
                    }
                }
            }

            fired.clear();
            let rows = self.weights.chunks_exact_mut(self.n_in);
            for (j, (n, row)) in self.neurons.iter_mut().zip(rows).enumerate() {
                if n.fire(ti, self.thresholds[j], lif) {
                    fired.push(j as u16);
                    spikes[j] += 1;
                    if learn {
                        self.stdp.post_spike(row, train, t);
                    }
                }
            }

            obs.step(&Step {
                t,
                active,
                neurons: &self.neurons,
                thresholds: &self.thresholds,
                fired: &fired,
                winner,
            });
        }
        self.fired = fired;

        if learn {
            if let Some(w) = winner {
                let row = &mut self.weights[w * self.n_in..(w + 1) * self.n_in];
                self.stdp.depress_silent(row, train, p.silent_penalty);
                normalize(row, p.w_norm, p.stdp.w_max);
            }
            // drive every neuron toward an equal share of wins
            let share = 1.0 / self.n_out as f32;
            for (j, th) in self.theta.iter_mut().enumerate() {
                let won = (winner == Some(j)) as u8 as f32;
                *th = (*th + p.theta_rate * (won - share)).max(-0.9);
            }
        }

        Outcome {
            winner,
            spikes,
            base_threshold: base,
        }
    }

    // free neuron furthest past its own threshold, if any reached it
    fn pick_winner(&self, t: i32) -> Option<usize> {
        let mut best = None;
        let mut margin = 0.0;
        for (j, n) in self.neurons.iter().enumerate() {
            if n.is_refractory(t) {
                continue;
            }
            let m = n.p - self.thresholds[j];
            if m >= margin {
                margin = m;
                best = Some(j);
            }
        }
        best
    }
}

/// Scales a row to L2 norm `norm`, capped at `w_max`.
fn normalize(row: &mut [f32], norm: f32, w_max: f32) {
    let len = row.iter().map(|w| w * w).sum::<f32>().sqrt();
    if norm > 0.0 && len > 0.0 {
        let k = norm / len;
        row.iter_mut().for_each(|w| *w = (*w * k).min(w_max));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::receptive_field;

    fn encode(img: &[u8], p: &Params) -> SpikeTrain {
        let mut pot = Vec::new();
        receptive_field(img, 28, 28, &mut pot);
        let mut tr = SpikeTrain::new();
        tr.encode(&pot, p.time_steps, &p.encode, &mut Rng::new(0));
        tr
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

    struct Count(usize);
    impl Observer for Count {
        fn step(&mut self, s: &Step) {
            assert_eq!(s.t, self.0 + 1);
            self.0 += 1;
        }
    }

    #[test]
    fn single_winner_spikes() {
        let p = Params::default();
        let mut l = Layer::with_weights(784, 3, vec![0.5; 784 * 3], &p);
        l.weights[784..1568].iter_mut().for_each(|w| *w = 0.6);
        let mut obs = Count(0);
        let out = l.present(&encode(&bar(true), &p), &p, false, &mut obs);
        assert_eq!(obs.0, p.time_steps);
        assert_eq!(out.winner, Some(1));
        assert!(out.spikes[1] > 0);
        assert_eq!(out.spikes[0] + out.spikes[2], 0);
    }

    #[test]
    fn blank_image_never_fires() {
        let p = Params::default();
        let mut l = Layer::new(784, 4, &p, &mut Rng::new(1));
        let out = l.present(&encode(&[0; 784], &p), &p, true, &mut ());
        assert_eq!(out.winner, None);
        assert!(out.spikes.iter().all(|&s| s == 0));
    }

    #[test]
    fn inference_leaves_weights_alone() {
        let p = Params::default();
        let mut l = Layer::new(784, 4, &p, &mut Rng::new(2));
        let before = l.weights.clone();
        l.present(&encode(&bar(true), &p), &p, false, &mut ());
        assert_eq!(l.weights, before);
        assert!(l.theta.iter().all(|&t| t == 0.0));
    }

    #[test]
    fn learning_separates_two_patterns() {
        let p = Params::default();
        let mut l = Layer::new(784, 2, &p, &mut Rng::new(3));
        let (v, h) = (encode(&bar(true), &p), encode(&bar(false), &p));
        for _ in 0..30 {
            l.present(&v, &p, true, &mut ());
            l.present(&h, &p, true, &mut ());
        }
        let wv = l.present(&v, &p, false, &mut ()).winner;
        let wh = l.present(&h, &p, false, &mut ()).winner;
        assert!(wv.is_some() && wh.is_some());
        assert_ne!(wv, wh);
        // the winner's weights now favour its own pattern's pixels
        let row = l.row(wv.unwrap());
        assert!(row[14 * 28 + 13] > row[2 * 28 + 2]);
    }
}
