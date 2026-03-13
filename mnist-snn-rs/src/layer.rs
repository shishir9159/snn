//! Winner-take-all output layer fully connected to the input spike train.

use crate::dopamine::{Decision, Dopamine};
use crate::neuron::Neuron;
use crate::params::{Params, Rule};
use crate::stdp::{bounded, PostTrace, Stdp, Triplet};
use snn_core::encode::SpikeTrain;
use snn_core::rng::Rng;
use snn_core::weights;
use snn_core::{Activity, Quantizer};

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
    /// spikes of the input train that were delivered
    pub input_spikes: u32,
    /// threshold before homeostatic scaling
    pub base_threshold: f32,
}

impl Outcome {
    /// Each input spike is added into the drive of every output, and the winner
    /// inhibits the other outputs once.
    pub fn activity(&self, n_out: usize) -> Activity {
        let fired: u64 = self.spikes.iter().map(|&s| s as u64).sum();
        let inhibited = if self.winner.is_some() {
            n_out as u64 - 1
        } else {
            0
        };
        let input = self.input_spikes as u64;
        Activity::new(input + fired, input * n_out as u64 + inhibited)
    }
}

/// What a training presentation under [`Rule::Reward`] or [`Rule::Delayed`]
/// is told: neuron j answers class `j % classes`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reward {
    pub label: u8,
    pub classes: usize,
    /// learning-rate scale of a right answer and of a wrong one
    pub reward: f32,
    pub punish: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Learn {
    Off,
    On,
    Reward(Reward),
    Delayed(Reward),
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub n_in: usize,
    pub n_out: usize,
    /// row-major, one row of `n_in` weights per output neuron
    weights: Vec<f32>,
    /// the same weights input-major, so a spike adds one contiguous run
    by_input: Vec<f32>,
    drive: Vec<f32>,
    pub neurons: Vec<Neuron>,
    /// homeostatic threshold scale: threshold = base * (1 + theta)
    pub theta: Vec<f32>,
    stdp: Stdp,
    triplet: Triplet,
    /// each neuron's triplet traces within the presentation
    post: Vec<PostTrace>,
    thresholds: Vec<f32>,
    fired: Vec<u16>,
    /// eligibility traces and dopamine of the delayed rule
    pub dopamine: Dopamine,
    /// (neuron, step) of each output spike, which the delayed rule tags from
    spikes: Vec<(u16, u16)>,
    /// the weight grid, when the weights are stored in fewer bits
    quant: Option<Quantizer>,
    qrng: Rng,
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

    /// Weights of `n_in` per output neuron; with `weight_bits` they are snapped
    /// to the nearest level.
    pub fn with_weights(n_in: usize, n_out: usize, mut weights: Vec<f32>, p: &Params) -> Self {
        assert_eq!(weights.len(), n_in * n_out);
        let quant = Quantizer::new(
            p.weight_bits,
            p.stdp.w_min,
            p.stdp.w_max,
            p.stochastic_rounding,
        );
        if let Some(q) = &quant {
            q.nearest(&mut weights);
        }
        let mut by_input = vec![0.0; n_in * n_out];
        for (j, row) in weights.chunks_exact(n_in).enumerate() {
            for (i, &w) in row.iter().enumerate() {
                by_input[i * n_out + j] = w;
            }
        }
        Self {
            n_in,
            n_out,
            weights,
            by_input,
            drive: vec![0.0; n_out],
            neurons: vec![Neuron::new(&p.lif); n_out],
            theta: vec![0.0; n_out],
            stdp: Stdp::new(&p.stdp),
            triplet: Triplet::new(&p.stdp, p.time_steps),
            post: vec![PostTrace::default(); n_out],
            thresholds: vec![0.0; n_out],
            fired: Vec::with_capacity(n_out),
            dopamine: Dopamine::new(&p.dopamine, n_in, n_out),
            spikes: Vec::new(),
            quant,
            qrng: Rng::new(0x51ab),
        }
    }

    /// Row-major weights, one row per output neuron.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn row(&self, j: usize) -> &[f32] {
        &self.weights[j * self.n_in..(j + 1) * self.n_in]
    }

    /// MACs per image of a dense ANN layer of the same shape.
    pub fn dense_macs(&self) -> u64 {
        (self.n_in * self.n_out) as u64
    }

    // copies neuron j's row into the input-major view
    fn sync(&mut self, j: usize) {
        let row = &self.weights[j * self.n_in..(j + 1) * self.n_in];
        for (i, &w) in row.iter().enumerate() {
            self.by_input[i * self.n_out + j] = w;
        }
    }

    /// Runs one image through the layer. With `learn`, applies the pair or
    /// triplet rule on every output spike and updates homeostasis afterwards.
    pub fn present<O: Observer>(
        &mut self,
        train: &SpikeTrain,
        p: &Params,
        learn: bool,
        obs: &mut O,
    ) -> Outcome {
        let learn = if learn { Learn::On } else { Learn::Off };
        self.run(train, p, learn, obs)
    }

    /// A training presentation under reward-modulated STDP: once the winner
    /// answers, its spikes apply the pair rule scaled by `r.reward` if the
    /// answer is right, and depress the inputs around them scaled by
    /// `r.punish` if it is wrong. Under [`Rule::Delayed`] they only tag the
    /// synapses, and the reward reaches them later through [`Dopamine`].
    pub fn present_rewarded<O: Observer>(
        &mut self,
        train: &SpikeTrain,
        p: &Params,
        r: Reward,
        obs: &mut O,
    ) -> Outcome {
        let learn = match p.rule {
            Rule::Delayed => Learn::Delayed(r),
            _ => Learn::Reward(r),
        };
        self.run(train, p, learn, obs)
    }

    fn run<O: Observer>(
        &mut self,
        train: &SpikeTrain,
        p: &Params,
        learn: Learn,
        obs: &mut O,
    ) -> Outcome {
        assert_eq!(train.n_inputs(), self.n_in);
        let lif = &p.lif;
        let triplet = learn == Learn::On && p.rule == Rule::Triplet;
        let delayed = matches!(learn, Learn::Delayed(_));
        // a blank image gives no drive; keep the threshold above rest
        let base = (train.max_simultaneous() as f32 / p.threshold_div).max(1.0);
        for (th, &theta) in self.thresholds.iter_mut().zip(&self.theta) {
            *th = base * (1.0 + theta);
        }
        for n in &mut self.neurons {
            n.reset(lif);
        }
        self.post.fill(PostTrace::default());
        let mut spikes = vec![0u32; self.n_out];
        let mut input_spikes = 0u32;
        let mut winner = None;
        let mut decided = 0;
        self.spikes.clear();
        // sign and size of the pair rule for the winner, set by a reward
        let mut scale = 1.0;
        let mut fired = std::mem::take(&mut self.fired);

        for t in 1..=train.steps() {
            let ti = t as i32;
            let active = train.active(t);
            input_spikes += active.len() as u32;
            self.drive.fill(0.0);
            for &i in active {
                let col = &self.by_input[i as usize * self.n_out..][..self.n_out];
                for (d, &w) in self.drive.iter_mut().zip(col) {
                    *d += w;
                }
            }
            for (n, &input) in self.neurons.iter_mut().zip(&self.drive) {
                if !n.is_refractory(ti) {
                    n.integrate(ti, input, lif);
                }
            }

            if winner.is_none() {
                winner = self.pick_winner(ti);
                if let Some(w) = winner {
                    decided = t;
                    for (j, n) in self.neurons.iter_mut().enumerate() {
                        if j != w {
                            n.inhibit(lif);
                        }
                    }
                    if let Learn::Reward(r) = learn {
                        let right = w % r.classes == r.label as usize;
                        scale = if right { r.reward } else { -r.punish };
                    }
                }
            }
            if triplet {
                self.depress_on_pre(active, t, p);
            }

            fired.clear();
            let rows = self.weights.chunks_exact_mut(self.n_in);
            for (j, (n, row)) in self.neurons.iter_mut().zip(rows).enumerate() {
                if !n.fire(ti, self.thresholds[j], lif) {
                    continue;
                }
                fired.push(j as u16);
                spikes[j] += 1;
                if learn == Learn::Off {
                    continue;
                }
                if delayed {
                    self.spikes.push((j as u16, t as u16));
                    continue;
                }
                let (quant, rng) = (&self.quant, &mut self.qrng);
                let round = |w: f32| match quant {
                    Some(q) => q.snap(w, rng),
                    None => w,
                };
                if triplet {
                    let o = &mut self.post[j];
                    self.triplet.post_spike(row, train, t, o, round);
                } else {
                    self.stdp.post_spike_rounded(row, train, t, scale, round);
                }
            }
            if learn != Learn::Off && !delayed {
                for &j in &fired {
                    self.sync(j as usize);
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

        if let Learn::Delayed(r) = learn {
            self.release(train, p, r, winner, decided);
        }
        if learn != Learn::Off {
            if let Some(w) = winner.filter(|_| !delayed) {
                let row = &mut self.weights[w * self.n_in..(w + 1) * self.n_in];
                // a punished winner keeps its silent inputs and loses weight
                if scale > 0.0 {
                    self.stdp.depress_silent(row, train, p.silent_penalty);
                    normalize(row, p.w_norm, p.stdp.w_max);
                }
                if let Some(q) = &self.quant {
                    q.apply(row, &mut self.qrng);
                }
                self.sync(w);
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
            input_spikes,
            base_threshold: base,
        }
    }

    // hands the presentation to the dopamine and applies the weight changes it
    // releases; rows that took more reward than punishment are renormalised
    fn release(
        &mut self,
        train: &SpikeTrain,
        p: &Params,
        r: Reward,
        winner: Option<usize>,
        decided: usize,
    ) {
        let decision = winner.map(|w| Decision {
            row: w,
            step: decided,
            amount: if w % r.classes == r.label as usize {
                r.reward
            } else {
                -r.punish
            },
        });
        let (n_in, n_out) = (self.n_in, self.n_out);
        let (weights, by_input) = (&mut self.weights, &mut self.by_input);
        let (quant, rng) = (&self.quant, &mut self.qrng);
        // in the units of the kernel, so a full reward takes silent_penalty off
        let silent = p.silent_penalty / (p.stdp.sigma * p.dopamine.gain.max(f32::MIN_POSITIVE));
        let spikes = &self.spikes;
        self.dopamine.present(
            &self.stdp,
            train,
            spikes,
            decision,
            silent,
            |j, dw, rewarded| {
                let row = &mut weights[j * n_in..(j + 1) * n_in];
                let (lo, hi, rate) = (p.stdp.w_min, p.stdp.w_max, p.stdp.sigma);
                for (w, &d) in row.iter_mut().zip(dw) {
                    // stdp::bounded with a select for the branch, so it vectorises
                    let room = if d > 0.0 { hi - *w } else { *w - lo };
                    *w += rate * d * room;
                }
                if rewarded {
                    normalize(row, p.w_norm, p.stdp.w_max);
                }
                if let Some(q) = quant {
                    q.apply(row, rng);
                }
                for (i, &w) in row.iter().enumerate() {
                    by_input[i * n_out + j] = w;
                }
            },
        );
    }

    // triplet depression: inputs spiking at `t` onto neurons that fired before
    fn depress_on_pre(&mut self, active: &[u16], t: usize, p: &Params) {
        for j in 0..self.n_out {
            let dw = self.triplet.pre_spike(&self.post[j], t);
            if dw == 0.0 {
                continue;
            }
            for &i in active {
                let i = i as usize;
                let w = &mut self.weights[j * self.n_in + i];
                let mut v = bounded(*w, dw, &p.stdp);
                if let Some(q) = &self.quant {
                    v = q.snap(v, &mut self.qrng);
                }
                *w = v;
                self.by_input[i * self.n_out + j] = v;
            }
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
    use snn_core::encode::receptive_field;

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
        let mut w = vec![0.5; 784 * 3];
        w[784..1568].iter_mut().for_each(|w| *w = 0.6);
        let mut l = Layer::with_weights(784, 3, w, &p);
        let mut obs = Count(0);
        let out = l.present(&encode(&bar(true), &p), &p, false, &mut obs);
        assert_eq!(obs.0, p.time_steps);
        assert_eq!(out.winner, Some(1));
        assert!(out.spikes[1] > 0);
        assert_eq!(out.spikes[0] + out.spikes[2], 0);
    }

    #[test]
    fn activity_counts_deliveries_and_inhibition() {
        let p = Params::default();
        let mut l = Layer::new(784, 4, &p, &mut Rng::new(1));
        let tr = encode(&bar(true), &p);
        let out = l.present(&tr, &p, false, &mut ());
        let sent: usize = (1..=tr.steps()).map(|t| tr.active(t).len()).sum();
        assert_eq!(out.input_spikes as usize, sent);
        let a = out.activity(4);
        let fired: u32 = out.spikes.iter().sum();
        assert_eq!(a.spikes, (sent as u32 + fired) as u64);
        let inhibition = if out.winner.is_some() { 3 } else { 0 };
        assert_eq!(a.synops, (sent * 4 + inhibition) as u64);

        let quiet = l.present(&encode(&[0; 784], &p), &p, false, &mut ());
        assert_eq!(quiet.activity(4), Activity::default());
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
        let before = l.weights().to_vec();
        l.present(&encode(&bar(true), &p), &p, false, &mut ());
        assert_eq!(l.weights(), before);
        assert!(l.theta.iter().all(|&t| t == 0.0));
    }

    #[test]
    fn input_major_view_follows_learning() {
        let p = Params::default();
        let mut l = Layer::new(784, 5, &p, &mut Rng::new(4));
        for _ in 0..5 {
            l.present(&encode(&bar(true), &p), &p, true, &mut ());
            l.present(&encode(&bar(false), &p), &p, true, &mut ());
        }
        for j in 0..5 {
            for i in 0..784 {
                assert_eq!(l.by_input[i * 5 + j], l.row(j)[i]);
            }
        }
    }

    fn on_grid(w: &[f32], bits: u32, p: &Params) -> bool {
        let step = (p.stdp.w_max - p.stdp.w_min) / ((1u32 << bits) - 1) as f32;
        w.iter().all(|&x| {
            let t = (x - p.stdp.w_min) / step;
            (t - t.round()).abs() < 1e-3
        })
    }

    #[test]
    fn quantised_weights_stay_on_the_grid_and_still_learn() {
        for stochastic in [true, false] {
            let p = Params {
                weight_bits: 5,
                stochastic_rounding: stochastic,
                ..Params::default()
            };
            let mut l = Layer::new(784, 2, &p, &mut Rng::new(3));
            assert!(on_grid(l.weights(), 5, &p), "initial weights");
            let (v, h) = (encode(&bar(true), &p), encode(&bar(false), &p));
            for _ in 0..30 {
                l.present(&v, &p, true, &mut ());
                l.present(&h, &p, true, &mut ());
            }
            assert!(on_grid(l.weights(), 5, &p), "learned weights");
            let wv = l.present(&v, &p, false, &mut ()).winner;
            let wh = l.present(&h, &p, false, &mut ()).winner;
            assert!(
                wv.is_some() && wh.is_some() && wv != wh,
                "stochastic {stochastic}"
            );
            for j in 0..2 {
                for i in 0..784 {
                    assert_eq!(l.by_input[i * 2 + j], l.row(j)[i], "input-major copy");
                }
            }
        }
    }

    #[test]
    fn loaded_weights_snap_to_the_grid() {
        let p = Params {
            weight_bits: 3,
            ..Params::default()
        };
        let w = snn_core::weights::random(784 * 2, 0.0, 2.0, &mut Rng::new(9));
        let l = Layer::with_weights(784, 2, w.clone(), &p);
        assert!(on_grid(l.weights(), 3, &p));
        assert_ne!(l.weights(), w.as_slice());
        let full = Layer::with_weights(784, 2, w.clone(), &Params::default());
        assert_eq!(full.weights(), w.as_slice(), "no bits, no change");
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
