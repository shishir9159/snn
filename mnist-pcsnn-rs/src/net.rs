//! PC-SNN: time-to-first-spike coding, at most one spike per neuron. Firing
//! times are the latent variables. A layer predicts the next layer's times
//! through non-leaky integrate-and-fire neurons; inference moves the hidden times
//! to shrink the timing errors while the outputs are held at their targets; then
//! a local Hebbian rule changes a synapse whose spike came before the
//! postsynaptic one, by the postsynaptic error.

use crate::params::Params;
use snn_core::{weights, Rng};

#[derive(Debug, Clone)]
pub struct Net {
    pub p: Params,
    pub n_in: usize,
    pub classes: usize,
    /// input to hidden, input-major: input i reaches hidden j through `w1[i*H + j]`
    pub w1: Vec<f32>,
    /// hidden to output, hidden-major: hidden j reaches output k through `w2[j*C + k]`
    pub w2: Vec<f32>,
    /// spike time of each input; t_max for black pixels
    pub t_in: Vec<f32>,
    /// hidden times predicted by the forward pass
    pub hid_pred: Vec<f32>,
    /// hidden times after inference
    pub hid: Vec<f32>,
    /// output times of the forward pass, which make the decision
    pub out_fwd: Vec<f32>,
    /// how far each output overshot its threshold when it fired
    pub out_over: Vec<f32>,
    /// output times predicted from the inferred hidden times
    pub out_pred: Vec<f32>,
    /// output times inference holds the outputs to
    pub target: Vec<f32>,
    pub eps_hid: Vec<f32>,
    pub eps_out: Vec<f32>,
    /// free energy before inference and after each iteration
    pub energy: Vec<f32>,
    /// hidden times at each iteration, iteration-major
    pub trail: Vec<f32>,
    /// hidden neurons silenced by dropout for this image
    pub dropped: Vec<bool>,
    /// output threshold for this image: raised when nothing is dropped, so the
    /// outputs see the drive they were trained with
    out_threshold: f32,
    /// images each hidden neuron fired in since the last revival check
    fired: Vec<u32>,
    seen: usize,
    order: Vec<usize>,
    pot: Vec<f32>,
    scratch: Vec<f32>,
}

impl Net {
    pub fn new(n_in: usize, classes: usize, p: Params, rng: &mut Rng) -> Self {
        let h = p.hidden;
        Self {
            w1: weights::random(n_in * h, 0.0, p.init_hidden, rng),
            w2: weights::random(h * classes, 0.0, p.init_output, rng),
            t_in: vec![p.t_max; n_in],
            hid_pred: vec![p.t_max; h],
            hid: vec![p.t_max; h],
            out_fwd: vec![p.t_max; classes],
            out_over: vec![0.0; classes],
            out_pred: vec![p.t_max; classes],
            target: vec![p.t_max; classes],
            eps_hid: vec![0.0; h],
            eps_out: vec![0.0; classes],
            energy: Vec::new(),
            trail: Vec::new(),
            dropped: vec![false; h],
            out_threshold: p.threshold,
            fired: vec![0; h],
            seen: 0,
            order: Vec::new(),
            pot: Vec::new(),
            scratch: vec![0.0; h.max(classes)],
            n_in,
            classes,
            p,
        }
    }

    /// Brighter pixels spike earlier: t = (255 - p) / 255 * t_max.
    pub fn encode(&mut self, pixels: &[u8]) {
        let t_max = self.p.t_max;
        for (t, &px) in self.t_in.iter_mut().zip(pixels) {
            *t = ((255 - px) as f32 / 255.0 * t_max).round();
        }
    }

    /// Silences a random share of hidden neurons, or none.
    pub fn drop_hidden(&mut self, rng: Option<&mut Rng>) {
        match rng {
            Some(rng) => {
                let p = self.p.dropout;
                self.dropped.iter_mut().for_each(|d| *d = rng.f32() < p);
                self.out_threshold = self.p.threshold;
            }
            None => {
                self.dropped.fill(false);
                self.out_threshold = self.p.threshold / (1.0 - self.p.dropout);
            }
        }
    }

    /// Feeds the input times forward; returns the first output to fire.
    pub fn forward(&mut self) -> Option<u8> {
        let (h, c) = (self.p.hidden, self.classes);
        let all = vec![false; self.n_in];
        let into_hidden = Layer {
            skip: &all,
            w: &self.w1,
            n_post: h,
            threshold: self.p.threshold,
            p: &self.p,
        };
        into_hidden.fire(
            &self.t_in,
            &mut self.order,
            &mut self.pot,
            &mut self.hid_pred,
            &mut self.scratch[..h],
        );
        for (t, &d) in self.hid_pred.iter_mut().zip(&self.dropped) {
            if d {
                *t = self.p.t_max;
            }
        }
        let into_output = Layer {
            skip: &self.dropped,
            w: &self.w2,
            n_post: c,
            threshold: self.out_threshold,
            p: &self.p,
        };
        into_output.fire(
            &self.hid_pred,
            &mut self.order,
            &mut self.pot,
            &mut self.out_fwd,
            &mut self.out_over,
        );
        self.decision()
    }

    /// Output that fired first in the forward pass, if any fired; on a shared
    /// step, the one that overshot its threshold most.
    pub fn decision(&self) -> Option<u8> {
        let first = |a: &(usize, &f32), b: &(usize, &f32)| {
            a.1.total_cmp(b.1)
                .then(self.out_over[b.0].total_cmp(&self.out_over[a.0]))
        };
        let (k, &t) = self.out_fwd.iter().enumerate().min_by(first)?;
        (t < self.p.t_max).then_some(k as u8)
    }

    /// Holds the outputs to targets for `label` and lets the hidden times settle.
    pub fn infer(&mut self, label: u8) {
        let p = &self.p;
        let (h, c) = (p.hidden, self.classes);
        // the true class should fire strictly first, the others at least gamma later
        let winner = self.decision();
        let first = winner.map_or(p.t_max - p.gamma, |k| self.out_fwd[k as usize]);
        for (k, t) in self.target.iter_mut().enumerate() {
            *t = if k != label as usize {
                (first + p.gamma).max(self.out_fwd[k]).min(p.t_max)
            } else if winner == Some(label) {
                first
            } else {
                (first - 1.0).max(0.0)
            };
        }
        self.hid.copy_from_slice(&self.hid_pred);
        self.energy.clear();
        self.trail.clear();
        let into_output = Layer {
            skip: &self.dropped,
            w: &self.w2,
            n_post: c,
            threshold: self.out_threshold,
            p,
        };
        for it in 0..=p.iterations {
            into_output.fire(
                &self.hid,
                &mut self.order,
                &mut self.pot,
                &mut self.out_pred,
                &mut self.scratch[..c],
            );
            for j in 0..h {
                self.eps_hid[j] = (self.hid[j] - self.hid_pred[j]) / p.sigma;
            }
            for k in 0..c {
                self.eps_out[k] = (self.target[k] - self.out_pred[k]) / p.sigma;
            }
            let f: f32 = self
                .eps_hid
                .iter()
                .chain(&self.eps_out)
                .map(|e| e * e)
                .sum();
            self.energy.push(0.5 * f);
            self.trail.extend_from_slice(&self.hid);
            if it == p.iterations {
                break;
            }
            // gradient of the free energy, dV/dt taken as alpha
            for j in 0..h {
                if self.dropped[j] {
                    continue;
                }
                let (t, row) = (self.hid[j], &self.w2[j * c..][..c]);
                let up: f32 = row
                    .iter()
                    .zip(self.out_pred.iter().zip(&self.eps_out))
                    .filter(|&(_, (&out, _))| t < p.t_max && t <= out)
                    .map(|(&w, (_, &e))| e * w / p.alpha)
                    .sum();
                let step = p.infer_rate * (up - self.eps_hid[j]);
                self.hid[j] = (self.hid[j] + step).clamp(0.0, p.t_max);
            }
        }
    }

    /// Hebbian step: a synapse whose spike preceded the postsynaptic (predicted)
    /// spike moves against the postsynaptic error.
    pub fn learn(&mut self) {
        for (f, &t) in self.fired.iter_mut().zip(&self.hid_pred) {
            *f += (t < self.p.t_max) as u32;
        }
        self.seen += 1;
        let p = &self.p;
        let (h, c) = (p.hidden, self.classes);
        for (j, row) in self.w2.chunks_exact_mut(c).enumerate() {
            if self.dropped[j] {
                continue;
            }
            let t = self.hid[j];
            if t >= p.t_max {
                continue;
            }
            for (w, (&out, &e)) in row.iter_mut().zip(self.out_pred.iter().zip(&self.eps_out)) {
                if t <= out {
                    *w -= p.eta_output * e / p.alpha;
                }
            }
        }
        for (i, row) in self.w1.chunks_exact_mut(h).enumerate() {
            let t = self.t_in[i];
            if t >= p.t_max {
                continue;
            }
            let post = self.hid_pred.iter().zip(&self.eps_hid).zip(&self.dropped);
            for (w, ((&tj, &e), &d)) in row.iter_mut().zip(post) {
                if t <= tj && !d {
                    *w -= p.eta_hidden * e / p.alpha;
                }
            }
        }
    }

    /// Gives fresh input weights to hidden neurons that hardly fired lately: a
    /// silent neuron gets no error, so it would never learn again. Returns how
    /// many were revived.
    pub fn revive(&mut self, rng: &mut Rng) -> usize {
        if self.seen < self.p.revive_every {
            return 0;
        }
        let (h, low) = (self.p.hidden, self.p.revive_below * self.seen as f32);
        let mut n = 0;
        for j in 0..h {
            if (self.fired[j] as f32) < low {
                for i in 0..self.n_in {
                    self.w1[i * h + j] = rng.range(0.0, self.p.init_hidden);
                }
                n += 1;
            }
        }
        self.fired.fill(0);
        self.seen = 0;
        n
    }

    /// Weights from the input into hidden neuron `j`.
    pub fn field(&self, j: usize, out: &mut Vec<f32>) {
        let h = self.p.hidden;
        out.clear();
        out.extend((0..self.n_in).map(|i| self.w1[i * h + j]));
    }
}

/// Integrate-and-fire neurons fed by one layer of spikes.
struct Layer<'a> {
    /// presynaptic neurons that stay silent
    skip: &'a [bool],
    /// pre-major weights
    w: &'a [f32],
    n_post: usize,
    threshold: f32,
    p: &'a Params,
}

impl Layer<'_> {
    /// First time each neuron's potential reaches threshold for presynaptic
    /// spikes at `pre`. Spikes at the same time arrive together; `over` gets how
    /// far each neuron overshot when it fired, which orders a shared step.
    fn fire(
        &self,
        pre: &[f32],
        order: &mut Vec<usize>,
        pot: &mut Vec<f32>,
        out: &mut [f32],
        over: &mut [f32],
    ) {
        let n = self.n_post;
        order.clear();
        // a neuron at t_max never spiked
        order.extend((0..pre.len()).filter(|&i| !self.skip[i] && pre[i] < self.p.t_max));
        order.sort_by(|&a, &b| pre[a].total_cmp(&pre[b]));
        pot.clear();
        pot.resize(n, 0.0);
        out.fill(f32::INFINITY);
        over.fill(0.0);
        let (mut left, mut at) = (n, 0);
        while at < order.len() && left > 0 {
            let t = pre[order[at]];
            while at < order.len() && pre[order[at]] == t {
                let row = &self.w[order[at] * n..][..n];
                pot.iter_mut().zip(row).for_each(|(v, &w)| *v += w);
                at += 1;
            }
            for k in 0..n {
                if pot[k] >= self.threshold && out[k] == f32::INFINITY {
                    out[k] = t;
                    over[k] = pot[k] - self.threshold;
                    left -= 1;
                }
            }
        }
        out.iter_mut().for_each(|t| *t = t.min(self.p.t_max));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digit(d: u8, rng: &mut Rng) -> Vec<u8> {
        let mut img = vec![0u8; 784];
        snn_core::dataset::render_digit(d, 28, rng, &mut img);
        img
    }

    #[test]
    fn bright_pixels_spike_first() {
        let p = Params {
            hidden: 3,
            ..Params::default()
        };
        let mut net = Net::new(3, 2, p, &mut Rng::new(1));
        net.encode(&[255, 0, 128]);
        assert_eq!(net.t_in, vec![0.0, 256.0, 127.0]);
    }

    #[test]
    fn integrate_and_fire_times() {
        let p = Params::default();
        let (mut order, mut pot) = (Vec::new(), Vec::new());
        let (mut out, mut over) = (vec![0.0; 2], vec![0.0; 2]);
        // post 0 crosses 100 with the second spike; post 1 never does
        let w = [60.0, 10.0, 50.0, 10.0, 100.0, 10.0];
        let pre = [1.0, 2.0, 5.0];
        let mut l = Layer {
            skip: &[false; 3],
            w: &w,
            n_post: 2,
            threshold: p.threshold,
            p: &p,
        };
        l.fire(&pre, &mut order, &mut pot, &mut out, &mut over);
        assert_eq!(out, [2.0, 256.0]);
        assert_eq!(over, [10.0, 0.0]);
        // silencing the second input delays post 0 to the third
        l.skip = &[false, true, false];
        l.fire(&pre, &mut order, &mut pot, &mut out, &mut over);
        assert_eq!(out, [5.0, 256.0]);
    }

    #[test]
    fn ties_go_to_the_larger_overshoot() {
        let mut net = Net::new(1, 3, Params::default(), &mut Rng::new(1));
        net.out_fwd = vec![9.0, 4.0, 4.0];
        net.out_over = vec![50.0, 1.0, 3.0];
        assert_eq!(net.decision(), Some(2));
        net.out_fwd = vec![256.0; 3];
        assert_eq!(net.decision(), None);
    }

    #[test]
    fn inference_targets_the_label_first() {
        let mut rng = Rng::new(2);
        let mut net = Net::new(784, 10, Params::default(), &mut rng);
        net.encode(&digit(3, &mut rng));
        let winner = net.forward().unwrap();
        let label = (winner + 1) % 10;
        net.infer(label);
        let first = net.out_fwd[winner as usize];
        assert!(net.target[label as usize] < first);
        for k in (0..10).filter(|&k| k != label as usize) {
            assert!(net.target[k] >= (first + net.p.gamma).min(net.p.t_max));
        }
        assert_eq!(net.energy.len(), net.p.iterations + 1);
        assert_eq!(net.trail.len(), net.energy.len() * net.p.hidden);
        // the hidden layer moved to explain the targets
        assert!(net.hid.iter().zip(&net.hid_pred).any(|(a, b)| a != b));
    }

    #[test]
    fn learning_makes_the_label_fire_first() {
        let mut rng = Rng::new(3);
        let mut net = Net::new(784, 10, Params::default(), &mut rng);
        net.encode(&digit(5, &mut rng));
        for _ in 0..200 {
            if net.forward() == Some(4) {
                return;
            }
            net.infer(4);
            net.learn();
        }
        panic!("never fired first");
    }
}
