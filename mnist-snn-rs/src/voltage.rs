//! Voltage-based STDP (Clopath, Büsing, Vasilaki & Gerstner 2010,
//! "Connectivity reflects coding: a model of voltage-based STDP with
//! homeostasis"). The postsynaptic membrane voltage u drives plasticity
//! instead of post spike times. Two low-pass traces of u, ū− and ū+, lag it
//! by a step. A pre spike depresses its synapse by A_LTD [ū− − θ−]+, and while
//! u is above θ+ the synapse potentiates by A_LTP x̄ [u − θ+]+ [ū+ − θ−]+,
//! x̄ being a trace of the pre spikes. With `u_ref` set, A_LTD grows with the
//! square of the neuron's mean depolarisation over about a second, keeping
//! busy neurons in check.
//!
//! Voltages are in units of the neuron's threshold above rest. The LIF
//! resets on the step it fires, so the rule sees a spike as one step at
//! `v_spike`. With θ+ at threshold only spikes potentiate, and a spike's lift
//! of ū− depresses the inputs that follow it, as pair STDP does.
//! Weights change once per presentation, by the exact sums of both terms.

use crate::neuron::Neuron;
use crate::params::{StdpParams, VoltageParams};
use snn_core::encode::SpikeTrain;

/// What the last learning presentation changed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Changes {
    /// inputs that spiked, ascending; no other synapse changed
    pub inputs: Vec<u16>,
    /// change of each listed input's synapse onto every neuron, input-major
    pub dw: Vec<f32>,
    /// per neuron, the potentiation and depression it took over all inputs
    pub ltp: Vec<f32>,
    pub ltd: Vec<f32>,
}

impl Changes {
    /// Whether neuron `j`'s weights moved.
    pub fn touched(&self, j: usize) -> bool {
        self.ltp[j] + self.ltd[j] > 0.0
    }
}

// share of the way a trace moves toward its input each step
fn rate(tau: f32) -> f32 {
    if tau > 0.0 {
        1.0 - (-1.0 / tau).exp()
    } else {
        1.0
    }
}

#[derive(Debug, Clone)]
pub struct Voltage {
    p: VoltageParams,
    n_out: usize,
    keep_x: f32,
    rate_minus: f32,
    rate_plus: f32,
    rate_bar: f32,
    /// ū− and ū+ of each neuron as of the last step
    pub minus: Vec<f32>,
    pub plus: Vec<f32>,
    /// each neuron's mean depolarisation over about tau_bar, across
    /// presentations; starts at u_ref
    pub bar: Vec<f32>,
    /// LTD amplitude of each neuron this presentation
    pub a_ltd: Vec<f32>,
    inv: Vec<f32>,
    u: Vec<f32>,
    /// per step, then neuron: LTD per pre spike, and the post factor of LTP,
    /// which `learn` sums backward through the pre trace
    ltd: Vec<f32>,
    ltp: Vec<f32>,
    steps: usize,
    pot: Vec<f32>,
    dep: Vec<f32>,
    pub last: Changes,
}

impl Voltage {
    pub fn new(p: &VoltageParams, n_out: usize) -> Self {
        let keep_x = 1.0 - rate(p.tau_x);
        Self {
            p: *p,
            n_out,
            keep_x,
            rate_minus: rate(p.tau_minus),
            rate_plus: rate(p.tau_plus),
            rate_bar: rate(p.tau_bar),
            minus: vec![0.0; n_out],
            plus: vec![0.0; n_out],
            bar: vec![p.u_ref; n_out],
            a_ltd: vec![p.a_ltd; n_out],
            inv: vec![1.0; n_out],
            u: vec![0.0; n_out],
            ltd: Vec::new(),
            ltp: Vec::new(),
            steps: 0,
            pot: vec![0.0; n_out],
            dep: vec![0.0; n_out],
            last: Changes::default(),
        }
    }

    pub fn params(&self) -> &VoltageParams {
        &self.p
    }

    /// Starts a learning presentation of `steps` steps: traces back at rest,
    /// voltages scaled to each neuron's threshold, and LTD set by the mean
    /// depolarisation so far.
    pub fn begin(&mut self, thresholds: &[f32], rest: f32, steps: usize) {
        let n = self.n_out;
        self.minus.fill(0.0);
        self.plus.fill(0.0);
        for (inv, &th) in self.inv.iter_mut().zip(thresholds) {
            *inv = 1.0 / (th - rest).max(f32::MIN_POSITIVE);
        }
        let (a, r) = (self.p.a_ltd, self.p.u_ref);
        for (a_ltd, &bar) in self.a_ltd.iter_mut().zip(&self.bar) {
            *a_ltd = if r > 0.0 { a * (bar / r).powi(2) } else { a };
        }
        self.steps = steps;
        self.ltd.resize((steps + 1) * n, 0.0);
        self.ltp.resize((steps + 1) * n, 0.0);
    }

    /// Step `t` of the presentation, from each neuron's potential after it
    /// and the neurons that `fired`.
    pub fn step(&mut self, t: usize, neurons: &[Neuron], fired: &[u16], rest: f32) {
        let n = self.n_out;
        for ((u, nrn), &inv) in self.u.iter_mut().zip(neurons).zip(&self.inv) {
            *u = (nrn.p - rest) * inv;
        }
        for &j in fired {
            self.u[j as usize] = self.p.v_spike;
        }
        let (lo, hi) = (self.p.theta_minus, self.p.theta_plus);
        let (rm, rp, rb) = (self.rate_minus, self.rate_plus, self.rate_bar);
        let ltd = self.ltd[t * n..(t + 1) * n].iter_mut();
        let ltp = self.ltp[t * n..(t + 1) * n].iter_mut();
        let traces = self.minus.iter_mut().zip(self.plus.iter_mut());
        let each = traces.zip(self.bar.iter_mut()).zip(ltd.zip(ltp));
        for ((((um, up), bar), (d, g)), (&u, &a)) in each.zip(self.u.iter().zip(&self.a_ltd)) {
            // the traces as of the step before, so a spike does not gate itself
            *d = a * (*um - lo).max(0.0);
            *g = (u - hi).max(0.0) * (*up - lo).max(0.0);
            *um += (u - *um) * rm;
            *up += (u - *up) * rp;
            *bar += (u.max(0.0) - *bar) * rb;
        }
    }

    /// Applies the presentation's plasticity to both copies of the weights,
    /// `weights` row-major and `by_input` input-major, soft bounded as pair
    /// STDP is, and records the changes in `last`.
    pub fn learn(
        &mut self,
        train: &SpikeTrain,
        weights: &mut [f32],
        by_input: &mut [f32],
        s: &StdpParams,
    ) {
        let n = self.n_out;
        let n_in = train.n_inputs();
        // a pre spike at t meets g(t) + keep_x g(t + 1) + ... through its trace
        for t in (1..self.steps).rev() {
            let (now, later) = self.ltp.split_at_mut((t + 1) * n);
            for (g, &next) in now[t * n..].iter_mut().zip(&later[..n]) {
                *g += self.keep_x * next;
            }
        }
        let last = &mut self.last;
        let spiked = train.silent().enumerate().filter(|s| !s.1);
        last.inputs.clear();
        last.inputs.extend(spiked.map(|s| s.0 as u16));
        last.dw.clear();
        last.dw.resize(last.inputs.len() * n, 0.0);
        last.ltp.clear();
        last.ltp.resize(n, 0.0);
        last.ltd.clear();
        last.ltd.resize(n, 0.0);
        let (lo, hi) = (s.w_min, s.w_max);
        let (up_rate, down_rate) = (s.sigma * self.p.a_ltp, s.sigma);
        let (pot, dep) = (&mut self.pot, &mut self.dep);
        for (&i, dw) in last.inputs.iter().zip(last.dw.chunks_exact_mut(n)) {
            pot.fill(0.0);
            dep.fill(0.0);
            for &t in train.times(i as usize) {
                let at = t as usize * n;
                let (g, d) = (&self.ltp[at..at + n], &self.ltd[at..at + n]);
                for ((pv, dv), (&g, &d)) in pot.iter_mut().zip(dep.iter_mut()).zip(g.iter().zip(d))
                {
                    *pv += g;
                    *dv += d;
                }
            }
            // soft bounds as stdp::bounded, both terms against the same weight
            let col = &mut by_input[i as usize * n..][..n];
            let sums = last.ltp.iter_mut().zip(last.ltd.iter_mut());
            let each = col.iter_mut().zip(dw).zip(pot.iter().zip(dep.iter()));
            for (((w, dw), (&pv, &dv)), (lt, ld)) in each.zip(sums) {
                let up = up_rate * pv * (hi - *w);
                let down = down_rate * dv * (*w - lo);
                *dw = up - down;
                *lt += up;
                *ld += down;
                *w = *w + up - down;
            }
        }
        // the row-major copy, for the rows that moved
        for (j, row) in weights.chunks_exact_mut(n_in).enumerate() {
            if last.touched(j) {
                for &i in &last.inputs {
                    row[i as usize] = by_input[i as usize * n + j];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use snn_core::encode::{Coding, EncodeParams};
    use snn_core::Rng;

    // an input per entry spiking every `period` steps (0: never), spikes
    // outside `keep` dropped
    fn periodic(periods: &[usize], keep: std::ops::RangeInclusive<usize>) -> SpikeTrain {
        let ep = EncodeParams::default();
        let pot: Vec<f32> = periods
            .iter()
            .map(|&per| {
                if per == 0 {
                    return 0.0;
                }
                let f = ep.period_scale / (per as f32 - 0.5);
                let k = (f - ep.freq_min) / (ep.freq_max - ep.freq_min);
                ep.pot_min + k * (ep.pot_max - ep.pot_min)
            })
            .collect();
        let mut tr = SpikeTrain::new();
        tr.encode(&pot, 200, &ep, &mut Rng::new(0));
        tr.retime(|t| keep.contains(&(t as usize)).then_some(t as usize));
        tr
    }

    // a presentation in which neuron j sits at potential(j, t) and fires
    // where fires(j, t), thresholds 2 and rest 0; returns the changes
    fn present<P, F>(
        v: &mut Voltage,
        tr: &SpikeTrain,
        w: &mut [f32],
        potential: P,
        fires: F,
    ) -> Vec<f32>
    where
        P: Fn(usize, usize) -> f32,
        F: Fn(usize, usize) -> bool,
    {
        let (n, n_in) = (v.n_out, tr.n_inputs());
        let mut by_input = vec![0.0; w.len()];
        for j in 0..n {
            for i in 0..n_in {
                by_input[i * n + j] = w[j * n_in + i];
            }
        }
        let before = w.to_vec();
        v.begin(&vec![2.0; n], 0.0, tr.steps());
        for t in 1..=tr.steps() {
            let fired: Vec<u16> = (0..n as u16).filter(|&j| fires(j as usize, t)).collect();
            let neurons: Vec<Neuron> = (0..n)
                .map(|j| Neuron {
                    p: if fires(j, t) { 0.0 } else { potential(j, t) },
                    t_rest: -1,
                })
                .collect();
            v.step(t, &neurons, &fired, 0.0);
        }
        v.learn(tr, w, &mut by_input, &StdpParams::default());
        for j in 0..n {
            for i in 0..n_in {
                assert_eq!(by_input[i * n + j], w[j * n_in + i], "copies agree");
            }
        }
        w.iter().zip(&before).map(|(a, b)| a - b).collect()
    }

    // climbs from rest to threshold over 20 steps up to a spike at 100
    fn ramp(t: usize) -> f32 {
        if (80..100).contains(&t) {
            (t - 80) as f32 / 10.0
        } else {
            0.0
        }
    }

    #[test]
    fn matches_a_step_by_step_simulation() {
        let p = VoltageParams {
            u_ref: 0.3,
            ..VoltageParams::default()
        };
        let s = StdpParams::default();
        let (n, n_in) = (3, 10);
        let mut rng = Rng::new(4);
        let pot: Vec<f32> = (0..n_in)
            .map(|i| if i < 2 { 0.0 } else { rng.f32() * 2.5 })
            .collect();
        let ep = EncodeParams {
            coding: Coding::Poisson,
            ..EncodeParams::default()
        };
        let mut tr = SpikeTrain::new();
        tr.encode(&pot, 200, &ep, &mut Rng::new(9));
        let level = |j: usize, t: usize| ((t * (j + 3)) % 23) as f32 / 5.0 - 0.4;
        let fires = |j: usize, t: usize| (t + 7 * j).is_multiple_of(37);
        let bar = [0.1f32, 0.3, 0.6];
        let mut v = Voltage::new(&p, n);
        v.bar = bar.to_vec();
        let w0: Vec<f32> = (0..n * n_in).map(|k| 0.1 + (k % 7) as f32 * 0.2).collect();
        let mut w = w0.clone();
        present(&mut v, &tr, &mut w, level, fires);

        // forward in time: x̄ per input, ū± per neuron, both sums per synapse
        let kx = (-1.0 / p.tau_x).exp();
        let (mut x, mut um, mut up) = (vec![0.0f32; n_in], [0.0f32; 3], [0.0f32; 3]);
        let (mut ltp, mut ltd) = (vec![0.0f32; n * n_in], vec![0.0f32; n * n_in]);
        for t in 1..=tr.steps() {
            x.iter_mut().for_each(|x| *x *= kx);
            tr.active(t).iter().for_each(|&i| x[i as usize] += 1.0);
            for j in 0..n {
                let u = if fires(j, t) {
                    p.v_spike
                } else {
                    level(j, t) / 2.0
                };
                let g = (u - p.theta_plus).max(0.0) * (up[j] - p.theta_minus).max(0.0);
                let d = p.a_ltd * (bar[j] / p.u_ref).powi(2) * (um[j] - p.theta_minus).max(0.0);
                for i in 0..n_in {
                    ltp[j * n_in + i] += x[i] * g;
                }
                for &i in tr.active(t) {
                    ltd[j * n_in + i as usize] += d;
                }
                um[j] += (u - um[j]) * rate(p.tau_minus);
                up[j] += (u - up[j]) * rate(p.tau_plus);
            }
        }
        let mut moved = 0;
        for k in 0..n * n_in {
            let up = s.sigma * p.a_ltp * ltp[k] * (s.w_max - w0[k]);
            let down = s.sigma * ltd[k] * (w0[k] - s.w_min);
            let want = w0[k] + up - down;
            assert!(
                (w[k] - want).abs() < 1e-5,
                "synapse {k}: {} vs {want}",
                w[k]
            );
            moved += (want != w0[k]) as usize;
        }
        assert!(moved >= n * (n_in - 2) - 1, "{moved}");
        assert_eq!(v.last.inputs.len(), n_in - 2, "two silent inputs");
        let total: f32 = v.last.dw.iter().sum();
        let want: f32 = w.iter().zip(&w0).map(|(a, b)| a - b).sum();
        assert!((total - want).abs() < 1e-4);
    }

    #[test]
    fn a_spike_potentiates_the_input_before_it_and_depresses_the_one_after() {
        // θ− at rest, as in the paper
        let p = VoltageParams {
            theta_minus: 0.0,
            ..VoltageParams::default()
        };
        let spike = |_: usize, t: usize| t == 100;
        // one input spiking 5 steps before the post spike, one 4 after, one never
        let before = periodic(&[95, 0], 1..=99);
        let after = periodic(&[52, 0], 101..=150);
        assert_eq!(
            (before.times(0), after.times(0)),
            (&[95u16][..], &[104u16][..])
        );
        let mut v = Voltage::new(&p, 1);
        let dw = present(&mut v, &before, &mut [0.5, 0.5], |_, t| ramp(t), spike);
        assert!(dw[0] > 0.0 && dw[1] == 0.0, "{dw:?}");
        assert!(v.last.ltp[0] > v.last.ltd[0] && v.last.touched(0));
        let dw = present(&mut v, &after, &mut [0.5, 0.5], |_, t| ramp(t), spike);
        assert!(dw[0] < 0.0 && dw[1] == 0.0, "{dw:?}");
        assert_eq!(v.last.ltp[0], 0.0, "no post spike after it");
        // the spike's peak is what deepens the depression after it
        let flat = VoltageParams { v_spike: 1.0, ..p };
        let mut v = Voltage::new(&flat, 1);
        let shallow = present(&mut v, &after, &mut [0.5, 0.5], |_, t| ramp(t), spike);
        assert!(
            shallow[0] < 0.0 && shallow[0] > dw[0],
            "{shallow:?} vs {dw:?}"
        );
    }

    #[test]
    fn depolarisation_past_theta_minus_depresses_and_less_leaves_weights() {
        let tr = periodic(&[30, 45, 0], 1..=200);
        let p = VoltageParams::default();
        assert_eq!(p.theta_minus, 0.4);
        let mut v = Voltage::new(&p, 4);
        // at rest, at 0.3, 0.6 and 0.9 of threshold
        let level = |j: usize, _: usize| [0.0, 0.6, 1.2, 1.8][j];
        let mut w = [0.5; 12];
        let dw = present(&mut v, &tr, &mut w, level, |_, _| false);
        assert_eq!(&dw[..6], &[0.0; 6], "at rest and below θ−");
        assert!(dw[6] < 0.0 && dw[7] < 0.0 && dw[8] == 0.0, "{dw:?}");
        assert!(dw[9] < dw[6], "deeper depolarisation depresses more");
        assert!(dw[6] < dw[7], "the busier input loses more");
        assert_eq!(v.last.ltp, [0.0; 4]);
        assert!(!v.last.touched(1) && v.last.touched(2));
        // inhibited far below rest: nothing either
        let dw = present(&mut v, &tr, &mut w, |_, _| -500.0, |_, _| false);
        assert!(dw.iter().all(|&d| d == 0.0));
    }

    #[test]
    fn subthreshold_potentiation_needs_theta_plus_below_threshold() {
        let tr = periodic(&[30], 1..=200);
        // held at 0.9 of threshold, never firing
        let held = |_: usize, _: usize| 1.8;
        for (theta_plus, potentiates) in [(1.0, false), (0.8, true)] {
            let p = VoltageParams {
                theta_plus,
                a_ltd: 0.0,
                ..VoltageParams::default()
            };
            let mut v = Voltage::new(&p, 1);
            let dw = present(&mut v, &tr, &mut [0.5], held, |_, _| false);
            assert_eq!(dw[0] > 0.0, potentiates, "θ+ {theta_plus}: {dw:?}");
        }
    }

    #[test]
    fn ltd_follows_the_square_of_the_mean_depolarisation() {
        let p = VoltageParams {
            u_ref: 0.2,
            ..VoltageParams::default()
        };
        let mut v = Voltage::new(&p, 2);
        assert_eq!(v.bar, [0.2, 0.2], "starts at the reference");
        v.bar = vec![0.1, 0.4];
        v.begin(&[2.0, 2.0], 0.0, 200);
        let a = p.a_ltd;
        assert!((v.a_ltd[0] - a / 4.0).abs() < 1e-6 && (v.a_ltd[1] - 4.0 * a).abs() < 1e-6);
        // the mean drifts toward the depolarisation it sees, spikes included
        let tr = periodic(&[0, 0], 1..=200);
        let mut w = [0.5; 4];
        for _ in 0..40 {
            present(&mut v, &tr, &mut w, |j, _| [1.0, 0.0][j], |_, _| false);
        }
        assert!(
            (v.bar[0] - 0.5).abs() < 0.01 && v.bar[1] < 0.01,
            "{:?}",
            v.bar
        );
        let fixed = Voltage::new(&VoltageParams { u_ref: 0.0, ..p }, 2);
        assert_eq!(fixed.a_ltd, [a, a]);
    }
}
