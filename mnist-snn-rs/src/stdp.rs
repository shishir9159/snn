//! Spike-timing-dependent plasticity with soft weight bounds: the pair rule
//! and the minimal triplet rule.

use crate::params::StdpParams;
use snn_core::encode::SpikeTrain;

/// Weight change for a pre spike `dt = t_pre - t_post` steps from the post spike:
/// potentiation when the pre spike leads (or coincides), depression when it follows.
pub fn kernel(dt: i32, p: &StdpParams) -> f32 {
    if dt > 0 {
        -p.a_plus * (-(dt as f32) / p.tau_plus).exp()
    } else {
        p.a_minus * (dt as f32 / p.tau_minus).exp()
    }
}

/// Moves `w` by `dw` scaled with its distance to the bound it heads for,
/// so weights approach but never cross [w_min, w_max].
pub fn bounded(w: f32, dw: f32, p: &StdpParams) -> f32 {
    if dw > 0.0 {
        w + p.sigma * dw * (p.w_max - w)
    } else {
        w + p.sigma * dw * (w - p.w_min)
    }
}

/// STDP rule with the kernel tabulated over the window.
#[derive(Debug, Clone)]
pub struct Stdp {
    p: StdpParams,
    table: Vec<f32>,
}

impl Stdp {
    pub fn new(p: &StdpParams) -> Self {
        let back = p.t_back as i32;
        let table = (-back..=p.t_fore as i32).map(|dt| kernel(dt, p)).collect();
        Self { p: *p, table }
    }

    pub fn params(&self) -> &StdpParams {
        &self.p
    }

    /// Applies every pre spike within the window around a post spike at step `t`
    /// to one neuron's weight row.
    pub fn post_spike(&self, row: &mut [f32], train: &SpikeTrain, t: usize) {
        self.post_spike_rounded(row, train, t, 1.0, |w| w);
    }

    /// As [`Stdp::post_spike`] with the kernel times `scale`, and each updated
    /// weight passed through `round` as it is written, the way a low-precision
    /// synapse memory stores it. A negative scale punishes: it depresses every
    /// input in the window. Reversing the kernel instead would potentiate the
    /// inputs spiking just after the post spike, which under a rate code are
    /// the same pixels that led it.
    pub fn post_spike_rounded<R: FnMut(f32) -> f32>(
        &self,
        row: &mut [f32],
        train: &SpikeTrain,
        t: usize,
        scale: f32,
        mut round: R,
    ) {
        self.pairs(train, t, |i, k| {
            let dw = if scale < 0.0 {
                scale * k.abs()
            } else {
                scale * k
            };
            let w = &mut row[i];
            *w = round(bounded(*w, dw, &self.p));
        });
    }

    /// Calls `f(input, kernel)` for every pre spike within the window around
    /// a post spike at step `t`.
    pub fn pairs<F: FnMut(usize, f32)>(&self, train: &SpikeTrain, t: usize, mut f: F) {
        for (k, active) in self.window(train, t) {
            active.iter().for_each(|&i| f(i as usize, k));
        }
    }

    /// The kernel and the inputs spiking at each step of the window around a
    /// post spike at step `t`.
    pub fn window<'a>(
        &'a self,
        train: &'a SpikeTrain,
        t: usize,
    ) -> impl Iterator<Item = (f32, &'a [u16])> + 'a {
        let lo = t.saturating_sub(self.p.t_back as usize).max(1);
        let hi = (t + self.p.t_fore as usize).min(train.steps());
        (lo..=hi).map(move |s| (self.table[s + self.p.t_back as usize - t], train.active(s)))
    }

    /// Depresses synapses from inputs that stayed silent for the whole image.
    pub fn depress_silent(&self, row: &mut [f32], train: &SpikeTrain, amount: f32) {
        for (i, w) in row.iter_mut().enumerate() {
            if train.times(i).is_empty() {
                *w = (*w - amount).max(self.p.w_min);
            }
        }
    }
}

/// A post neuron's fast and slow traces as of step `at`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PostTrace {
    fast: f32,
    slow: f32,
    at: usize,
}

/// Minimal triplet STDP (Pfister & Gerstner 2006). A post spike potentiates
/// each synapse by its pre trace times (a2_ltp + a3_ltp × the slow post trace
/// of earlier post spikes), so bursts potentiate and lone spikes hardly do; a
/// pre spike depresses by a2_ltd × the fast post trace. Traces sum exponentials
/// over past spikes and start empty every presentation.
#[derive(Debug, Clone)]
pub struct Triplet {
    p: StdpParams,
    /// exp(-k / tau) for k = 0..=steps, per trace
    pre: Vec<f32>,
    fast: Vec<f32>,
    slow: Vec<f32>,
}

fn decay(table: &[f32], k: usize) -> f32 {
    table.get(k).copied().unwrap_or(0.0)
}

impl Triplet {
    pub fn new(p: &StdpParams, steps: usize) -> Self {
        let table = |tau: f32| (0..=steps).map(|k| (-(k as f32) / tau).exp()).collect();
        Self {
            p: *p,
            pre: table(p.tau_r),
            fast: table(p.tau_o),
            slow: table(p.tau_y),
        }
    }

    /// Pre trace at step `t` of an input spiking at `times`, ascending.
    pub fn pre_trace(&self, times: &[u16], t: usize) -> f32 {
        let past = times.iter().take_while(|&&s| s as usize <= t);
        past.map(|&s| decay(&self.pre, t - s as usize)).sum()
    }

    /// Depression an input spike at step `t` brings to the synapse onto a post
    /// neuron with traces `o` (from its spikes before `t`); 0 if it never fired.
    pub fn pre_spike(&self, o: &PostTrace, t: usize) -> f32 {
        -self.p.a2_ltd * o.fast * decay(&self.fast, t - o.at)
    }

    /// Potentiates `row` for a post spike at step `t`, then adds the spike to
    /// the neuron's traces `o`.
    pub fn post_spike<R: FnMut(f32) -> f32>(
        &self,
        row: &mut [f32],
        train: &SpikeTrain,
        t: usize,
        o: &mut PostTrace,
        mut round: R,
    ) {
        let (fast, slow) = (
            o.fast * decay(&self.fast, t - o.at),
            o.slow * decay(&self.slow, t - o.at),
        );
        let gain = self.p.a2_ltp + self.p.a3_ltp * slow;
        if gain > 0.0 {
            for (i, w) in row.iter_mut().enumerate() {
                let r = self.pre_trace(train.times(i), t);
                if r > 0.0 {
                    *w = round(bounded(*w, gain * r, &self.p));
                }
            }
        }
        *o = PostTrace {
            fast: fast + 1.0,
            slow: slow + 1.0,
            at: t,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use snn_core::encode::{Coding, EncodeParams};
    use snn_core::rng::Rng;

    fn p() -> StdpParams {
        StdpParams::default()
    }

    #[test]
    fn kernel_sign_and_decay() {
        let p = p();
        assert_eq!(kernel(0, &p), p.a_minus);
        assert!(kernel(-1, &p) > kernel(-10, &p));
        assert!(kernel(-10, &p) > 0.0);
        assert!(kernel(1, &p) < 0.0);
        assert!(kernel(1, &p) < kernel(10, &p));
    }

    #[test]
    fn soft_bounds_hold() {
        let p = p();
        let mut w = 0.0;
        for _ in 0..1000 {
            w = bounded(w, 1.0, &p);
        }
        assert!(w <= p.w_max && w > p.w_max - 0.01);
        for _ in 0..1000 {
            w = bounded(w, -1.0, &p);
        }
        assert!(w >= p.w_min && w < p.w_min + 0.01);
    }

    #[test]
    fn depression_lowers_any_weight() {
        // the reference bound (w - |w_min|) raised weights below |w_min|
        let p = StdpParams { w_min: -1.2, ..p() };
        for w in [-1.0, 0.0, 0.5, 1.9] {
            assert!(bounded(w, -0.5, &p) < w);
            assert!(bounded(w, 0.5, &p) > w);
        }
    }

    fn train(pot: &[f32]) -> SpikeTrain {
        let ep = EncodeParams {
            coding: Coding::Regular,
            ..EncodeParams::default()
        };
        let mut tr = SpikeTrain::new();
        tr.encode(pot, 200, &ep, &mut Rng::new(0));
        tr
    }

    #[test]
    fn post_spike_potentiates_leading_and_depresses_following() {
        // input 0 spikes at 95 and 190, input 1 is silent
        let tr = train(&[0.01, 0.0]);
        assert_eq!(tr.times(0), &[95, 190]);
        let stdp = Stdp::new(&p());

        let mut row = [0.5, 0.5];
        stdp.post_spike(&mut row, &tr, 100);
        assert!(row[0] > 0.5);
        assert_eq!(row[1], 0.5);

        let mut row = [0.5, 0.5];
        stdp.post_spike(&mut row, &tr, 90);
        assert!(row[0] < 0.5);

        let mut row = [0.5, 0.5];
        stdp.post_spike(&mut row, &tr, 140);
        assert_eq!(row[0], 0.5, "no pre spike within the window");
    }

    #[test]
    fn rounded_updates_land_on_the_grid_and_leave_the_rest() {
        let tr = train(&[0.01, 0.0, 0.01]);
        let stdp = Stdp::new(&p());
        let q = snn_core::Quantizer::new(3, p().w_min, p().w_max, true).unwrap();
        let mut rng = Rng::new(5);
        let mut row = [0.5, 0.3, 0.7];
        stdp.post_spike_rounded(&mut row, &tr, 100, 1.0, |w| q.snap(w, &mut rng));
        let on_grid = |w: f32| {
            let t = (w - p().w_min) / q.step();
            (t - t.round()).abs() < 1e-3
        };
        assert!(on_grid(row[0]) && on_grid(row[2]));
        assert_eq!(row[1], 0.3, "the silent input was not written");
        let mut plain = [0.5, 0.3, 0.7];
        stdp.post_spike(&mut plain, &tr, 100);
        assert!(
            (row[0] - plain[0]).abs() <= q.step(),
            "within a step of the exact update"
        );
    }

    #[test]
    fn negative_scale_depresses_both_sides_of_the_spike() {
        let tr = train(&[0.01, 0.0]);
        let stdp = Stdp::new(&p());
        // input 0 leads a post spike at 100 and follows one at 90
        for (t, pair_sign) in [(100, 1.0), (90, -1.0)] {
            let (mut pair, mut punished) = ([0.5, 0.5], [0.5, 0.5]);
            stdp.post_spike_rounded(&mut pair, &tr, t, 1.0, |w| w);
            stdp.post_spike_rounded(&mut punished, &tr, t, -1.0, |w| w);
            assert_eq!((pair[0] - 0.5).signum(), pair_sign, "at {t}");
            assert!(
                punished[0] < 0.5 && punished[1] == 0.5,
                "at {t}: {punished:?}"
            );
        }
    }

    #[test]
    fn triplet_potentiates_bursts_and_depresses_late_inputs() {
        let tr = train(&[0.01, 0.0]);
        assert_eq!(tr.times(0), &[95, 190]);
        let t = Triplet::new(&p(), 200);
        let mut o = PostTrace::default();
        assert_eq!(t.pre_spike(&o, 50), 0.0, "no post spike yet, no depression");
        assert!((t.pre_trace(tr.times(0), 100) - (-5.0f32 / 16.8).exp()).abs() < 1e-6);
        assert_eq!(t.pre_trace(tr.times(0), 94), 0.0);
        // a lone post spike leaves the weights: minimal rule, no pair potentiation
        let mut row = [0.5, 0.5];
        t.post_spike(&mut row, &tr, 100, &mut o, |w| w);
        assert_eq!(row, [0.5, 0.5]);
        // the second spike of a burst potentiates the input that led it
        t.post_spike(&mut row, &tr, 110, &mut o, |w| w);
        assert!(row[0] > 0.5 && row[1] == 0.5, "{row:?}");
        // an input spiking after the burst is depressed by the fast trace
        let d = t.pre_spike(&o, 120);
        assert!(d < 0.0 && d > t.pre_spike(&o, 111), "the trace decays");
    }

    #[test]
    fn window_clips_at_train_edges() {
        let tr = train(&[10.0]);
        let stdp = Stdp::new(&p());
        let mut row = [0.0];
        stdp.post_spike(&mut row, &tr, 1);
        stdp.post_spike(&mut row, &tr, 200);
        assert!(row[0] != 0.0);
    }

    #[test]
    fn silent_inputs_are_depressed_to_floor() {
        let tr = train(&[10.0, 0.0]);
        let stdp = Stdp::new(&p());
        let mut row = [0.5, p().w_min + 0.01];
        stdp.depress_silent(&mut row, &tr, 0.06);
        assert_eq!(row[0], 0.5);
        assert_eq!(row[1], p().w_min);
    }
}
