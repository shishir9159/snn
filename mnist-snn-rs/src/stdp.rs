//! Spike-timing-dependent plasticity with soft weight bounds.

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
        self.post_spike_rounded(row, train, t, |w| w);
    }

    /// As [`Stdp::post_spike`], with each updated weight passed through `round`
    /// as it is written, the way a low-precision synapse memory stores it.
    pub fn post_spike_rounded<R: FnMut(f32) -> f32>(
        &self,
        row: &mut [f32],
        train: &SpikeTrain,
        t: usize,
        mut round: R,
    ) {
        let lo = t.saturating_sub(self.p.t_back as usize).max(1);
        let hi = (t + self.p.t_fore as usize).min(train.steps());
        for s in lo..=hi {
            let dw = self.table[s + self.p.t_back as usize - t];
            for &i in train.active(s) {
                let w = &mut row[i as usize];
                *w = round(bounded(*w, dw, &self.p));
            }
        }
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
        stdp.post_spike_rounded(&mut row, &tr, 100, |w| q.snap(w, &mut rng));
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
