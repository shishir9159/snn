//! A population of leaky integrate-and-fire units with a filtered input
//! current and a spike trace, stored as parallel arrays.

use crate::params::Params;

#[derive(Debug, Clone)]
pub struct Lif {
    /// input current
    pub j: Vec<f32>,
    /// membrane potential
    pub v: Vec<f32>,
    /// low-pass filtered spikes
    pub z: Vec<f32>,
    /// steps left in the refractory period
    refr: Vec<u16>,
    /// units that spiked in the last step, ascending
    pub spikes: Vec<u16>,
    /// spikes per unit since the last reset
    pub counts: Vec<u32>,
    /// firing threshold per unit, adapted by homeostasis
    pub thr: Vec<f32>,
}

impl Lif {
    pub fn new(n: usize, thr: f32) -> Self {
        Self {
            thr: vec![thr; n],
            j: vec![0.0; n],
            v: vec![0.0; n],
            z: vec![0.0; n],
            refr: vec![0; n],
            spikes: Vec::with_capacity(n),
            counts: vec![0; n],
        }
    }

    pub fn len(&self) -> usize {
        self.v.len()
    }

    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
    }

    pub fn reset(&mut self) {
        for x in [&mut self.j, &mut self.v, &mut self.z] {
            x.fill(0.0);
        }
        self.refr.fill(0);
        self.counts.fill(0);
        self.spikes.clear();
    }

    /// One Euler step: the current follows `drive`, the membrane follows the
    /// current; units over threshold spike, reset to 0 and go refractory.
    pub fn step(&mut self, drive: &[f32], p: &Params) {
        let (kj, km) = (p.dt / p.tau_j, p.dt / p.tau_m);
        let t_ref = (p.t_ref / p.dt).round() as u16;
        let decay = p.trace_decay();
        self.spikes.clear();
        for (i, &d) in drive.iter().enumerate() {
            let j = self.j[i] + kj * (d - p.kappa * self.j[i]);
            self.j[i] = j;
            self.z[i] *= decay;
            if self.refr[i] > 0 {
                self.refr[i] -= 1;
                continue;
            }
            let v = self.v[i] + km * (p.resistance * j - p.gamma * self.v[i]);
            if v >= self.thr[i] {
                self.v[i] = 0.0;
                self.refr[i] = t_ref;
                self.z[i] += 1.0;
                self.counts[i] += 1;
                self.spikes.push(i as u16);
            } else {
                self.v[i] = v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_drive_fires_regularly() {
        let p = Params {
            resistance: 1.0,
            ..Params::default()
        };
        let mut l = Lif::new(2, p.v_thr);
        let mut fired = [0; 2];
        for _ in 0..200 {
            l.step(&[2.0, 0.0], &p);
            for &i in &l.spikes {
                fired[i as usize] += 1;
            }
        }
        assert!(fired[0] > 10, "{fired:?}");
        assert_eq!(fired[1], 0);
        assert_eq!(l.counts[0], fired[0]);
        assert!(l.z[0] > 0.0 && l.z[1] == 0.0);
    }

    #[test]
    fn refractory_unit_skips_a_step() {
        let p = Params {
            t_ref: 2.0,
            ..Params::default()
        };
        let mut l = Lif::new(1, p.v_thr);
        let mut last = None;
        for t in 0..100 {
            l.step(&[1000.0], &p);
            if !l.spikes.is_empty() {
                if let Some(prev) = last {
                    assert!(t - prev >= 3, "spikes at {prev} and {t}");
                }
                last = Some(t);
            }
        }
        assert!(last.is_some());
    }

    #[test]
    fn reset_clears_state() {
        let p = Params::default();
        let mut l = Lif::new(3, p.v_thr);
        l.step(&[100.0; 3], &p);
        l.reset();
        assert!(l.v.iter().chain(&l.j).chain(&l.z).all(|&x| x == 0.0));
        assert!(l.counts.iter().all(|&c| c == 0));
    }
}
