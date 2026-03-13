//! Delayed reward through eligibility traces (Izhikevich 2007, "Solving the
//! distal reward problem through linkage of STDP and dopamine signaling").
//! STDP does not move a weight here: every pre/post pairing tags the synapse
//! with eligibility c, which fades with tau_c. An answer's reward arrives as a
//! puff of dopamine d some delay later and fades with tau_d, and each weight
//! follows dw/dt = c d. By then later answers have tagged synapses of their
//! own, and the puff reaches all of them in proportion to their tags.
//!
//! Tags are kept apart by sign: LTP tags move with d and LTD tags with -|d|,
//! so punishment depresses the whole window, as in the immediate rule.
//! Weights change once per presentation, by the exact sum of c d over its steps.

use crate::params::DopamineParams;
use crate::stdp::Stdp;
use snn_core::encode::SpikeTrain;
use snn_core::Rng;

/// A row is dropped once its last tag has faded to this share.
const CUTOFF: f32 = 0.01;

/// An answer, rewarded once the presentation ends.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    pub row: usize,
    /// step of the presentation it was made at
    pub step: usize,
    /// reward if positive, punishment if negative
    pub amount: f32,
}

/// A reward on its way; times are steps of training.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Puff {
    /// training presentation that earned it, from 0
    pub image: u64,
    pub row: usize,
    pub sent: u64,
    pub at: u64,
    pub amount: f32,
}

/// What one training presentation did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Presented {
    pub image: u64,
    /// training time before its first step, and its steps without the gap
    pub start: u64,
    pub steps: usize,
    pub sent: Option<Puff>,
    /// rewards that arrived, each with the share of all tags its row held then
    pub arrived: Vec<(Puff, f32)>,
    /// size of the tags it made, as of its last step
    pub tagged: f32,
}

#[derive(Debug, Clone)]
pub struct Dopamine {
    p: DopamineParams,
    n_in: usize,
    n_out: usize,
    /// per-step decay of a tag and of dopamine
    keep_c: f32,
    keep_d: f32,
    /// steps after its last tag a row is dropped
    max_age: u64,
    rng: Rng,
    /// training steps so far
    pub clock: u64,
    /// training presentations so far
    pub images: u64,
    /// concentration after the last step
    pub level: f32,
    pending: Vec<Puff>,
    /// per row its LTP and LTD tags, both positive, as of `clock`
    plus: Vec<f32>,
    minus: Vec<f32>,
    /// L1 norm of each row's tags as of `clock`, and the step of its last tag
    norm: Vec<f32>,
    tagged: Vec<u64>,
    held: Vec<bool>,
    live: Vec<usize>,
    pub last: Presented,
    // scratch
    decay: Vec<f32>,
    levels: Vec<f32>,
    suf: Vec<(f32, f32)>,
    dw: Vec<f32>,
    /// (row, step, size) of each tag this presentation, and the norms before it
    tags: Vec<(usize, usize, f32)>,
    before: Vec<(usize, f32)>,
}

fn keep(tau: f32) -> f32 {
    (-1.0 / tau.max(f32::MIN_POSITIVE)).exp()
}

impl Dopamine {
    pub fn new(p: &DopamineParams, n_in: usize, n_out: usize) -> Self {
        let age = p.tau_c.max(0.0) * (1.0 / CUTOFF).ln();
        Self {
            p: *p,
            n_in,
            n_out,
            keep_c: keep(p.tau_c),
            keep_d: keep(p.tau_d),
            max_age: age.ceil() as u64,
            rng: Rng::new(0xd0a),
            clock: 0,
            images: 0,
            level: 0.0,
            pending: Vec::new(),
            plus: Vec::new(),
            minus: Vec::new(),
            norm: vec![0.0; n_out],
            tagged: vec![0; n_out],
            held: vec![false; n_out],
            live: Vec::new(),
            last: Presented::default(),
            decay: Vec::new(),
            levels: Vec::new(),
            suf: Vec::new(),
            dw: vec![0.0; n_in],
            tags: Vec::new(),
            before: Vec::new(),
        }
    }

    pub fn params(&self) -> &DopamineParams {
        &self.p
    }

    /// Rows holding tags.
    pub fn live(&self) -> &[usize] {
        &self.live
    }

    /// A row's LTP and LTD tags as of `clock`.
    pub fn tags(&self, row: usize) -> (&[f32], &[f32]) {
        if self.plus.is_empty() {
            return (&[], &[]);
        }
        let at = row * self.n_in..(row + 1) * self.n_in;
        (&self.plus[at.clone()], &self.minus[at])
    }

    /// Size of a row's tags as of `clock`, and the steps since its last tag.
    pub fn trace(&self, row: usize) -> (f32, u64) {
        (self.norm[row], self.clock - self.tagged[row])
    }

    /// Rewards on their way, soonest first.
    pub fn pending(&self) -> &[Puff] {
        &self.pending
    }

    // sums over the steps after `from` up to `end` of keep_c^(s - from - 1)
    // times the dopamine and times its size, and the dopamine at `end`. With
    // nothing to tag, dopamine only decays between the `puffs`, so each stretch
    // is a geometric series
    fn gap(
        &self,
        level: f32,
        from: u64,
        end: u64,
        puffs: &[Puff],
        pulse: f32,
    ) -> ((f32, f32), f32) {
        let (kc, kd) = (self.keep_c as f64, self.keep_d as f64);
        let q = kc * kd;
        let series = |n: i32| {
            if q > 1.0 - 1e-12 {
                n as f64
            } else {
                (1.0 - q.powi(n)) / (1.0 - q)
            }
        };
        let (mut a, mut b, mut l) = (0.0, 0.0, level as f64);
        let (mut at, mut k) = (from, 0);
        while at < end {
            let stop = puffs.get(k).map_or(end, |p| p.at.min(end));
            let m = (stop - at) as i32;
            if m > 1 {
                let w = kc.powi((at - from) as i32) * kd * series(m - 1);
                a += l * w;
                b += l.abs() * w;
            }
            l *= kd.powi(m);
            while puffs.get(k).is_some_and(|p| p.at == stop) {
                l += (puffs[k].amount * pulse) as f64;
                k += 1;
            }
            let w = kc.powi((stop - from - 1) as i32);
            a += l * w;
            b += l.abs() * w;
            at = stop;
        }
        ((a as f32, b as f32), l as f32)
    }

    /// One training presentation of `train` and the gap after it. The post
    /// spikes (row, step) tag their synapses through `stdp`, and the answering
    /// row's silent inputs get an LTD tag of `silent`; `decision` sends its
    /// reward on its way; dopamine arriving meanwhile is released, and each
    /// tagged row's weight change goes to `apply(row, dw, rewarded)`,
    /// `rewarded` telling whether the dopamine reaching its tags was more
    /// reward than punishment.
    pub fn present<F>(
        &mut self,
        stdp: &Stdp,
        train: &SpikeTrain,
        spikes: &[(u16, u16)],
        decision: Option<Decision>,
        silent: f32,
        mut apply: F,
    ) where
        F: FnMut(usize, &[f32], bool),
    {
        let (n, steps) = (self.n_in, train.steps());
        let period = steps + self.p.gap as usize;
        if self.plus.is_empty() {
            self.plus = vec![0.0; n * self.n_out];
            self.minus = vec![0.0; n * self.n_out];
        }
        if self.decay.len() != period + 1 {
            let tau = self.p.tau_c.max(f32::MIN_POSITIVE);
            self.decay = (0..=period).map(|k| (-(k as f32) / tau).exp()).collect();
        }
        let (start, end) = (self.clock, self.clock + period as u64);
        let mut last = Presented {
            image: self.images,
            start,
            steps,
            ..Presented::default()
        };
        if let Some(d) = decision {
            let extra = match self.p.jitter {
                0 => 0,
                j => self.rng.below(j as usize + 1) as u64,
            };
            let sent = start + d.step as u64;
            let puff = Puff {
                image: self.images,
                row: d.row,
                sent,
                at: sent + self.p.delay as u64 + extra,
                amount: d.amount,
            };
            self.pending.push(puff);
            last.sent = Some(puff);
        }

        // dopamine after each step, a puff scaled so that a tag made as it
        // arrives meets r of it in all: the immediate rule's reward
        let pulse = self.p.gain * (1.0 - self.keep_c * self.keep_d);
        self.pending.sort_by_key(|p| p.at);
        let due = self.pending.iter().take_while(|p| p.at <= end).count();
        self.levels.clear();
        self.levels.push(self.level);
        let (mut level, mut next) = (self.level, 0);
        for s in 1..=steps as u64 {
            level *= self.keep_d;
            while next < due && self.pending[next].at <= start + s {
                level += self.pending[next].amount * pulse;
                next += 1;
            }
            self.levels.push(level);
        }
        // from step t on, a tag fades by keep_c a step: suf[t] sums what it
        // meets, the signed dopamine for LTP tags and its size for LTD tags
        let gap = &self.pending[next..due];
        let (tail, level) = self.gap(level, start + steps as u64, end, gap, pulse);
        self.suf.clear();
        self.suf.resize(steps + 2, tail);
        for s in (1..=steps).rev() {
            let (l, (a, b)) = (self.levels[s], self.suf[s + 1]);
            self.suf[s] = (l + self.keep_c * a, l.abs() + self.keep_c * b);
        }
        let (gp, gm) = (self.keep_c * self.suf[1].0, self.keep_c * self.suf[1].1);

        self.before.clear();
        self.before
            .extend(self.live.iter().map(|&j| (j, self.norm[j])));
        self.tags.clear();
        let mut rows = std::mem::take(&mut self.live);
        let fresh = spikes
            .iter()
            .map(|s| s.0 as usize)
            .chain(decision.map(|d| d.row));
        for j in fresh {
            if !self.held[j] {
                self.held[j] = true;
                rows.push(j);
            }
        }
        let fade = self.decay[period];
        for &j in &rows {
            let at = j * n..(j + 1) * n;
            let (plus, minus) = (&mut self.plus[at.clone()], &mut self.minus[at]);
            let dw = &mut self.dw;
            // the old tags meet the whole presentation's dopamine
            let mut drive = gp * self.norm[j];
            let mut grown = 0.0;
            for ((d, p), m) in dw.iter_mut().zip(plus.iter_mut()).zip(minus.iter_mut()) {
                *d = gp * *p - gm * *m;
                *p *= fade;
                *m *= fade;
            }
            for &(_, t) in spikes.iter().filter(|s| s.0 as usize == j) {
                let t = t as usize;
                let (sp, sm) = self.suf[t];
                let late = self.decay[period - t];
                let mut size = 0.0;
                for (k, active) in stdp.window(train, t) {
                    let (g, tags) = if k >= 0.0 {
                        (sp, &mut *plus)
                    } else {
                        (sm, &mut *minus)
                    };
                    for &i in active {
                        dw[i as usize] += g * k;
                        tags[i as usize] += late * k.abs();
                    }
                    size += k.abs() * active.len() as f32;
                }
                drive += sp * size;
                grown += late * size;
                self.tags.push((j, t, size));
                self.tagged[j] = start + t as u64;
            }
            if let Some(d) = decision.filter(|d| d.row == j && silent > 0.0) {
                let (sp, sm) = self.suf[d.step];
                let late = self.decay[period - d.step];
                let mut quiet = 0;
                for ((d, m), off) in dw.iter_mut().zip(minus.iter_mut()).zip(train.silent()) {
                    let q = silent * off as u8 as f32;
                    *d -= sm * q;
                    *m += late * q;
                    quiet += off as usize;
                }
                let size = silent * quiet as f32;
                drive += sp * size;
                grown += late * size;
                self.tags.push((j, d.step, size));
                self.tagged[j] = self.tagged[j].max(start + d.step as u64);
            }
            apply(j, dw, drive > 0.0);
            self.norm[j] = fade * self.norm[j] + grown;
            if end - self.tagged[j] > self.max_age {
                plus.fill(0.0);
                minus.fill(0.0);
                self.norm[j] = 0.0;
                self.held[j] = false;
            }
        }
        rows.retain(|&j| self.held[j]);
        self.live = rows;

        // each arriving puff meets the tags as they stood at its step
        for puff in self.pending.drain(..due) {
            let s = (puff.at - start) as usize;
            let old = self
                .before
                .iter()
                .map(|&(j, size)| (j, size * self.decay[s]));
            let new = self.tags.iter().filter(|t| t.1 <= s);
            let new = new.map(|t| (t.0, t.2 * self.decay[s - t.1]));
            let (mut all, mut mine) = (0.0, 0.0);
            for (j, size) in old.chain(new) {
                all += size;
                if j == puff.row {
                    mine += size;
                }
            }
            let credit = if all > 0.0 { mine / all } else { 0.0 };
            last.arrived.push((puff, credit));
        }
        let mine = self.tags.iter().map(|t| t.2 * self.decay[steps - t.1]);
        last.tagged = mine.sum();
        self.last = last;
        self.level = level;
        self.clock = end;
        self.images += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::StdpParams;
    use snn_core::encode::{Coding, EncodeParams};

    fn train(pot: &[f32], coding: Coding, seed: u64) -> SpikeTrain {
        let ep = EncodeParams {
            coding,
            ..EncodeParams::default()
        };
        let mut tr = SpikeTrain::new();
        tr.encode(pot, 200, &ep, &mut Rng::new(seed));
        tr
    }

    // evenly spaced: 1.0 spikes every 54 steps, 10.0 every 30
    fn regular(pot: &[f32]) -> SpikeTrain {
        train(pot, Coding::Regular, 0)
    }

    fn params(tau_c: f32, delay: u32, jitter: u32) -> DopamineParams {
        DopamineParams {
            tau_c,
            tau_d: 50.0,
            delay,
            jitter,
            gain: 1.0,
            gap: 0,
        }
    }

    /// The same model one step at a time: tags join the traces at their post
    /// spike, every trace and the dopamine decay each step, and each step adds
    /// d times the LTP tags less |d| times the LTD tags.
    struct Naive {
        plus: Vec<f32>,
        minus: Vec<f32>,
        level: f32,
        clock: u64,
        dw: Vec<f32>,
    }

    impl Naive {
        #[allow(clippy::too_many_arguments)]
        fn present(
            &mut self,
            p: &DopamineParams,
            stdp: &Stdp,
            tr: &SpikeTrain,
            spikes: &[(u16, u16)],
            puffs: &[Puff],
            decision: Option<Decision>,
            silent: f32,
        ) {
            let n = tr.n_inputs();
            let (kc, kd) = (keep(p.tau_c), keep(p.tau_d));
            let period = tr.steps() + p.gap as usize;
            for s in 1..=period {
                let now = self.clock + s as u64;
                self.level *= kd;
                for q in puffs.iter().filter(|q| q.at == now) {
                    self.level += p.gain * q.amount * (1.0 - kc * kd);
                }
                self.plus.iter_mut().for_each(|c| *c *= kc);
                self.minus.iter_mut().for_each(|c| *c *= kc);
                for &(j, t) in spikes.iter().filter(|x| x.1 as usize == s) {
                    let j = j as usize;
                    stdp.pairs(tr, t as usize, |i, k| {
                        if k >= 0.0 {
                            self.plus[j * n + i] += k;
                        } else {
                            self.minus[j * n + i] -= k;
                        }
                    });
                }
                if let Some(d) = decision.filter(|d| d.step == s) {
                    for i in (0..n).filter(|&i| tr.times(i).is_empty()) {
                        self.minus[d.row * n + i] += silent;
                    }
                }
                let d = self.level;
                for (w, (p, m)) in self.dw.iter_mut().zip(self.plus.iter().zip(&self.minus)) {
                    *w += d * p - d.abs() * m;
                }
            }
            self.clock += period as u64;
        }
    }

    #[test]
    fn matches_a_step_by_step_simulation() {
        step_by_step(params(400.0, 150, 120));
        // puffs landing in the gap, summed there in closed form
        step_by_step(DopamineParams {
            gain: 3.0,
            gap: 250,
            ..params(400.0, 150, 300)
        });
    }

    fn step_by_step(p: DopamineParams) {
        let (n, rows) = (12, 3);
        let stdp = Stdp::new(&StdpParams::default());
        let mut dop = Dopamine::new(&p, n, rows);
        let mut naive = Naive {
            plus: vec![0.0; n * rows],
            minus: vec![0.0; n * rows],
            level: 0.0,
            clock: 0,
            dw: vec![0.0; n * rows],
        };
        let mut total = vec![0.0; n * rows];
        let mut rng = Rng::new(3);
        let mut sent = Vec::new();
        for image in 0..8u64 {
            let pot: Vec<f32> = (0..n)
                .map(|i| if i < 2 { 0.0 } else { rng.f32() })
                .collect();
            let tr = train(&pot, Coding::Poisson, image);
            let row = rng.below(rows) as u16;
            let first = 20 + rng.below(60) as u16;
            let spikes = [(row, first), (row, first + 40), ((row + 1) % 3, first + 90)];
            let decision = Some(Decision {
                row: row as usize,
                step: first as usize,
                amount: if image % 3 == 0 { -0.5 } else { 0.7 },
            });
            dop.present(&stdp, &tr, &spikes, decision, 0.2, |j, dw, _| {
                total[j * n..][..n]
                    .iter_mut()
                    .zip(dw)
                    .for_each(|(t, d)| *t += d);
            });
            sent.extend(dop.last.sent);
            let puffs = sent.clone();
            naive.present(&p, &stdp, &tr, &spikes, &puffs, decision, 0.2);
            assert_eq!(dop.clock, naive.clock);
            assert!(
                (dop.level - naive.level).abs() < 1e-5,
                "{} vs {}",
                dop.level,
                naive.level
            );
        }
        assert!(dop.pending().len() < sent.len(), "some rewards arrived");
        let top = naive.dw.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(top > 1e-3);
        for (a, b) in total.iter().zip(&naive.dw) {
            assert!((a - b).abs() < 1e-4 * top, "{a} vs {b}");
        }
        for j in 0..rows {
            let (plus, minus) = dop.tags(j);
            for i in 0..n {
                assert!((plus[i] - naive.plus[j * n + i]).abs() < 1e-4);
                assert!((minus[i] - naive.minus[j * n + i]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn rewards_arrive_after_their_delay() {
        let stdp = Stdp::new(&StdpParams::default());
        let tr = regular(&[1.0, 0.5, 0.0]);
        let mut dop = Dopamine::new(&params(1000.0, 450, 100), 3, 2);
        let decision = Decision {
            row: 1,
            step: 50,
            amount: 1.0,
        };
        dop.present(&stdp, &tr, &[(1, 50)], Some(decision), 0.0, |_, _, _| {});
        let puff = dop.last.sent.unwrap();
        assert_eq!((puff.sent, puff.row, puff.image), (50, 1, 0));
        assert!((500..=600).contains(&puff.at), "{}", puff.at);
        assert_eq!(dop.pending(), &[puff]);
        let mut seen = 0.0;
        dop.present(&stdp, &tr, &[], None, 0.0, |_, dw, rewarded| {
            seen = dw.iter().fold(seen, |m: f32, d| m.max(*d));
            assert!(!rewarded, "the puff is still on its way");
        });
        assert_eq!(seen, 0.0);
        let mut rewarded = false;
        dop.present(&stdp, &tr, &[], None, 0.0, |j, _, r| {
            rewarded |= j == 1 && r
        });
        assert!(rewarded && dop.pending().is_empty());
        let (p, c) = dop.last.arrived[0];
        assert_eq!((p, c), (puff, 1.0), "the only tagged row takes it all");
    }

    #[test]
    fn credit_follows_the_tags_still_held() {
        let stdp = Stdp::new(&StdpParams::default());
        let tr = regular(&[1.0; 4]);
        let mut dop = Dopamine::new(&params(200.0, 350, 0), 4, 3);
        let reward = |row| Decision {
            row,
            step: 100,
            amount: 1.0,
        };
        // row 0 answers, row 2 answers the next image, then row 0's reward lands
        dop.present(&stdp, &tr, &[(0, 100)], Some(reward(0)), 0.0, |_, _, _| {});
        dop.present(&stdp, &tr, &[(2, 100)], Some(reward(2)), 0.0, |_, _, _| {});
        dop.present(&stdp, &tr, &[], None, 0.0, |_, _, _| {});
        let (puff, credit) = dop.last.arrived[0];
        assert_eq!(puff.row, 0);
        // row 2's tags are 200 steps younger
        let (n0, n2) = (dop.trace(0).0, dop.trace(2).0);
        let want = n0 / (n0 + n2);
        assert!((credit - want).abs() < 0.02, "{credit} vs {want}");
        assert!(credit < 0.5, "the later answer holds more of the tags");
    }

    #[test]
    fn punishment_depresses_both_kinds_of_tag() {
        let stdp = Stdp::new(&StdpParams::default());
        // input 0 spikes at 95, input 1 at 110, input 2 never
        let tr = regular(&[0.01, 0.96, 0.0]);
        assert_eq!((tr.times(0)[0], tr.times(1)[1]), (95, 110));
        for amount in [1.0, -1.0] {
            let mut dop = Dopamine::new(&params(1000.0, 0, 0), 3, 1);
            let d = Decision {
                row: 0,
                step: 100,
                amount,
            };
            let mut dw = Vec::new();
            dop.present(&stdp, &tr, &[(0, 100)], Some(d), 0.1, |_, w, _| {
                dw = w.to_vec()
            });
            let (plus, minus) = dop.tags(0);
            assert!(plus[0] > 0.0 && minus[0] == 0.0, "led the spike");
            assert!(plus[1] == 0.0 && minus[1] > 0.0, "followed it");
            assert!(plus[2] == 0.0 && minus[2] > 0.0, "silent");
            let want = if amount > 0.0 {
                [1.0, -1.0, -1.0]
            } else {
                [-1.0; 3]
            };
            assert_eq!(
                dw.iter().map(|d| d.signum()).collect::<Vec<_>>(),
                want,
                "under {amount}"
            );
        }
    }

    #[test]
    fn faded_rows_are_dropped() {
        let stdp = Stdp::new(&StdpParams::default());
        let tr = regular(&[10.0, 10.0]);
        let mut dop = Dopamine::new(&params(100.0, 0, 0), 2, 2);
        dop.present(&stdp, &tr, &[(1, 10)], None, 0.0, |_, _, _| {});
        assert_eq!(dop.live(), &[1]);
        assert!(dop.trace(1).0 > 0.0);
        let mut calls = 0;
        for _ in 0..3 {
            dop.present(&stdp, &tr, &[], None, 0.0, |_, _, _| calls += 1);
        }
        assert!(dop.live().is_empty() && calls == 2, "{calls}");
        assert_eq!(dop.trace(1).0, 0.0);
        assert!(dop.tags(1).0.iter().all(|&c| c == 0.0));
    }
}
