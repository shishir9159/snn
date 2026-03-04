//! Predictive Coding Light. Simple cells learn edge detectors from ON/OFF events
//! by excitatory STDP. Plastic lateral (simple to simple) and top-down (complex
//! to simple) inhibition learn by inhibitory STDP to cancel the spikes the
//! network can already predict, so fewer, less redundant spikes go up. No error
//! is sent upward. Cells are event-driven LIF: a potential decays only when
//! something touches it.

use crate::params::Params;
use snn_core::{weights, Activity, Rng};

const NEVER: i32 = -1_000_000;

/// Event-driven leaky integrate-and-fire cells.
#[derive(Debug, Clone)]
pub struct Cells {
    pub v: Vec<f32>,
    /// step each potential was last brought up to date
    seen: Vec<u32>,
    /// last and previous spike step
    pub last: Vec<i32>,
    pub prev: Vec<i32>,
    /// spikes since the last reset
    pub counts: Vec<u32>,
    tau: f32,
}

impl Cells {
    fn new(n: usize, tau: f32) -> Self {
        Self {
            v: vec![0.0; n],
            seen: vec![0; n],
            last: vec![NEVER; n],
            prev: vec![NEVER; n],
            counts: vec![0; n],
            tau,
        }
    }

    pub fn len(&self) -> usize {
        self.v.len()
    }

    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
    }

    fn reset(&mut self) {
        self.v.fill(0.0);
        self.seen.fill(0);
        self.last.fill(NEVER);
        self.prev.fill(NEVER);
        self.counts.fill(0);
    }

    /// Potential of cell `i` at step `t`, decayed toward rest since last seen.
    pub fn at(&mut self, i: usize, t: u32) -> f32 {
        let dt = t.saturating_sub(self.seen[i]);
        if dt > 0 {
            self.v[i] *= (-(dt as f32) / self.tau).exp();
            self.seen[i] = t;
        }
        self.v[i]
    }

    /// Potential of cell `i` at step `t` without bringing it up to date.
    pub fn peek(&self, i: usize, t: u32) -> f32 {
        let dt = t.saturating_sub(self.seen[i]);
        self.v[i] * (-(dt as f32) / self.tau).exp()
    }

    fn fire(&mut self, i: usize, t: u32, reset: f32) {
        self.prev[i] = self.last[i];
        self.last[i] = t as i32;
        self.v[i] = reset;
        self.counts[i] += 1;
    }
}

#[derive(Debug, Clone)]
pub struct Net {
    pub p: Params,
    /// event frame side
    pub side: usize,
    /// simple-cell locations per side
    pub n: usize,
    /// complex-cell locations per side
    pub m: usize,
    /// shared simple-cell features: feature, polarity, row, column
    pub w_s: Vec<f32>,
    /// lateral inhibition into each simple cell: neighbour offset, feature
    pub lat: Vec<f32>,
    /// top-down inhibition into each simple cell: covering complex location, cell
    pub td: Vec<f32>,
    /// complex-cell weights: simple location in the field, feature
    pub w_c: Vec<f32>,
    pub simple: Cells,
    pub complex: Cells,
    /// last event step of each input
    in_last: Vec<i32>,
    /// simple and complex cells that spiked this step
    pub fired_s: Vec<u16>,
    pub fired_c: Vec<u16>,
    /// excitatory and inhibitory drive given to simple cells this step
    pub excitation: f32,
    pub inhibition: f32,
    /// synapses reached by a spike since the last reset
    synops: u64,
}

impl Net {
    pub fn new(side: usize, p: Params, rng: &mut Rng) -> Self {
        let n = (side - p.rf) / p.stride + 1;
        let m = n + 1 - p.complex_rf;
        let (k, kc) = (p.features, p.complex_features);
        let mut net = Self {
            w_s: weights::random(k * 2 * p.rf * p.rf, 0.0, 1.0, rng),
            lat: weights::random(n * n * k * lateral_len(&p), 0.0, 1.0, rng),
            td: weights::random(n * n * k * p.complex_rf.pow(2) * kc, 0.0, 1.0, rng),
            w_c: weights::random(m * m * kc * p.complex_rf.pow(2) * k, 0.0, 1.0, rng),
            simple: Cells::new(n * n * k, p.tau_m),
            complex: Cells::new(m * m * kc, p.complex_tau_m),
            in_last: vec![NEVER; 2 * side * side],
            fired_s: Vec::new(),
            fired_c: Vec::new(),
            excitation: 0.0,
            inhibition: 0.0,
            synops: 0,
            side,
            n,
            m,
            p,
        };
        // inhibition from outside the grid does not exist
        for s in 0..net.simple.len() {
            net.mask_inhibition(s);
        }
        let p = &net.p;
        let fsize = 2 * p.rf * p.rf;
        net.w_s
            .chunks_exact_mut(fsize)
            .for_each(|w| set_norm(w, p.norm_exc));
        let cin = p.complex_rf.pow(2) * p.features;
        net.w_c
            .chunks_exact_mut(cin)
            .for_each(|w| set_norm(w, p.norm_complex));
        net.lat
            .chunks_exact_mut(lateral_len(p))
            .for_each(|w| set_norm(w, p.norm_lateral));
        let tdl = p.complex_rf.pow(2) * p.complex_features;
        net.td
            .chunks_exact_mut(tdl)
            .for_each(|w| set_norm(w, p.norm_top_down));
        net
    }

    fn mask_inhibition(&mut self, s: usize) {
        let (k, kc, r) = (
            self.p.features,
            self.p.complex_features,
            self.p.lateral_range as i32,
        );
        let (sy, sx) = self.loc(s);
        let ll = lateral_len(&self.p);
        for (d, (dy, dx)) in offsets(r).enumerate() {
            if !self.valid(sy + dy, sx + dx, self.n) {
                self.lat[s * ll + d * k..][..k].fill(0.0);
            }
        }
        let crf = self.p.complex_rf as i32;
        let tdl = self.p.complex_rf.pow(2) * kc;
        for q in 0..crf * crf {
            let (cy, cx) = (sy - q / crf, sx - q % crf);
            if !self.valid(cy, cx, self.m) {
                self.td[s * tdl + q as usize * kc..][..kc].fill(0.0);
            }
        }
    }

    fn valid(&self, y: i32, x: i32, n: usize) -> bool {
        y >= 0 && x >= 0 && (y as usize) < n && (x as usize) < n
    }

    /// Location (row, column) of simple cell `s`.
    fn loc(&self, s: usize) -> (i32, i32) {
        let l = s / self.p.features;
        ((l / self.n) as i32, (l % self.n) as i32)
    }

    pub fn reset(&mut self) {
        self.simple.reset();
        self.complex.reset();
        self.in_last.fill(NEVER);
        self.fired_s.clear();
        self.fired_c.clear();
        self.synops = 0;
    }

    /// The input events and the cells' spikes since the reset, and the synapses
    /// they reached.
    pub fn activity(&self, events: u64) -> Activity {
        let cells = self.simple.counts.iter().chain(&self.complex.counts);
        let spikes: u64 = cells.map(|&c| c as u64).sum();
        Activity::new(events + spikes, self.synops)
    }

    /// Delivers the events of step `t` and everything they set off.
    pub fn step(&mut self, t: u32, events: &[u16], learn: bool) {
        self.fired_s.clear();
        self.fired_c.clear();
        self.excitation = 0.0;
        self.inhibition = 0.0;
        let (side, rf, stride, k) = (self.side, self.p.rf, self.p.stride, self.p.features);
        for &id in events {
            self.in_last[id as usize] = t as i32;
            let (pol, pix) = (id as usize / (side * side), id as usize % (side * side));
            let (y, x) = (pix / side, pix % side);
            for sy in covering(y, rf, stride, self.n) {
                for sx in covering(x, rf, stride, self.n) {
                    let off = pol * rf * rf + (y - sy * stride) * rf + (x - sx * stride);
                    let loc = sy * self.n + sx;
                    for f in 0..k {
                        let w = self.w_s[f * 2 * rf * rf + off];
                        self.excite(loc * k + f, t, w);
                    }
                }
            }
        }
        let mut i = 0;
        while i < self.fired_s.len() {
            let s = self.fired_s[i] as usize;
            if learn {
                self.learn_simple(s, t);
            }
            self.spread_simple(s, t);
            i += 1;
        }
        let mut i = 0;
        while i < self.fired_c.len() {
            let c = self.fired_c[i] as usize;
            if learn {
                self.learn_complex(c, t);
            }
            self.spread_complex(c, t);
            i += 1;
        }
    }

    fn excite(&mut self, s: usize, t: u32, w: f32) {
        self.synops += 1;
        let p = &self.p;
        let mut v = self.simple.at(s, t) + w;
        let last = self.simple.last[s];
        if last != NEVER {
            v -= p.eta_rp * (-((t as i32 - last) as f32) / p.tau_rp).exp();
        }
        self.excitation += w;
        if v >= p.v_thr && last != t as i32 {
            self.simple.fire(s, t, p.v_reset);
            self.fired_s.push(s as u16);
        } else {
            self.simple.v[s] = v.max(p.v_min);
        }
    }

    fn inhibit_simple(&mut self, s: usize, t: u32, w: f32) {
        self.synops += 1;
        let v = self.simple.at(s, t) - w;
        self.simple.v[s] = v.max(self.p.v_min);
        self.inhibition += w;
    }

    // a simple spike: static and lateral inhibition, drive to complex cells
    fn spread_simple(&mut self, s: usize, t: u32) {
        let (k, kc, crf) = (
            self.p.features,
            self.p.complex_features,
            self.p.complex_rf as i32,
        );
        let (sy, sx) = self.loc(s);
        let (f, loc) = (s % k, s / k);
        for g in (0..k).filter(|&g| g != f) {
            self.inhibit_simple(loc * k + g, t, self.p.w_static);
        }
        let (r, ll) = (self.p.lateral_range as i32, lateral_len(&self.p));
        for (dy, dx) in offsets(r) {
            let (ny, nx) = (sy + dy, sx + dx);
            if !self.valid(ny, nx, self.n) {
                continue;
            }
            // the inhibited cell sees this one at the opposite offset
            let d = offset_index(-dy, -dx, r);
            for g in 0..k {
                let target = (ny as usize * self.n + nx as usize) * k + g;
                let w = self.lat[target * ll + d * k + f];
                self.inhibit_simple(target, t, w);
            }
        }
        let cin = (crf * crf) as usize * k;
        for q in 0..crf * crf {
            let (cy, cx) = (sy - q / crf, sx - q % crf);
            if !self.valid(cy, cx, self.m) {
                continue;
            }
            let i = q as usize * k + f;
            for c in 0..kc {
                let cell = (cy as usize * self.m + cx as usize) * kc + c;
                self.synops += 1;
                let v = self.complex.at(cell, t) + self.w_c[cell * cin + i];
                if v >= self.p.complex_thr && self.complex.last[cell] != t as i32 {
                    self.complex.fire(cell, t, self.p.v_reset);
                    self.fired_c.push(cell as u16);
                } else {
                    self.complex.v[cell] = v.max(self.p.v_min);
                }
            }
        }
    }

    // a complex spike: static inhibition of its neighbours, top-down inhibition below
    fn spread_complex(&mut self, c: usize, t: u32) {
        let (k, kc, crf) = (self.p.features, self.p.complex_features, self.p.complex_rf);
        let (cl, g) = (c / kc, c % kc);
        for h in (0..kc).filter(|&h| h != g) {
            let other = cl * kc + h;
            self.synops += 1;
            let v = self.complex.at(other, t) - self.p.complex_w_static;
            self.complex.v[other] = v.max(self.p.v_min);
        }
        let (cy, cx) = (cl / self.m, cl % self.m);
        let tdl = crf * crf * kc;
        for q in 0..crf * crf {
            let (sy, sx) = (cy + q / crf, cx + q % crf);
            for f in 0..k {
                let s = (sy * self.n + sx) * k + f;
                let w = self.td[s * tdl + q * kc + g];
                self.inhibit_simple(s, t, w);
            }
        }
    }

    fn learn_simple(&mut self, s: usize, t: u32) {
        let p = &self.p;
        let (k, rf, stride, side) = (p.features, p.rf, p.stride, self.side);
        let (sy, sx) = self.loc(s);
        let (t, prev) = (t as i32, self.simple.prev[s]);
        // excitatory STDP on the shared feature
        let f = s % k;
        let row = &mut self.w_s[f * 2 * rf * rf..][..2 * rf * rf];
        for (j, w) in row.iter_mut().enumerate() {
            let (pol, ry, rx) = (j / (rf * rf), j / rf % rf, j % rf);
            let y = sy as usize * stride + ry;
            let x = sx as usize * stride + rx;
            let ti = self.in_last[pol * side * side + y * side + x];
            *w = (*w + stdp(ti, t, prev, p.ltp, p.ltd, p)).max(0.0);
        }
        set_norm(row, p.norm_exc);
        // inhibitory STDP: inhibition that came just before this spike grows
        let (r, ll) = (p.lateral_range as i32, lateral_len(p));
        for (d, (dy, dx)) in offsets(r).enumerate() {
            let (ny, nx) = (sy + dy, sx + dx);
            if !self.valid(ny, nx, self.n) {
                continue;
            }
            for g in 0..k {
                let pre = (ny as usize * self.n + nx as usize) * k + g;
                let tp = self.simple.last[pre];
                let w = &mut self.lat[s * ll + d * k + g];
                *w = (*w + stdp(tp, t, prev, p.i_ltp, p.i_ltd, p)).max(0.0);
            }
        }
        set_norm(&mut self.lat[s * ll..][..ll], p.norm_lateral);
        let (crf, kc) = (p.complex_rf as i32, p.complex_features);
        let tdl = (crf * crf) as usize * kc;
        for q in 0..crf * crf {
            let (cy, cx) = (sy - q / crf, sx - q % crf);
            if !self.valid(cy, cx, self.m) {
                continue;
            }
            for c in 0..kc {
                let pre = (cy as usize * self.m + cx as usize) * kc + c;
                let tp = self.complex.last[pre];
                let w = &mut self.td[s * tdl + q as usize * kc + c];
                *w = (*w + stdp(tp, t, prev, p.i_ltp, p.i_ltd, p)).max(0.0);
            }
        }
        set_norm(&mut self.td[s * tdl..][..tdl], p.norm_top_down);
    }

    // rectangular windows, so the cell grows tolerant to the inputs' exact timing
    fn learn_complex(&mut self, c: usize, t: u32) {
        let p = &self.p;
        let (k, kc, crf) = (p.features, p.complex_features, p.complex_rf);
        let (cl, cin) = (c / kc, crf * crf * k);
        let (cy, cx) = (cl / self.m, cl % self.m);
        let (t, prev) = (t as i32, self.complex.prev[c]);
        let row = &mut self.w_c[c * cin..][..cin];
        for (i, w) in row.iter_mut().enumerate() {
            let (q, f) = (i / k, i % k);
            let s = ((cy + q / crf) * self.n + cx + q % crf) * k + f;
            let ti = self.simple.last[s];
            let mut dw = 0.0;
            if ti != NEVER && ((ti - t).abs() as f32) <= p.c_win {
                dw += p.c_ltp;
            }
            if ti != NEVER && prev != NEVER && ((prev - ti).abs() as f32) <= p.c_win {
                dw -= p.c_ltd;
            }
            *w = (*w + dw).max(0.0);
        }
        set_norm(row, p.norm_complex);
    }

    /// Spike count of every simple then complex cell, square-rooted.
    pub fn code(&self, out: &mut Vec<f32>) {
        out.clear();
        let all = self.simple.counts.iter().chain(&self.complex.counts);
        out.extend(all.map(|&c| (c as f32).sqrt()));
    }

    /// Weights of simple feature `f`: ON image then OFF image, rf × rf each.
    pub fn feature(&self, f: usize) -> &[f32] {
        let n = 2 * self.p.rf * self.p.rf;
        &self.w_s[f * n..(f + 1) * n]
    }
}

/// Exponential STDP from a presynaptic spike at `tp` to a postsynaptic spike at
/// `t`: potentiation when it came before, depression when it came after the
/// previous postsynaptic spike `prev`.
fn stdp(tp: i32, t: i32, prev: i32, ltp: f32, ltd: f32, p: &Params) -> f32 {
    if tp == NEVER {
        return 0.0;
    }
    let mut dw = 0.0;
    if tp <= t {
        dw += ltp * ((tp - t) as f32 / p.tau_ltp).exp();
    }
    if prev != NEVER && tp >= prev && tp < t {
        dw -= ltd * ((prev - tp) as f32 / p.tau_ltd).exp();
    }
    dw
}

/// Inhibitory inputs of a simple cell: (2r+1)² - 1 locations times features.
fn lateral_len(p: &Params) -> usize {
    ((2 * p.lateral_range + 1).pow(2) - 1) * p.features
}

/// Neighbour offsets within `r`, centre left out.
fn offsets(r: i32) -> impl Iterator<Item = (i32, i32)> {
    (-r..=r)
        .flat_map(move |dy| (-r..=r).map(move |dx| (dy, dx)))
        .filter(|&o| o != (0, 0))
}

fn offset_index(dy: i32, dx: i32, r: i32) -> usize {
    let i = ((dy + r) * (2 * r + 1) + dx + r) as usize;
    let centre = (r * (2 * r + 1) + r) as usize;
    if i > centre {
        i - 1
    } else {
        i
    }
}

/// Locations along one axis whose field covers pixel `x`.
fn covering(x: usize, rf: usize, stride: usize, n: usize) -> impl Iterator<Item = usize> {
    (0..n).filter(move |&l| x >= l * stride && x < l * stride + rf)
}

fn set_norm(v: &mut [f32], norm: f32) {
    let len = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if len > 0.0 {
        let k = norm / len;
        v.iter_mut().for_each(|x| *x *= k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use snn_core::events::{Events, Saccades};

    fn bar_events() -> Events {
        let mut img = vec![0u8; 784];
        for y in 6..22 {
            for x in 12..16 {
                img[y * 28 + x] = 255;
            }
        }
        let mut e = Events::new();
        e.generate(&img, 28, 28, &Saccades::default());
        e
    }

    #[test]
    fn layout_follows_the_retina() {
        let net = Net::new(34, Params::default(), &mut Rng::new(1));
        assert_eq!((net.n, net.m), (5, 4));
        assert_eq!(net.simple.len(), 5 * 5 * 32);
        assert_eq!(net.complex.len(), 4 * 4 * 16);
        assert_eq!(covering(13, 10, 6, 5).collect::<Vec<_>>(), [1, 2]);
        assert_eq!(covering(33, 10, 6, 5).collect::<Vec<_>>(), [4]);
    }

    #[test]
    fn one_event_reaches_every_feature_of_its_fields() {
        let mut net = Net::new(34, Params::default(), &mut Rng::new(1));
        net.reset();
        // pixel (13, 13) lies in 2x2 simple-cell fields of 32 features each
        net.step(1, &[13 * 34 + 13], false);
        assert!(net.fired_s.is_empty());
        assert_eq!(net.activity(1), Activity::new(1, 4 * 32));
        net.reset();
        assert_eq!(net.activity(0), Activity::default());
    }

    #[test]
    fn spiking_cells_add_inhibition_and_complex_input() {
        let p = Params {
            v_thr: 8.0,
            ..Params::default()
        };
        let mut net = Net::new(34, p, &mut Rng::new(2));
        let ev = bar_events();
        net.reset();
        let mut events = 0u64;
        for t in 1..=ev.steps() {
            net.step(t as u32, ev.at(t), false);
            events += ev.at(t).len() as u64;
        }
        let count = |c: &Cells| c.counts.iter().map(|&n| n as u64).sum::<u64>();
        let (simple, complex) = (count(&net.simple), count(&net.complex));
        assert!(simple > 0);
        let a = net.activity(events);
        assert_eq!(a.spikes, events + simple + complex);
        // each event excites at least one field of 32 features, each simple spike
        // inhibits the 31 other features at its location
        assert!(a.synops >= events * 32 + simple * 31);
    }

    #[test]
    fn neighbour_offsets_index_both_ways() {
        let all: Vec<usize> = offsets(1).map(|(y, x)| offset_index(y, x, 1)).collect();
        assert_eq!(all, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn corner_cells_get_no_inhibition_from_outside() {
        let net = Net::new(34, Params::default(), &mut Rng::new(1));
        let (k, ll) = (32, lateral_len(&net.p));
        // cell at location (0, 0): the three upper-left neighbours do not exist
        for d in [0, 1, 2, 3, 5] {
            assert!(net.lat[d * k..(d + 1) * k].iter().all(|&w| w == 0.0));
        }
        assert!(net.lat[4 * k..5 * k].iter().any(|&w| w > 0.0));
        let n: f32 = net.lat[..ll].iter().map(|w| w * w).sum::<f32>().sqrt();
        assert!((n - net.p.norm_lateral).abs() < 1e-2);
    }

    #[test]
    fn potentials_decay_between_inputs() {
        let mut c = Cells::new(1, 20.0);
        c.v[0] = 10.0;
        assert!((c.at(0, 20) - 10.0 * (-1.0f32).exp()).abs() < 1e-4);
        assert!((c.at(0, 20) - 10.0 * (-1.0f32).exp()).abs() < 1e-4);
    }

    #[test]
    fn events_make_simple_cells_spike() {
        let p = Params {
            v_thr: 8.0,
            ..Params::default()
        };
        let mut net = Net::new(34, p, &mut Rng::new(2));
        let e = bar_events();
        for t in 1..=e.steps() {
            net.step(t as u32, e.at(t), false);
        }
        assert!(net.simple.counts.iter().sum::<u32>() > 0);
    }

    #[test]
    fn learning_keeps_weights_non_negative_at_their_norms() {
        let p = Params {
            v_thr: 8.0,
            ..Params::default()
        };
        let mut net = Net::new(34, p, &mut Rng::new(3));
        let e = bar_events();
        for _ in 0..3 {
            net.reset();
            for t in 1..=e.steps() {
                net.step(t as u32, e.at(t), true);
            }
        }
        assert!(net
            .w_s
            .iter()
            .chain(&net.lat)
            .chain(&net.td)
            .all(|&w| w >= 0.0));
        let n: f32 = net.feature(0).iter().map(|w| w * w).sum::<f32>().sqrt();
        assert!((n - net.p.norm_exc).abs() < 1e-3);
    }
}
