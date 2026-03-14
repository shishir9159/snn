//! Trainer-side encoder: turns training events into packets on the hub.

use crate::dopamine::Dopamine;
use crate::layer::{Layer, Step};
use crate::train::{ImageResult, Labels, Monitor};
use crate::voltage::Voltage;
use snn_core::encode::SpikeTrain;
use snn_core::{EpochResult, ImageInfo};
use snn_live::frame::Buf;
use snn_live::{Hub, Stream};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const STEPS: u8 = 3;
pub const RESULT: u8 = 4;
pub const WEIGHTS: u8 = 5;
pub const THRESH: u8 = 9;
/// The delayed rule's recent trials, rewards and dopamine.
pub const DOPAMINE: u8 = 14;
/// The delayed rule's tags on each row that holds some.
pub const TAGS: u8 = 15;
/// The voltage rule's constants and what it changed in the last training image.
pub const VOLTAGE: u8 = 51;

const WEIGHTS_GAP: Duration = Duration::from_millis(100);
/// Trials the dopamine timeline is told about; at 0.2 s each they still
/// cover rewards 3 s late.
const TRIALS: usize = 48;

/// One training presentation under the delayed rule, as the timeline draws it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trial {
    pub image: u64,
    /// training time before its first step
    pub start: u64,
    pub label: u8,
    pub winner: Option<usize>,
    pub correct: bool,
    /// step it answered at, 0 for no answer, and the reward that earned
    pub decided: usize,
    pub amount: f32,
    /// training time the reward arrives at
    pub at: u64,
    /// size of its tags after its last step
    pub tagged: f32,
    /// share of all tags the answering row held as the reward arrived; NaN
    /// while it is on its way
    pub credit: f32,
}

/// The clock, the constants and the dopamine now, then the recent trials.
pub fn dopamine(b: &mut Buf, d: &Dopamine, steps: usize, trials: &VecDeque<Trial>) {
    let p = d.params();
    b.begin(DOPAMINE);
    b.f64(d.clock as f64);
    b.u16(steps as u16);
    b.u32(p.gap);
    b.u32(p.delay);
    b.u32(p.jitter);
    b.f32(p.tau_c);
    b.f32(p.tau_d);
    b.f32(p.gain);
    b.f32(d.level);
    b.u16(d.live().len() as u16);
    b.u8(trials.len() as u8);
    for t in trials {
        b.f64(t.start as f64);
        b.u8(t.label);
        b.i16(t.winner.map_or(-1, |w| w as i16));
        b.u8(t.correct as u8);
        b.u16(t.decided as u16);
        b.f32(t.amount);
        b.f64(t.at as f64);
        b.f32(t.tagged);
        b.f32(t.credit);
    }
    b.end();
}

/// Per row holding tags: its LTP less its LTD tags as bytes over +-scale, the
/// size of its tags and the steps since its last.
pub fn tags(b: &mut Buf, d: &Dopamine, n_in: usize) {
    b.begin(TAGS);
    b.u16(d.live().len() as u16);
    b.u16(n_in as u16);
    for &j in d.live() {
        let (plus, minus) = d.tags(j);
        let net = plus.iter().zip(minus).map(|(p, m)| p - m);
        let scale = net.clone().fold(f32::MIN_POSITIVE, |s, v| s.max(v.abs()));
        let (size, age) = d.trace(j);
        b.u16(j as u16);
        b.f32(scale);
        b.f32(size);
        b.f32(age as f32);
        for v in net {
            b.u8((v / scale * 127.0).round() as i8 as u8);
        }
    }
    b.end();
}

/// The voltage rule's constants and resting potential; per neuron its LTD
/// amplitude, mean depolarisation, the LTP and LTD it took and the largest
/// change of a weight; then the inputs that spiked, and each one's change onto
/// every neuron as bytes over +-that neuron's largest.
pub fn voltage(b: &mut Buf, v: &Voltage) {
    let (p, c) = (v.params(), &v.last);
    let n = c.ltp.len();
    let mut top = vec![f32::MIN_POSITIVE; n];
    for row in c.dw.chunks_exact(n.max(1)) {
        top.iter_mut()
            .zip(row)
            .for_each(|(t, d)| *t = t.max(d.abs()));
    }
    b.begin(VOLTAGE);
    let consts = [
        p.tau_minus,
        p.tau_plus,
        p.tau_x,
        p.theta_minus,
        p.theta_plus,
    ];
    let amps = [p.v_spike, p.a_ltp, p.a_ltd, p.u_ref, v.rest];
    consts.iter().chain(&amps).for_each(|&x| b.f32(x));
    b.u16(n as u16);
    b.u16(c.inputs.len() as u16);
    for (j, &t) in top.iter().enumerate() {
        for x in [v.a_ltd[j], v.bar[j], c.ltp[j], c.ltd[j], t] {
            b.f32(x);
        }
    }
    c.inputs.iter().for_each(|&i| b.u16(i));
    for row in c.dw.chunks_exact(n.max(1)) {
        for (d, t) in row.iter().zip(&top) {
            b.u8((d / t * 127.0).round() as i8 as u8);
        }
    }
    b.end();
}

/// One step inside an open STEPS record.
pub fn step(b: &mut Buf, active: &[u16], fired: &[u16], potentials: &[f32], winner: Option<usize>) {
    b.u16(active.len() as u16);
    active.iter().for_each(|&i| b.u16(i));
    b.u16(fired.len() as u16);
    fired.iter().for_each(|&i| b.u16(i));
    potentials.iter().for_each(|&v| b.f32(v));
    b.i16(winner.map_or(-1, |w| w as i16));
}

pub fn thresholds(b: &mut Buf, th: &[f32]) {
    b.begin(THRESH);
    th.iter().for_each(|&v| b.f32(v));
    b.end();
}

pub struct Result<'a> {
    pub seq: u32,
    pub winner: Option<usize>,
    pub predicted: Option<u8>,
    pub correct: bool,
    pub learn: bool,
    pub running_accuracy: f32,
    pub spikes: &'a [u32],
    pub theta: &'a [f32],
    pub labels: &'a [Option<u8>],
}

pub fn result(b: &mut Buf, r: &Result) {
    b.begin(RESULT);
    b.u32(r.seq);
    b.i16(r.winner.map_or(-1, |w| w as i16));
    b.u8(r.predicted.unwrap_or(255));
    b.u8(r.correct as u8);
    b.u8(r.learn as u8);
    b.f32(r.running_accuracy);
    r.spikes
        .iter()
        .for_each(|&s| b.u16(s.min(u16::MAX as u32) as u16));
    r.theta.iter().for_each(|&t| b.f32(t));
    r.labels.iter().for_each(|l| b.u8(l.unwrap_or(255)));
    b.end();
}

pub struct Weights<'a> {
    pub rows: usize,
    pub cols: usize,
    pub w: &'a [f32],
    pub lo: f32,
    pub hi: f32,
    pub labels: &'a [Option<u8>],
    pub theta: &'a [f32],
}

/// Weights quantised to bytes over [lo, hi], then each row's label and theta.
pub fn weights(b: &mut Buf, x: &Weights) {
    b.begin(WEIGHTS);
    b.u16(x.rows as u16);
    b.u16(x.cols as u16);
    let k = 255.0 / (x.hi - x.lo).max(f32::EPSILON);
    for &v in x.w {
        b.u8(((v - x.lo) * k).clamp(0.0, 255.0) as u8);
    }
    x.labels.iter().for_each(|l| b.u8(l.unwrap_or(255)));
    x.theta.iter().for_each(|&t| b.f32(t));
    b.end();
}

pub struct Tap {
    s: Stream,
    w_range: (f32, f32),
    last_weights: Instant,
    /// latest encoded WEIGHTS record, the delayed rule's TAGS and DOPAMINE,
    /// and the voltage rule's VOLTAGE
    weights_rec: Vec<u8>,
    tags_rec: Vec<u8>,
    dopamine_rec: Vec<u8>,
    voltage_rec: Vec<u8>,
    /// presentations the voltage rule had learned from by the last image
    learned: u64,
    potentials: Vec<f32>,
    /// the delayed rule's latest trials, and the presentations counted so far
    trials: VecDeque<Trial>,
    seen: u64,
    steps: usize,
}

impl Tap {
    pub fn new(hub: Arc<Hub>, w_min: f32, w_max: f32) -> Self {
        Self {
            s: Stream::new(hub),
            w_range: (w_min, w_max),
            last_weights: Instant::now() - Duration::from_secs(1),
            weights_rec: Vec::new(),
            tags_rec: Vec::new(),
            dopamine_rec: Vec::new(),
            voltage_rec: Vec::new(),
            learned: 0,
            potentials: Vec::new(),
            trials: VecDeque::new(),
            seen: 0,
            steps: 0,
        }
    }

    // a presentation of the delayed rule: its trial, and the credit of the
    // rewards that arrived during it
    fn record_trial(&mut self, r: &ImageResult, d: &Dopamine) {
        self.seen = d.images;
        let last = &d.last;
        self.s.sim_steps += d.params().gap as u64;
        self.steps = last.steps;
        if self.trials.len() == TRIALS {
            self.trials.pop_front();
        }
        self.trials.push_back(Trial {
            image: last.image,
            start: last.start,
            label: r.info.label,
            winner: r.outcome.winner,
            correct: r.correct,
            decided: last.sent.map_or(0, |p| (p.sent - last.start) as usize),
            amount: last.sent.map_or(0.0, |p| p.amount),
            at: last.sent.map_or(0, |p| p.at),
            tagged: last.tagged,
            credit: f32::NAN,
        });
        // a reward can land in its own trial's gap
        for (puff, credit) in &last.arrived {
            if let Some(t) = self.trials.iter_mut().find(|t| t.image == puff.image) {
                t.credit = *credit;
            }
        }
    }

    // what a joining viewer starts from
    fn snapshot(&self) -> Vec<u8> {
        let (w, t, d, v) = (
            &self.weights_rec,
            &self.tags_rec,
            &self.dopamine_rec,
            &self.voltage_rec,
        );
        [&w[..], t, d, v].concat()
    }

    fn encode_weights(&mut self, layer: &Layer, labels: &Labels) {
        let assigned: Vec<Option<u8>> = (0..layer.n_out).map(|j| labels.of(j)).collect();
        let mut b = Buf::new();
        let w = Weights {
            rows: layer.n_out,
            cols: layer.n_in,
            w: layer.weights(),
            lo: self.w_range.0,
            hi: self.w_range.1,
            labels: &assigned,
            theta: &layer.theta,
        };
        weights(&mut b, &w);
        self.weights_rec = b.take();
        if !self.trials.is_empty() {
            tags(&mut b, &layer.dopamine, layer.n_in);
            self.tags_rec = b.take();
            dopamine(&mut b, &layer.dopamine, self.steps, &self.trials);
            self.dopamine_rec = b.take();
        }
    }
}

impl Monitor for Tap {
    fn image_start(&mut self, info: &ImageInfo, pixels: &[u8], _train: &SpikeTrain) {
        self.s.begin_image(info, pixels);
    }

    fn step(&mut self, st: &Step) {
        self.s.sim_steps += 1;
        if !self.s.streaming() {
            return;
        }
        if st.t == 1 {
            self.s.close_steps();
            thresholds(&mut self.s.buf, st.thresholds);
        }
        self.potentials.clear();
        self.potentials.extend(st.neurons.iter().map(|n| n.p));
        let b = self.s.step(STEPS, st.t);
        step(b, st.active, st.fired, &self.potentials, st.winner);
        if self.s.flush_due() {
            self.s.flush();
        }
    }

    fn image_end(&mut self, r: &ImageResult, layer: &Layer, labels: &Labels) {
        if layer.dopamine.images != self.seen {
            self.record_trial(r, &layer.dopamine);
        }
        // the voltage rule learned from this image
        let learned = layer.voltage.images != self.learned;
        self.learned = layer.voltage.images;
        self.s.end_image(&r.info, r.running_accuracy, r.activity);
        let viewers = self.s.viewers();
        let resume_due = self.s.resume_due();
        let send_weights = viewers && self.last_weights.elapsed() >= WEIGHTS_GAP;
        if send_weights || resume_due {
            self.encode_weights(layer, labels);
        }
        if learned && (self.s.streaming() || resume_due) {
            let mut b = Buf::new();
            voltage(&mut b, &layer.voltage);
            self.voltage_rec = b.take();
        }
        if resume_due {
            self.s.refresh_resume(&self.snapshot());
        }
        if !viewers {
            return;
        }
        let streaming = self.s.streaming();
        if streaming {
            let assigned: Vec<Option<u8>> = (0..layer.n_out).map(|j| labels.of(j)).collect();
            self.s.close_steps();
            let res = Result {
                seq: self.s.seq(),
                winner: r.outcome.winner,
                predicted: r.predicted,
                correct: r.correct,
                learn: r.info.learn,
                running_accuracy: r.running_accuracy,
                spikes: &r.outcome.spikes,
                theta: &layer.theta,
                labels: &assigned,
            };
            result(&mut self.s.buf, &res);
            if r.info.learn && !self.trials.is_empty() {
                dopamine(&mut self.s.buf, &layer.dopamine, self.steps, &self.trials);
            }
            if learned {
                self.s.buf.raw(&self.voltage_rec);
            }
        }
        if send_weights {
            self.s.close_steps();
            self.s.buf.raw(&self.weights_rec);
            self.s.buf.raw(&self.tags_rec);
            self.last_weights = Instant::now();
        }
        if streaming || send_weights || self.s.flush_due() {
            self.s.flush();
        }
    }

    fn epoch_end(&mut self, e: &EpochResult) {
        self.s.epoch_end(e, &self.snapshot());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::Trainer;
    use crate::Params;
    use snn_core::dataset::synthetic;
    use snn_core::Control;
    use snn_live::frame::{self, records};

    fn kinds(packet: &[u8]) -> Vec<u8> {
        records(packet).iter().map(|r| r.0).collect()
    }

    #[test]
    fn silent_without_viewers() {
        let hub = Arc::new(Hub::new());
        let mut tap = Tap::new(hub.clone(), 0.0, 2.0);
        let data = synthetic(5, 1);
        let mut t = Trainer::new(Params::default(), 784, 4, 10, 1);
        t.fit(&data, None, 1, &mut tap, &Control::new(0));
        let mut v = hub.join();
        assert!(v.poll(Duration::ZERO).is_empty());
        assert_eq!(tap.s.sim_steps, 5 * 200);
        assert_eq!(tap.s.history[0].points.len(), 5);
    }

    #[test]
    fn streams_a_full_presentation() {
        let hub = Arc::new(Hub::new());
        let mut v = hub.join();
        let mut tap = Tap::new(hub.clone(), 0.0, 2.0);
        let data = synthetic(3, 1);
        let mut t = Trainer::new(Params::default(), 784, 4, 10, 1);
        t.fit(&data, None, 1, &mut tap, &Control::new(0));

        let all: Vec<u8> = v
            .poll(Duration::ZERO)
            .iter()
            .flat_map(|p| kinds(p))
            .collect();
        let count = |kind| all.iter().filter(|&&k| k == kind).count();
        assert!(count(frame::IMAGE) >= 1);
        assert_eq!(count(THRESH), count(frame::IMAGE));
        assert_eq!(count(RESULT), count(frame::IMAGE));
        assert!(all.contains(&WEIGHTS));
        assert!(all.contains(&frame::PROGRESS));
        assert_eq!(all.last(), Some(&frame::PROGRESS));
        assert!(all.contains(&frame::EPOCH));
    }

    #[test]
    fn late_viewer_gets_a_snapshot() {
        let hub = Arc::new(Hub::new());
        let mut tap = Tap::new(hub.clone(), 0.0, 2.0);
        let data = synthetic(5, 1);
        let mut t = Trainer::new(Params::default(), 784, 4, 10, 1);
        t.fit(&data, None, 1, &mut tap, &Control::new(0));
        let v = hub.join();
        let intro: Vec<u8> = v.intro().iter().flat_map(|p| kinds(p)).collect();
        let want = [WEIGHTS, frame::HISTORY, frame::EPOCH, frame::PROGRESS];
        assert_eq!(intro, want);
    }

    #[test]
    fn the_delayed_rule_streams_its_trials_and_tags() {
        let hub = Arc::new(Hub::new());
        let mut v = hub.join();
        let mut tap = Tap::new(hub.clone(), 0.0, 2.0);
        let data = synthetic(4, 1);
        let mut p = Params {
            rule: crate::params::Rule::Delayed,
            ..Params::default()
        };
        // rewards land within the next trial
        p.dopamine.delay = 300;
        p.dopamine.jitter = 0;
        p.dopamine.gap = 100;
        let mut t = Trainer::new(p, 784, 4, 10, 1);
        t.fit(&data, None, 1, &mut tap, &Control::new(0));
        assert_eq!(tap.s.sim_steps, 4 * 300, "steps and gaps");
        assert_eq!(tap.trials.len(), 4);
        let first = tap.trials[0];
        assert_eq!(first.start, 0);
        if first.winner.is_some() {
            assert!(!first.credit.is_nan() && first.at == first.decided as u64 + 300);
        }
        let last = tap.trials[3];
        assert!(last.credit.is_nan(), "still on its way");
        let packets = v.poll(Duration::ZERO);
        let recs: Vec<(u8, Vec<u8>)> = packets
            .iter()
            .flat_map(|p| records(p).into_iter().map(|(k, b)| (k, b.to_vec())))
            .collect();
        let count = |kind| recs.iter().filter(|r| r.0 == kind).count();
        assert!(count(DOPAMINE) >= 1 && count(DOPAMINE) <= count(RESULT));
        let d = &recs.iter().rev().find(|r| r.0 == DOPAMINE).unwrap().1;
        let n = d[40] as usize;
        assert_eq!(d.len(), 41 + 34 * n);
        assert_eq!(
            f64::from_le_bytes(d[..8].try_into().unwrap()),
            (n * 300) as f64
        );
        let tags_rec = &recs.iter().find(|r| r.0 == TAGS).unwrap().1;
        let rows = u16::from_le_bytes([tags_rec[0], tags_rec[1]]) as usize;
        assert_eq!(tags_rec.len(), 4 + rows * (14 + 784));
    }

    #[test]
    fn the_voltage_rule_streams_what_it_changed() {
        let hub = Arc::new(Hub::new());
        let mut v = hub.join();
        let mut tap = Tap::new(hub.clone(), 0.0, 2.0);
        let data = synthetic(4, 1);
        let p = Params {
            rule: crate::params::Rule::Voltage,
            ..Params::default()
        };
        let mut t = Trainer::new(p, 784, 4, 10, 1);
        t.fit(&data, None, 1, &mut tap, &Control::new(0));
        let recs: Vec<(u8, Vec<u8>)> = v
            .poll(Duration::ZERO)
            .iter()
            .flat_map(|p| records(p).into_iter().map(|(k, b)| (k, b.to_vec())))
            .collect();
        let count = |kind| recs.iter().filter(|r| r.0 == kind).count();
        assert!(count(VOLTAGE) >= 1 && count(VOLTAGE) == count(RESULT));
        let d = &recs.iter().rev().find(|r| r.0 == VOLTAGE).unwrap().1;
        let (n, inputs) = (
            u16::from_le_bytes([d[40], d[41]]) as usize,
            u16::from_le_bytes([d[42], d[43]]) as usize,
        );
        assert_eq!(n, 4);
        assert!(inputs > 0 && inputs <= 784);
        assert_eq!(d.len(), 44 + 20 * n + inputs * (2 + n));
        let theta_minus = f32::from_le_bytes(d[12..16].try_into().unwrap());
        assert_eq!(theta_minus, t.params.voltage.theta_minus);
        // a late viewer starts from the last one too
        let intro: Vec<u8> = hub.join().intro().iter().flat_map(|p| kinds(p)).collect();
        assert!(intro.contains(&VOLTAGE));
    }

    #[test]
    fn steps_record_layout() {
        let mut b = Buf::new();
        let at = frame::steps_begin(&mut b, STEPS, 7);
        step(&mut b, &[3, 9], &[1], &[0.5, -1.0], Some(1));
        step(&mut b, &[], &[], &[0.0, 0.0], None);
        b.patch_u16(at, 2);
        b.end();
        let bytes = b.take();
        let r = records(&bytes);
        let p = r[0].1;
        // t0, count, then (2 + 2*2) + (2 + 2) + 8 + 2 and 2 + 2 + 8 + 2
        assert_eq!(p.len(), 4 + 20 + 14);
        assert_eq!(&p[..4], &[7, 0, 2, 0]);
        assert_eq!(&p[4..10], &[2, 0, 3, 0, 9, 0]);
        assert_eq!(i16::from_le_bytes([p[22], p[23]]), 1);
        assert_eq!(i16::from_le_bytes([p[36], p[37]]), -1);
    }

    #[test]
    fn weights_quantise_to_bytes() {
        let mut b = Buf::new();
        let x = Weights {
            rows: 1,
            cols: 3,
            w: &[0.0, 1.0, 5.0],
            lo: 0.0,
            hi: 2.0,
            labels: &[Some(4)],
            theta: &[0.0],
        };
        weights(&mut b, &x);
        let bytes = b.take();
        assert_eq!(
            records(&bytes)[0].1,
            &[1, 0, 3, 0, 0, 127, 255, 4, 0, 0, 0, 0]
        );
    }

    #[test]
    fn result_layout() {
        let mut b = Buf::new();
        let r = Result {
            seq: 1,
            winner: Some(2),
            predicted: None,
            correct: false,
            learn: true,
            running_accuracy: 0.25,
            spikes: &[0, 1, 2, 3],
            theta: &[0.0; 4],
            labels: &[Some(1), None, Some(3), None],
        };
        result(&mut b, &r);
        let bytes = b.take();
        let p = records(&bytes)[0].1;
        assert_eq!(p.len(), 4 + 2 + 3 + 4 + 8 + 16 + 4);
        assert_eq!(&p[p.len() - 4..], &[1, 255, 3, 255]);
    }
}
