//! Trainer-side encoder: turns training events into packets on the hub.

use crate::layer::{Layer, Step};
use crate::train::{ImageResult, Labels, Monitor};
use snn_core::encode::SpikeTrain;
use snn_core::{EpochResult, ImageInfo};
use snn_live::frame::Buf;
use snn_live::{Hub, Stream};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const STEPS: u8 = 3;
pub const RESULT: u8 = 4;
pub const WEIGHTS: u8 = 5;
pub const THRESH: u8 = 9;

const WEIGHTS_GAP: Duration = Duration::from_millis(100);

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
    /// latest encoded WEIGHTS record
    weights_rec: Vec<u8>,
    potentials: Vec<f32>,
}

impl Tap {
    pub fn new(hub: Arc<Hub>, w_min: f32, w_max: f32) -> Self {
        Self {
            s: Stream::new(hub),
            w_range: (w_min, w_max),
            last_weights: Instant::now() - Duration::from_secs(1),
            weights_rec: Vec::new(),
            potentials: Vec::new(),
        }
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
        self.s.end_image(&r.info, r.running_accuracy);
        let viewers = self.s.viewers();
        let resume_due = self.s.resume_due();
        let send_weights = viewers && self.last_weights.elapsed() >= WEIGHTS_GAP;
        if send_weights || resume_due {
            self.encode_weights(layer, labels);
        }
        if resume_due {
            self.s.refresh_resume(&self.weights_rec);
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
        }
        if send_weights {
            self.s.close_steps();
            self.s.buf.raw(&self.weights_rec);
            self.last_weights = Instant::now();
        }
        if streaming || send_weights || self.s.flush_due() {
            self.s.flush();
        }
    }

    fn epoch_end(&mut self, e: &EpochResult) {
        self.s.epoch_end(e, &self.weights_rec);
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
