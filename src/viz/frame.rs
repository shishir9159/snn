//! Binary records streamed to the dashboard. Little endian. Each record is
//! `kind: u8, len: u32, payload`; one WebSocket message carries several.

use crate::params::Params;
use crate::train::ImageInfo;

pub const META: u8 = 1;
pub const IMAGE: u8 = 2;
pub const STEPS: u8 = 3;
pub const RESULT: u8 = 4;
pub const WEIGHTS: u8 = 5;
pub const EPOCH: u8 = 6;
pub const STATE: u8 = 7;
pub const PROGRESS: u8 = 8;
pub const THRESH: u8 = 9;
pub const HISTORY: u8 = 10;

/// Record writer. `begin`/`end` bracket a record and patch in its length.
#[derive(Debug, Default)]
pub struct Buf {
    bytes: Vec<u8>,
    open: Option<usize>,
}

impl Buf {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn begin(&mut self, kind: u8) {
        assert!(self.open.is_none(), "record already open");
        self.bytes.push(kind);
        self.open = Some(self.bytes.len());
        self.bytes.extend([0; 4]);
    }

    pub fn end(&mut self) {
        let at = self.open.take().expect("no open record");
        let len = (self.bytes.len() - at - 4) as u32;
        self.bytes[at..at + 4].copy_from_slice(&len.to_le_bytes());
    }

    pub fn is_open(&self) -> bool {
        self.open.is_some()
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Hands out the finished records and starts over.
    pub fn take(&mut self) -> Vec<u8> {
        assert!(self.open.is_none(), "record still open");
        std::mem::take(&mut self.bytes)
    }

    pub fn u8(&mut self, v: u8) {
        self.bytes.push(v);
    }

    pub fn u16(&mut self, v: u16) {
        self.bytes.extend(v.to_le_bytes());
    }

    pub fn i16(&mut self, v: i16) {
        self.bytes.extend(v.to_le_bytes());
    }

    pub fn u32(&mut self, v: u32) {
        self.bytes.extend(v.to_le_bytes());
    }

    pub fn f32(&mut self, v: f32) {
        self.bytes.extend(v.to_le_bytes());
    }

    pub fn f64(&mut self, v: f64) {
        self.bytes.extend(v.to_le_bytes());
    }

    pub fn raw(&mut self, v: &[u8]) {
        self.bytes.extend_from_slice(v);
    }

    /// Patches a u16 written earlier at byte offset `at`.
    pub fn patch_u16(&mut self, at: usize, v: u16) {
        self.bytes[at..at + 2].copy_from_slice(&v.to_le_bytes());
    }

    pub fn str(&mut self, s: &str) {
        let s = &s.as_bytes()[..s.len().min(u16::MAX as usize)];
        self.u16(s.len() as u16);
        self.raw(s);
    }
}

/// Static description of the run, sent first to every viewer.
pub struct Meta<'a> {
    pub dataset: &'a str,
    pub width: usize,
    pub height: usize,
    pub n_out: usize,
    pub classes: usize,
    pub n_train: usize,
    pub n_test: usize,
    pub epochs: usize,
    pub params: &'a Params,
}

pub fn meta(b: &mut Buf, m: &Meta) {
    let p = m.params;
    b.begin(META);
    b.u16(m.width as u16);
    b.u16(m.height as u16);
    b.u16(m.n_out as u16);
    b.u16(p.time_steps as u16);
    b.u8(m.classes as u8);
    b.u32(m.n_train as u32);
    b.u32(m.n_test as u32);
    b.u16(m.epochs as u16);
    for v in [
        p.stdp.w_min,
        p.stdp.w_max,
        p.lif.rest,
        p.lif.floor,
        p.lif.leak,
    ] {
        b.f32(v);
    }
    b.u16(p.lif.t_ref as u16);
    b.f32(p.threshold_div);
    b.f32(p.stdp.sigma);
    b.str(m.dataset);
    b.end();
}

/// Start of a presentation whose steps follow.
pub fn image(b: &mut Buf, seq: u32, info: &ImageInfo, pixels: &[u8]) {
    b.begin(IMAGE);
    b.u32(seq);
    b.u16(info.epoch as u16);
    b.u32(info.index as u32);
    b.u32(info.total as u32);
    b.u8(info.label);
    b.u8(info.learn as u8);
    b.raw(pixels);
    b.end();
}

pub fn thresholds(b: &mut Buf, th: &[f32]) {
    b.begin(THRESH);
    th.iter().for_each(|&v| b.f32(v));
    b.end();
}

/// Opens a STEPS record; returns the offset of its step count for [`Buf::patch_u16`].
pub fn steps_begin(b: &mut Buf, t0: usize) -> usize {
    b.begin(STEPS);
    b.u16(t0 as u16);
    let at = b.len();
    b.u16(0);
    at
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

/// Weights quantised to bytes over [lo, hi].
pub fn weights(b: &mut Buf, rows: usize, cols: usize, w: &[f32], lo: f32, hi: f32) {
    b.begin(WEIGHTS);
    b.u16(rows as u16);
    b.u16(cols as u16);
    let k = 255.0 / (hi - lo).max(f32::EPSILON);
    for &v in w {
        b.u8(((v - lo) * k).clamp(0.0, 255.0) as u8);
    }
    b.end();
}

pub fn epoch(
    b: &mut Buf,
    epoch: usize,
    train_acc: f32,
    test_acc: Option<f32>,
    images_per_sec: f32,
) {
    b.begin(EPOCH);
    b.u16(epoch as u16);
    b.f32(train_acc);
    b.f32(test_acc.unwrap_or(f32::NAN));
    b.f32(images_per_sec);
    b.end();
}

pub fn state(b: &mut Buf, paused: bool, speed: u32, done: bool) {
    b.begin(STATE);
    b.u8(paused as u8);
    b.u32(speed);
    b.u8(done as u8);
    b.end();
}

pub struct Progress {
    pub epoch: usize,
    pub index: usize,
    pub total: usize,
    pub learn: bool,
    pub running_accuracy: f32,
    pub sim_steps: u64,
    pub images: u64,
    pub images_per_sec: f32,
}

pub fn progress(b: &mut Buf, p: &Progress) {
    b.begin(PROGRESS);
    b.u16(p.epoch as u16);
    b.u32(p.index as u32);
    b.u32(p.total as u32);
    b.u8(p.learn as u8);
    b.f32(p.running_accuracy);
    b.f64(p.sim_steps as f64);
    b.f64(p.images as f64);
    b.f32(p.images_per_sec);
    b.end();
}

/// Running accuracy samples as (images seen, accuracy).
pub fn history(b: &mut Buf, points: &[(u32, f32)]) {
    b.begin(HISTORY);
    b.u32(points.len() as u32);
    for &(n, a) in points {
        b.u32(n);
        b.f32(a);
    }
    b.end();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Splits a packet into (kind, payload) records.
    fn records(bytes: &[u8]) -> Vec<(u8, &[u8])> {
        let mut out = Vec::new();
        let mut at = 0;
        while at < bytes.len() {
            let len = u32::from_le_bytes(bytes[at + 1..at + 5].try_into().unwrap()) as usize;
            out.push((bytes[at], &bytes[at + 5..at + 5 + len]));
            at += 5 + len;
        }
        assert_eq!(at, bytes.len());
        out
    }

    #[test]
    fn records_carry_their_length() {
        let mut b = Buf::new();
        state(&mut b, true, 1000, false);
        epoch(&mut b, 2, 0.5, None, 100.0);
        let bytes = b.take();
        let r = records(&bytes);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0], (STATE, &[1, 0xe8, 3, 0, 0, 0][..]));
        assert_eq!(r[1].0, EPOCH);
        assert!(f32::from_le_bytes(r[1].1[6..10].try_into().unwrap()).is_nan());
        assert!(b.is_empty());
    }

    #[test]
    fn steps_record_layout() {
        let mut b = Buf::new();
        let at = steps_begin(&mut b, 7);
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
        weights(&mut b, 1, 3, &[0.0, 1.0, 5.0], 0.0, 2.0);
        let bytes = b.take();
        assert_eq!(records(&bytes)[0].1, &[1, 0, 3, 0, 0, 127, 255]);
    }

    #[test]
    fn meta_and_result_sizes() {
        let p = Params::default();
        let m = Meta {
            dataset: "synthetic",
            width: 28,
            height: 28,
            n_out: 4,
            classes: 10,
            n_train: 10,
            n_test: 5,
            epochs: 1,
            params: &p,
        };
        let mut b = Buf::new();
        meta(&mut b, &m);
        result(
            &mut b,
            &Result {
                seq: 1,
                winner: Some(2),
                predicted: None,
                correct: false,
                learn: true,
                running_accuracy: 0.25,
                spikes: &[0, 1, 2, 3],
                theta: &[0.0; 4],
                labels: &[Some(1), None, Some(3), None],
            },
        );
        let bytes = b.take();
        let r = records(&bytes);
        assert_eq!(r[0].1.len(), 8 + 1 + 8 + 2 + 20 + 2 + 8 + 2 + 9);
        assert_eq!(r[1].1.len(), 4 + 2 + 3 + 4 + 8 + 16 + 4);
        assert_eq!(&r[1].1[r[1].1.len() - 4..], &[1, 255, 3, 255]);
    }

    #[test]
    #[should_panic(expected = "record still open")]
    fn take_refuses_open_record() {
        let mut b = Buf::new();
        b.begin(STATE);
        b.take();
    }
}
