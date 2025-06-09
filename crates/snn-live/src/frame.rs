//! Binary records streamed to the dashboard. Little endian. Each record is
//! `kind: u8, len: u32, payload`; one WebSocket message carries several.
//!
//! Kinds below 16 are shared by every model, except 3, 4, 5 and 9 which belong
//! to the winner-take-all model (mnist-snn-rs). Other models use 16 and up.

use snn_core::{EpochResult, ImageInfo};

pub const META: u8 = 1;
pub const IMAGE: u8 = 2;
pub const EPOCH: u8 = 6;
pub const STATE: u8 = 7;
pub const PROGRESS: u8 = 8;
pub const HISTORY: u8 = 10;
pub const PARAMS: u8 = 11;
pub const POPS: u8 = 12;
pub const SERIES: u8 = 16;
pub const SPIKES: u8 = 17;
pub const PROBE: u8 = 18;
pub const OUTCOME: u8 = 19;
pub const FIELDS: u8 = 20;
pub const MAP: u8 = 21;

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
    /// selects the dashboard view
    pub model: &'a str,
    pub dataset: &'a str,
    pub width: usize,
    pub height: usize,
    pub classes: usize,
    pub n_train: usize,
    pub n_test: usize,
    pub epochs: usize,
    /// simulation steps per image
    pub steps: usize,
    pub dt_ms: f32,
}

pub fn meta(b: &mut Buf, m: &Meta) {
    b.begin(META);
    b.u16(m.width as u16);
    b.u16(m.height as u16);
    b.u8(m.classes as u8);
    b.u32(m.n_train as u32);
    b.u32(m.n_test as u32);
    b.u16(m.epochs as u16);
    b.u16(m.steps as u16);
    b.f32(m.dt_ms);
    b.str(m.model);
    b.str(m.dataset);
    b.end();
}

/// One row of the model panel: a number, or text when it doesn't fit one.
pub struct Param<'a> {
    pub name: &'a str,
    pub value: f32,
    pub text: String,
    pub unit: &'a str,
}

impl<'a> Param<'a> {
    pub fn num(name: &'a str, value: f32, unit: &'a str) -> Self {
        Self {
            name,
            value,
            text: String::new(),
            unit,
        }
    }

    pub fn text(name: &'a str, text: String) -> Self {
        Self {
            name,
            value: f32::NAN,
            text,
            unit: "",
        }
    }
}

pub fn params(b: &mut Buf, title: &str, ps: &[Param]) {
    b.begin(PARAMS);
    b.str(title);
    b.u8(ps.len() as u8);
    for p in ps {
        b.str(p.name);
        b.f32(p.value);
        b.str(&p.text);
        b.str(p.unit);
    }
    b.end();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Input = 0,
    Latent = 1,
    Prediction = 2,
    Error = 3,
    Label = 4,
    Output = 5,
}

/// A group of neurons, drawn as a `w`×`h` grid.
pub struct Pop<'a> {
    pub name: &'a str,
    pub size: usize,
    pub w: usize,
    pub h: usize,
    pub role: Role,
}

impl<'a> Pop<'a> {
    /// A population laid out as a near-square grid.
    pub fn new(name: &'a str, size: usize, role: Role) -> Self {
        let w = ((size as f32).sqrt().ceil() as usize).max(1);
        Self {
            name,
            size,
            w,
            h: size.div_ceil(w),
            role,
        }
    }

    pub fn grid(name: &'a str, w: usize, h: usize, role: Role) -> Self {
        Self {
            name,
            size: w * h,
            w,
            h,
            role,
        }
    }
}

pub fn pops(b: &mut Buf, ps: &[Pop]) {
    b.begin(POPS);
    b.u8(ps.len() as u8);
    for p in ps {
        b.str(p.name);
        b.u16(p.size as u16);
        b.u16(p.w as u16);
        b.u16(p.h as u16);
        b.u8(p.role as u8);
    }
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

/// Opens a record of per-step entries; returns the offset of its step count
/// for [`Buf::patch_u16`].
pub fn steps_begin(b: &mut Buf, kind: u8, t0: usize) -> usize {
    b.begin(kind);
    b.u16(t0 as u16);
    let at = b.len();
    b.u16(0);
    at
}

/// A finished epoch; `images` is the training images seen by then.
pub fn epoch(b: &mut Buf, e: &EpochResult, images: u64) {
    b.begin(EPOCH);
    b.u16(e.epoch as u16);
    b.f32(e.train_accuracy);
    b.f32(e.test_accuracy.unwrap_or(f32::NAN));
    b.f32(e.images_per_sec);
    b.f64(images as f64);
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

/// Samples of one series as (images seen, value); series 0 is running accuracy.
pub fn history(b: &mut Buf, series: u8, points: &[(u32, f32)]) {
    b.begin(HISTORY);
    b.u8(series);
    b.u32(points.len() as u32);
    for &(n, a) in points {
        b.u32(n);
        b.f32(a);
    }
    b.end();
}

/// Names of the per-step scalars, the probe's variables and the per-image scalars
/// that SPIKES and OUTCOME carry.
pub fn series(b: &mut Buf, step: &[&str], probe: &[&str], image: &[&str]) {
    b.begin(SERIES);
    for names in [step, probe, image] {
        b.u8(names.len() as u8);
        names.iter().for_each(|n| b.str(n));
    }
    b.end();
}

/// One step inside an open SPIKES record: per population the units that spiked,
/// then the step scalars and the probed neuron's variables.
pub fn spikes_step(b: &mut Buf, pops: &[&[u16]], scalars: &[f32], probe: &[f32]) {
    for units in pops {
        b.u16(units.len() as u16);
        units.iter().for_each(|&i| b.u16(i));
    }
    scalars.iter().for_each(|&v| b.f32(v));
    probe.iter().for_each(|&v| b.f32(v));
}

/// The neuron the probe variables belong to, and its firing threshold.
pub fn probe(b: &mut Buf, pop: usize, unit: usize, threshold: f32) {
    b.begin(PROBE);
    b.u8(pop as u8);
    b.u16(unit as u16);
    b.f32(threshold);
    b.end();
}

/// How a presentation ended.
pub struct Outcome<'a> {
    pub seq: u32,
    pub predicted: Option<u8>,
    pub correct: bool,
    pub learn: bool,
    pub running_accuracy: f32,
    /// evidence per class, any scale
    pub scores: &'a [f32],
    /// spikes per population
    pub spikes: &'a [u32],
    /// per-image scalars, raw and smoothed over recent images
    pub scalars: &'a [f32],
    pub smoothed: &'a [f32],
}

pub fn outcome(b: &mut Buf, o: &Outcome) {
    b.begin(OUTCOME);
    b.u32(o.seq);
    b.u8(o.predicted.unwrap_or(255));
    b.u8(o.correct as u8);
    b.u8(o.learn as u8);
    b.f32(o.running_accuracy);
    for list in [o.scores, o.scalars, o.smoothed] {
        b.u8(list.len() as u8);
        list.iter().for_each(|&v| b.f32(v));
    }
    b.u8(o.spikes.len() as u8);
    o.spikes.iter().for_each(|&s| b.u32(s));
    b.end();
}

/// Byte for `v` on a scale of ±`scale` (signed, 128 is zero) or 0..`scale`.
fn quantise(v: f32, scale: f32, signed: bool) -> u8 {
    let s = scale.max(f32::MIN_POSITIVE);
    if signed {
        (128.0 + v / s * 127.0).round().clamp(1.0, 255.0) as u8
    } else {
        (v / s * 255.0).round().clamp(0.0, 255.0) as u8
    }
}

/// Weight images of some units of one population, e.g. receptive fields.
pub struct Fields<'a> {
    pub title: &'a str,
    /// population the units belong to, for click-to-probe
    pub pop: usize,
    pub w: usize,
    pub h: usize,
    /// images per unit, e.g. 2 for ON/OFF
    pub channels: usize,
    pub units: &'a [u16],
    pub labels: &'a [Option<u8>],
    /// one number per tile, shown as a bar
    pub values: &'a [f32],
    /// tile by tile, channel by channel, row-major
    pub data: &'a [f32],
}

/// Fields quantised around zero to the largest magnitude.
pub fn fields(b: &mut Buf, f: &Fields) {
    let scale = f.data.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    b.begin(FIELDS);
    b.str(f.title);
    b.u8(f.pop as u8);
    b.u16(f.units.len() as u16);
    b.u16(f.w as u16);
    b.u16(f.h as u16);
    b.u8(f.channels as u8);
    b.f32(scale);
    for k in 0..f.units.len() {
        b.u16(f.units[k]);
        b.u8(f.labels[k].unwrap_or(255));
        b.f32(f.values[k]);
    }
    f.data.iter().for_each(|&v| b.u8(quantise(v, scale, true)));
    b.end();
}

/// A named image the view draws, e.g. the network's prediction of the input.
pub fn map(b: &mut Buf, id: u8, w: usize, h: usize, signed: bool, scale: f32, data: &[f32]) {
    b.begin(MAP);
    b.u8(id);
    b.u16(w as u16);
    b.u16(h as u16);
    b.u8(signed as u8);
    b.f32(scale);
    data.iter().for_each(|&v| b.u8(quantise(v, scale, signed)));
    b.end();
}

/// Splits a packet into (kind, payload) records.
pub fn records(bytes: &[u8]) -> Vec<(u8, &[u8])> {
    let mut out = Vec::new();
    let mut at = 0;
    while at + 5 <= bytes.len() {
        let len = u32::from_le_bytes(bytes[at + 1..at + 5].try_into().unwrap()) as usize;
        out.push((bytes[at], &bytes[at + 5..at + 5 + len]));
        at += 5 + len;
    }
    assert_eq!(at, bytes.len(), "truncated record");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_carry_their_length() {
        let mut b = Buf::new();
        state(&mut b, true, 1000, false);
        let e = EpochResult {
            epoch: 2,
            train_accuracy: 0.5,
            test_accuracy: None,
            images_per_sec: 100.0,
        };
        epoch(&mut b, &e, 7);
        let bytes = b.take();
        let r = records(&bytes);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0], (STATE, &[1, 0xe8, 3, 0, 0, 0][..]));
        assert_eq!(r[1].0, EPOCH);
        assert!(f32::from_le_bytes(r[1].1[6..10].try_into().unwrap()).is_nan());
        assert_eq!(f64::from_le_bytes(r[1].1[14..22].try_into().unwrap()), 7.0);
        assert!(b.is_empty());
    }

    #[test]
    fn steps_record_patches_its_count() {
        let mut b = Buf::new();
        let at = steps_begin(&mut b, 40, 7);
        b.u8(1);
        b.u8(2);
        b.patch_u16(at, 2);
        b.end();
        let bytes = b.take();
        assert_eq!(records(&bytes), vec![(40, &[7, 0, 2, 0, 1, 2][..])]);
    }

    #[test]
    fn meta_params_and_pops_layout() {
        let mut b = Buf::new();
        let m = Meta {
            model: "wta",
            dataset: "synthetic",
            width: 28,
            height: 28,
            classes: 10,
            n_train: 10,
            n_test: 5,
            epochs: 1,
            steps: 200,
            dt_ms: 1.0,
        };
        meta(&mut b, &m);
        let ps = [
            Param::num("Leak", 0.15, "/ms"),
            Param::text("Weights", "0-2".into()),
        ];
        params(&mut b, "LIF", &ps);
        let ps = [
            Pop::grid("input", 28, 28, Role::Input),
            Pop::new("output", 40, Role::Output),
        ];
        pops(&mut b, &ps);
        let bytes = b.take();
        let r = records(&bytes);
        assert_eq!(r[0].1.len(), 21 + 5 + 11);
        assert_eq!(&r[0].1[21..26], b"\x03\x00wta");
        let p = r[1].1;
        assert_eq!(p[5], 2, "two params");
        assert_eq!(&p[6..12], b"\x04\x00Leak");
        let q = r[2].1;
        assert_eq!(q[0], 2);
        // 40 outputs as a 7x6 grid
        let out = &q[1 + 7 + 7..];
        assert_eq!(&out[..8], b"\x06\x00output");
        assert_eq!(&out[8..15], &[40, 0, 7, 0, 6, 0, Role::Output as u8]);
    }

    #[test]
    fn history_carries_its_series() {
        let mut b = Buf::new();
        history(&mut b, 1, &[(5, 0.25)]);
        let bytes = b.take();
        let r = records(&bytes);
        assert_eq!(r[0].1[..5], [1, 1, 0, 0, 0]);
        assert_eq!(r[0].1.len(), 1 + 4 + 8);
    }

    #[test]
    fn spikes_step_layout() {
        let mut b = Buf::new();
        let at = steps_begin(&mut b, SPIKES, 1);
        spikes_step(&mut b, &[&[3, 9], &[]], &[0.5], &[1.0, 2.0]);
        b.patch_u16(at, 1);
        b.end();
        let bytes = b.take();
        let p = records(&bytes)[0].1;
        assert_eq!(&p[..10], &[1, 0, 1, 0, 2, 0, 3, 0, 9, 0]);
        assert_eq!(&p[10..12], &[0, 0]);
        assert_eq!(p.len(), 12 + 4 + 8);
    }

    #[test]
    fn fields_quantise_around_zero() {
        let mut b = Buf::new();
        let f = Fields {
            title: "W",
            pop: 2,
            w: 2,
            h: 1,
            channels: 1,
            units: &[7],
            labels: &[None],
            values: &[0.5],
            data: &[-2.0, 1.0],
        };
        fields(&mut b, &f);
        let bytes = b.take();
        let p = records(&bytes)[0].1;
        // title, pop, n, w, h, channels, scale, then unit 7, no label, value
        assert_eq!(&p[..11], &[1, 0, b'W', 2, 1, 0, 2, 0, 1, 0, 1]);
        assert_eq!(f32::from_le_bytes(p[11..15].try_into().unwrap()), 2.0);
        assert_eq!(&p[15..18], &[7, 0, 255]);
        assert_eq!(&p[22..], &[1, 192]);
    }

    #[test]
    fn outcome_and_map_layout() {
        let mut b = Buf::new();
        let o = Outcome {
            seq: 3,
            predicted: Some(4),
            correct: true,
            learn: false,
            running_accuracy: 0.5,
            scores: &[1.0, 2.0],
            spikes: &[10, 20, 30],
            scalars: &[0.1],
            smoothed: &[0.2],
        };
        outcome(&mut b, &o);
        map(&mut b, 1, 2, 1, false, 2.0, &[1.0, 4.0]);
        let bytes = b.take();
        let r = records(&bytes);
        assert_eq!(
            r[0].1.len(),
            4 + 3 + 4 + (1 + 8) + (1 + 4) + (1 + 4) + (1 + 12)
        );
        assert_eq!(&r[0].1[4..7], &[4, 1, 0]);
        assert_eq!(r[1].1, &[1, 2, 0, 1, 0, 0, 0, 0, 0, 64, 128, 255]);
    }

    #[test]
    #[should_panic(expected = "record still open")]
    fn take_refuses_open_record() {
        let mut b = Buf::new();
        b.begin(STATE);
        b.take();
    }
}
