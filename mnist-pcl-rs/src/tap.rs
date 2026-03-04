//! Streams training to the dashboard with the generic population records:
//! ON and OFF events, simple and complex spikes, the drive balance and the
//! learned simple-cell features.

use crate::train::{ImageResult, Monitor};
use crate::Net;
use snn_core::{Control, EpochResult, ImageInfo};
use snn_live::frame::{self, Buf};
use snn_live::pop::End;
use snn_live::{Hub, PopTap};
use std::sync::Arc;

pub struct Tap {
    t: PopTap,
    /// probed (population, unit): 2 simple, 3 complex
    probe: (usize, usize),
    /// decaying spike counts of simple cells
    usage: Vec<f32>,
    on: Vec<u16>,
    off: Vec<u16>,
    counts: [u32; 4],
}

impl Tap {
    pub fn new(hub: Arc<Hub>, ctl: Arc<Control>) -> Self {
        Self {
            t: PopTap::new(hub, ctl),
            probe: (2, 0),
            usage: Vec::new(),
            on: Vec::new(),
            off: Vec::new(),
            counts: [0; 4],
        }
    }

    /// Step scalars, probe variables and image scalars, as the dashboard is told them.
    pub fn series() -> ([&'static str; 2], [&'static str; 1], [&'static str; 2]) {
        (
            ["excitation", "inhibition"],
            ["V"],
            ["cell spikes", "suppression"],
        )
    }

    // a requested simple or complex cell, else the busiest simple cell
    fn pick_probe(&self, net: &Net) -> (usize, usize) {
        match self.t.probe_request() {
            Some((2, s)) if s < net.simple.len() => (2, s),
            Some((3, c)) if c < net.complex.len() => (3, c),
            _ => {
                let busiest = self
                    .usage
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1));
                (2, busiest.map_or(0, |(s, _)| s))
            }
        }
    }

    // each shared feature as an ON and an OFF image, shown at the centre location
    fn fields(&self, b: &mut Buf, net: &Net) {
        let (k, rf) = (net.p.features, net.p.rf);
        let centre = (net.n / 2) * net.n + net.n / 2;
        let units: Vec<u16> = (0..k).map(|f| (centre * k + f) as u16).collect();
        let mut use_f = vec![0.0f32; k];
        for (s, &u) in self.usage.iter().enumerate() {
            use_f[s % k] += u;
        }
        let top = use_f.iter().copied().fold(f32::MIN_POSITIVE, f32::max);
        let values: Vec<f32> = use_f.iter().map(|&u| u / top).collect();
        let data: Vec<f32> = (0..k)
            .flat_map(|f| net.feature(f).iter().copied())
            .collect();
        let f = frame::Fields {
            title: "Fig. 02 — simple-cell features · green ON · red OFF · bar = use",
            pop: 2,
            w: rf,
            h: rf,
            channels: 2,
            units: &units,
            labels: &vec![None; k],
            values: &values,
            data: &data,
        };
        frame::fields(b, &f);
    }
}

impl Monitor for Tap {
    fn image_start(&mut self, info: &ImageInfo, pixels: &[u8], net: &Net) {
        if self.usage.len() != net.simple.len() {
            self.usage = vec![0.0; net.simple.len()];
        }
        self.probe = self.pick_probe(net);
        self.counts = [0; 4];
        let thr = if self.probe.0 == 2 {
            net.p.v_thr
        } else {
            net.p.complex_thr
        };
        self.t
            .begin_image(info, pixels, (self.probe.0, self.probe.1, thr));
    }

    fn step(&mut self, t: usize, net: &Net, events: &[u16]) {
        let frame = (net.side * net.side) as u16;
        self.on.clear();
        self.off.clear();
        for &e in events {
            if e < frame {
                self.on.push(e);
            } else {
                self.off.push(e - frame);
            }
        }
        let lists: [&[u16]; 4] = [&self.on, &self.off, &net.fired_s, &net.fired_c];
        for (c, l) in self.counts.iter_mut().zip(lists) {
            *c += l.len() as u32;
        }
        if !self.t.streaming() {
            self.t.step(t, &[], &[], &[]);
            return;
        }
        let (pop, unit) = self.probe;
        let v = if pop == 2 {
            net.simple.peek(unit, t as u32)
        } else {
            net.complex.peek(unit, t as u32)
        };
        let drive = [net.excitation, net.inhibition];
        self.t.step(t, &lists, &drive, &[v]);
    }

    fn image_end(&mut self, r: &ImageResult, net: &Net) {
        for (u, &c) in self.usage.iter_mut().zip(&net.simple.counts) {
            *u = 0.995 * *u + c as f32;
        }
        let spikes = (self.counts[2] + self.counts[3]) as f32;
        let e = End {
            predicted: r.predicted,
            correct: r.correct,
            running_accuracy: r.running_accuracy,
            scores: &r.probs,
            spikes: &self.counts,
            scalars: &[spikes, r.suppression],
            activity: r.activity,
        };
        let mut fields = Buf::new();
        self.fields(&mut fields, net);
        let fields = fields.take();
        self.t.end_image(&r.info, &e, |_| {}, |b| b.raw(&fields));
    }

    fn epoch_end(&mut self, e: &EpochResult) {
        self.t.epoch_end(e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::Trainer;
    use crate::Params;
    use snn_core::dataset::synthetic;
    use snn_live::frame::records;
    use std::time::Duration;

    #[test]
    fn streams_events_spikes_and_features() {
        let hub = Arc::new(Hub::new());
        let mut v = hub.join();
        let ctl = Arc::new(Control::new(0));
        let mut tap = Tap::new(hub.clone(), ctl.clone());
        let data = synthetic(2, 1);
        let mut t = Trainer::new(Params::default(), 28, 28, 10, 1);
        t.fit(&data, None, 1, &mut tap, &ctl);
        let recs: Vec<(u8, Vec<u8>)> = v
            .poll(Duration::ZERO)
            .iter()
            .flat_map(|p| {
                records(p)
                    .iter()
                    .map(|r| (r.0, r.1.to_vec()))
                    .collect::<Vec<_>>()
            })
            .collect();
        let count = |k| recs.iter().filter(|r| r.0 == k).count();
        assert!(count(frame::IMAGE) >= 1);
        assert_eq!(count(frame::OUTCOME), count(frame::IMAGE));
        assert!(count(frame::SPIKES) >= 1);
        // pop 2, 32 features of 2 x 10 x 10 with their tile headers
        let f = &recs.iter().find(|r| r.0 == frame::FIELDS).unwrap().1;
        let title = u16::from_le_bytes([f[0], f[1]]) as usize;
        assert_eq!(&f[2 + title..][..8], &[2, 32, 0, 10, 0, 10, 0, 2]);
        assert_eq!(f.len(), 2 + title + 8 + 4 + 32 * 7 + 32 * 200);
    }
}
