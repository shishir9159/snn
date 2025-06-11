//! Streams training to the dashboard: the replayed spikes with the generic
//! population records, then an INFER record with what inference did.

use crate::train::{ImageResult, Monitor, Spikes};
use crate::Net;
use snn_core::{Control, EpochResult, ImageInfo};
use snn_live::frame::{self, Buf};
use snn_live::pop::End;
use snn_live::{Hub, PopTap};
use std::sync::Arc;

/// Record kind of the inference summary.
pub const INFER: u8 = 48;
/// Fields shown, busiest hidden neurons first.
const FIELDS: usize = 100;

/// Free energy per iteration, then per hidden neuron its forward and inferred
/// time and whether dropout silenced it, then per output its forward, predicted
/// and target time.
pub fn infer(b: &mut Buf, net: &Net) {
    b.begin(INFER);
    b.u8(net.energy.len() as u8);
    net.energy.iter().for_each(|&f| b.f32(f));
    b.u16(net.hid.len() as u16);
    for j in 0..net.hid.len() {
        b.f32(net.hid_pred[j]);
        b.f32(net.hid[j]);
        b.u8(net.dropped[j] as u8);
    }
    b.u8(net.classes as u8);
    for k in 0..net.classes {
        b.f32(net.out_fwd[k]);
        b.f32(net.out_pred[k]);
        b.f32(net.target[k]);
    }
    b.end();
}

pub struct Tap {
    t: PopTap,
    side: (usize, usize),
    /// probed (population, unit): 1 hidden, 2 output
    probe: (usize, usize),
    /// probed neuron's potential so far
    v: f32,
    /// decaying count of images each hidden neuron fired in
    usage: Vec<f32>,
    fired: u32,
    counts: [u32; 3],
}

impl Tap {
    pub fn new(hub: Arc<Hub>, ctl: Arc<Control>, width: usize, height: usize) -> Self {
        Self {
            t: PopTap::new(hub, ctl),
            side: (width, height),
            probe: (2, 0),
            v: 0.0,
            usage: Vec::new(),
            fired: 0,
            counts: [0; 3],
        }
    }

    /// Step scalars, probe variables and image scalars, as the dashboard is told them.
    pub fn series() -> ([&'static str; 1], [&'static str; 1], [&'static str; 2]) {
        (["hidden fired"], ["V"], ["free energy", "first spike"])
    }

    // a requested hidden or output neuron, else the label's output
    fn pick_probe(&self, net: &Net, label: u8) -> (usize, usize) {
        match self.t.probe_request() {
            Some((1, j)) if j < net.p.hidden => (1, j),
            Some((2, k)) if k < net.classes => (2, k),
            _ => (2, label as usize),
        }
    }

    fn fields(&self, b: &mut Buf, net: &Net) {
        let mut units: Vec<u16> = (0..net.p.hidden as u16).collect();
        units.sort_by(|&a, &c| self.usage[c as usize].total_cmp(&self.usage[a as usize]));
        units.truncate(FIELDS);
        let c = net.classes;
        let labels: Vec<Option<u8>> = units
            .iter()
            .map(|&j| {
                let row = &net.w2[j as usize * c..][..c];
                let (k, &w) = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
                (w > 0.0).then_some(k as u8)
            })
            .collect();
        let top = self.usage.iter().copied().fold(f32::MIN_POSITIVE, f32::max);
        let values: Vec<f32> = units
            .iter()
            .map(|&j| self.usage[j as usize] / top)
            .collect();
        let (mut data, mut field) = (Vec::new(), Vec::new());
        for &j in &units {
            net.field(j as usize, &mut field);
            data.extend_from_slice(&field);
        }
        let f = frame::Fields {
            title: "Fig. 02 — hidden receptive fields · tint = output it drives · bar = use",
            pop: 1,
            w: self.side.0,
            h: self.side.1,
            channels: 1,
            units: &units,
            labels: &labels,
            values: &values,
            data: &data,
        };
        frame::fields(b, &f);
    }
}

impl Monitor for Tap {
    fn image_start(&mut self, info: &ImageInfo, pixels: &[u8], net: &Net) {
        if self.usage.len() != net.p.hidden {
            self.usage = vec![0.0; net.p.hidden];
        }
        self.probe = self.pick_probe(net, info.label);
        self.v = 0.0;
        self.fired = 0;
        self.counts = [0; 3];
        let (pop, unit) = self.probe;
        self.t
            .begin_image(info, pixels, (pop, unit, net.p.threshold));
    }

    fn step(&mut self, t: usize, net: &Net, s: &Spikes) {
        for (c, list) in self.counts.iter_mut().zip([s.input, s.hidden, s.output]) {
            *c += list.len() as u32;
        }
        self.fired += s.hidden.len() as u32;
        if !self.t.streaming() {
            self.t.step(t, &[], &[], &[]);
            return;
        }
        let (h, c) = (net.p.hidden, net.classes);
        let (pop, unit) = self.probe;
        self.v += match pop {
            1 => s
                .input
                .iter()
                .map(|&i| net.w1[i as usize * h + unit])
                .sum::<f32>(),
            _ => s
                .hidden
                .iter()
                .map(|&j| net.w2[j as usize * c + unit])
                .sum(),
        };
        let active = net.dropped.iter().filter(|&&d| !d).count().max(1);
        let share = self.fired as f32 / active as f32;
        self.t
            .step(t, &[s.input, s.hidden, s.output], &[share], &[self.v]);
    }

    fn image_end(&mut self, r: &ImageResult, net: &Net) {
        for (u, &t) in self.usage.iter_mut().zip(&net.hid_pred) {
            *u = 0.995 * *u + (t < net.p.t_max) as u8 as f32;
        }
        let t_max = net.p.t_max;
        let scores: Vec<f32> = net.out_fwd.iter().map(|&t| (t_max - t) / t_max).collect();
        let free = if r.info.learn {
            *net.energy.last().unwrap_or(&0.0)
        } else {
            0.0
        };
        let e = End {
            predicted: r.predicted,
            correct: r.correct,
            running_accuracy: r.running_accuracy,
            scores: &scores,
            spikes: &self.counts,
            scalars: &[free, r.first_spike],
        };
        let learn = r.info.learn;
        let streamed = |b: &mut Buf| {
            if learn {
                infer(b, net);
            }
        };
        let mut fields = Buf::new();
        self.fields(&mut fields, net);
        let fields = fields.take();
        self.t.end_image(&r.info, &e, streamed, |b| b.raw(&fields));
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
    fn streams_spikes_inference_and_fields() {
        let hub = Arc::new(Hub::new());
        let mut v = hub.join();
        let ctl = Arc::new(Control::new(0));
        let mut tap = Tap::new(hub.clone(), ctl.clone(), 28, 28);
        let data = synthetic(2, 1);
        let mut t = Trainer::new(Params::default(), 784, 10, 1);
        t.fit(&data, None, 1, &mut tap, &ctl);
        let packets = v.poll(Duration::ZERO);
        let recs: Vec<(u8, Vec<u8>)> = packets
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
        assert_eq!(count(INFER), count(frame::IMAGE));
        assert!(count(frame::SPIKES) >= 1 && count(frame::FIELDS) >= 1);
        // iterations + 1 energies, 200 hidden of 9 bytes, 10 outputs of 12
        let inf = &recs.iter().find(|r| r.0 == INFER).unwrap().1;
        assert_eq!(inf.len(), 1 + 11 * 4 + 2 + 200 * 9 + 1 + 10 * 12);
    }
}
