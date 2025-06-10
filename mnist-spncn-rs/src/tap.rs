//! Streams training to the dashboard with the generic population records, plus
//! the network's prediction of the image every few steps.

use crate::readout::Readout;
use crate::train::{ImageResult, Monitor};
use crate::Net;
use snn_core::{Control, EpochResult, ImageInfo};
use snn_live::frame::{self, Buf};
use snn_live::pop::End;
use snn_live::{Hub, PopTap};
use std::sync::Arc;

/// MAP ids: the prediction of the input now, and its mean over the settled second half.
pub const PREDICTION: u8 = 0;
pub const SETTLED: u8 = 1;
/// Steps between prediction maps.
const MAP_EVERY: usize = 4;
/// Fields shown, busiest layer-1 units first.
const FIELDS: usize = 64;

pub struct Tap {
    t: PopTap,
    side: (usize, usize),
    /// probed (population, unit)
    probe: (usize, usize),
    /// decaying spike counts of layer-1 units
    usage: Vec<f32>,
    settled: Vec<f32>,
    input_spikes: u32,
}

impl Tap {
    pub fn new(hub: Arc<Hub>, ctl: Arc<Control>, width: usize, height: usize) -> Self {
        Self {
            t: PopTap::new(hub, ctl),
            side: (width, height),
            probe: (1, 0),
            usage: Vec::new(),
            settled: Vec::new(),
            input_spikes: 0,
        }
    }

    /// Population names, step scalars, probe variables and image scalars, as the
    /// dashboard is told them.
    pub fn series(depth: usize) -> (Vec<String>, [&'static str; 3], [&'static str; 2]) {
        let sup = ["⁰", "¹", "²", "³", "⁴", "⁵"];
        let step = (0..depth).map(|l| format!("e{}", sup[l.min(5)])).collect();
        (step, ["V_m", "J", "z"], ["unexplained", "spikes / img"])
    }

    // the requested neuron if it exists, else the busiest layer-1 unit
    fn pick_probe(&self, net: &Net) -> (usize, usize) {
        match self.t.probe_request() {
            Some((pop, unit))
                if pop >= 1 && pop <= net.lat.len() && unit < net.lat[pop - 1].len() =>
            {
                (pop, unit)
            }
            _ => {
                let busiest = self
                    .usage
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1));
                (1, busiest.map_or(0, |(j, _)| j))
            }
        }
    }

    fn fields(&self, b: &mut Buf, net: &Net, readout: &Readout) {
        let mut units: Vec<u16> = (0..net.lat[0].len() as u16).collect();
        units.sort_by(|&a, &c| self.usage[c as usize].total_cmp(&self.usage[a as usize]));
        units.truncate(FIELDS);
        let top = self.usage.iter().copied().fold(f32::MIN_POSITIVE, f32::max);
        let labels: Vec<Option<u8>> = units
            .iter()
            .map(|&j| readout.favourite(j as usize))
            .collect();
        let values: Vec<f32> = units
            .iter()
            .map(|&j| self.usage[j as usize] / top)
            .collect();
        let data: Vec<f32> = units
            .iter()
            .flat_map(|&j| net.field(j as usize).iter().copied())
            .collect();
        let f = frame::Fields {
            title: "Fig. 02 — generative fields W¹ · tint = readout class · bar = use",
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
        if self.usage.len() != net.lat[0].len() {
            self.usage = vec![0.0; net.lat[0].len()];
        }
        self.probe = self.pick_probe(net);
        let (pop, unit) = self.probe;
        let thr = net.lat[pop - 1].thr[unit];
        self.t.begin_image(info, pixels, (pop, unit, thr));
        self.settled.clear();
        self.settled.resize(net.sizes[0], 0.0);
        self.input_spikes = 0;
    }

    fn step(&mut self, t: usize, net: &Net) {
        self.input_spikes += net.input.len() as u32;
        let steps = net.p.steps();
        if t > steps / 2 {
            let k = 1.0 / (steps - steps / 2) as f32;
            self.settled
                .iter_mut()
                .zip(&net.pred[0])
                .for_each(|(s, &m)| *s += k * m);
        }
        if !self.t.streaming() {
            self.t.step(t, &[], &[], &[]);
            return;
        }
        let mut lists: Vec<&[u16]> = vec![&net.input];
        lists.extend(net.lat.iter().map(|l| l.spikes.as_slice()));
        let (pop, unit) = self.probe;
        let l = &net.lat[pop - 1];
        self.t
            .step(t, &lists, &net.energy, &[l.v[unit], l.j[unit], l.z[unit]]);
        if t % MAP_EVERY == 0 {
            let (w, h) = self.side;
            let pred = &net.pred[0];
            let scale = pred.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            self.t
                .extra(|b| frame::map(b, PREDICTION, w, h, true, scale, pred));
        }
    }

    fn image_end(&mut self, r: &ImageResult, net: &Net, readout: &Readout) {
        let mut spikes = vec![self.input_spikes];
        spikes.extend(net.lat.iter().map(|l| l.counts.iter().sum::<u32>()));
        for (u, &c) in self.usage.iter_mut().zip(&net.lat[0].counts) {
            *u = 0.995 * *u + c as f32;
        }
        let latent: u32 = spikes[1..].iter().sum();
        let e = End {
            predicted: r.predicted,
            correct: r.correct,
            running_accuracy: r.running_accuracy,
            scores: &r.probs,
            spikes: &spikes,
            scalars: &[r.unexplained, latent as f32],
        };
        let (w, h) = self.side;
        let settled = &self.settled;
        let scale = settled.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let streamed = |b: &mut Buf| frame::map(b, SETTLED, w, h, true, scale, settled);
        let mut fields = Buf::new();
        self.fields(&mut fields, net, readout);
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

    fn small() -> Params {
        Params {
            layers: vec![30, 10],
            ..Params::default()
        }
    }

    #[test]
    fn streams_spikes_maps_and_fields() {
        let hub = Arc::new(Hub::new());
        let mut v = hub.join();
        let ctl = Arc::new(Control::new(0));
        let mut tap = Tap::new(hub.clone(), ctl.clone(), 28, 28);
        let data = synthetic(2, 1);
        let mut t = Trainer::new(small(), 784, 10, 1);
        t.fit(&data, None, 1, &mut tap, &ctl);
        let all: Vec<u8> = v
            .poll(Duration::ZERO)
            .iter()
            .flat_map(|p| records(p).iter().map(|r| r.0).collect::<Vec<_>>())
            .collect();
        let count = |k| all.iter().filter(|&&x| x == k).count();
        assert!(count(frame::IMAGE) >= 1);
        assert_eq!(count(frame::PROBE), count(frame::IMAGE));
        assert_eq!(count(frame::OUTCOME), count(frame::IMAGE));
        assert!(count(frame::MAP) >= 200 / MAP_EVERY);
        assert!(count(frame::SPIKES) >= 1 && count(frame::FIELDS) >= 1);
    }

    #[test]
    fn probe_follows_a_valid_request_only() {
        let hub = Arc::new(Hub::new());
        let ctl = Arc::new(Control::new(0));
        let tap = Tap::new(hub, ctl.clone(), 28, 28);
        let net = Net::new(784, small(), &mut snn_core::Rng::new(1));
        ctl.set_probe(Some((2, 7)));
        assert_eq!(tap.pick_probe(&net), (2, 7));
        ctl.set_probe(Some((2, 70)));
        assert_eq!(tap.pick_probe(&net).0, 1);
        ctl.set_probe(Some((0, 3)));
        assert_eq!(tap.pick_probe(&net).0, 1);
    }
}
