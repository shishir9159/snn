//! Streams training to the dashboard: the spike waves of S1, L1, L2 and the
//! answering decision neuron with the generic population records, plus a
//! DCSNN record per streamed image with the phase, the neurons that learn, the
//! learning rates and the classes' evidence over time.

use crate::conv::Conv;
use crate::train::{ImageResult, Monitor, Spikes};
use crate::wave::{Shape, Wave};
use crate::{Net, Phase};
use snn_core::{Control, EpochResult, ImageInfo};
use snn_live::frame::{self, Buf};
use snn_live::pop::End;
use snn_live::{Hub, PopTap};
use std::sync::Arc;

/// Record kind of the layer shapes, sent once with the meta packet.
pub const ARCH: u8 = 49;
/// Record kind of what one image did beyond its spikes.
pub const DCSNN: u8 = 50;
/// L2 features shown, busiest first.
const FIELDS: usize = 64;

/// Per layer (S1, L1, L2, L3) its maps, height, width, window (0 for S1),
/// threshold (0 for none), the pooling after it, and the winners that learn
/// with the radius each silences; then the decision maps per class.
pub fn arch(b: &mut Buf, net: &Net) {
    b.begin(ARCH);
    b.u8(4);
    let (l1, l2) = (&net.p.l1, &net.p.l2);
    let layers = [
        (net.input.shape, 0, 0.0, 1, 0, 0),
        (
            net.l1.output,
            l1.kernel,
            l1.threshold,
            l1.pool,
            l1.winners,
            l1.radius,
        ),
        (
            net.l2.output,
            l2.kernel,
            l2.threshold,
            l2.pool,
            l2.winners,
            l2.radius,
        ),
        (net.l3.output, net.l3.kernel, 0.0, 1, 1, 0),
    ];
    for (s, kernel, th, pool, winners, radius) in layers {
        b.u16(s.maps as u16);
        b.u16(s.h as u16);
        b.u16(s.w as u16);
        b.u8(kernel as u8);
        b.f32(th);
        b.u8(pool as u8);
        b.u8(winners as u8);
        b.u8(radius as u8);
    }
    b.u8(net.p.per_class as u8);
    b.end();
}

/// Phase, layers run, the STDP rates of L1 and L2 and the R-STDP rates after
/// a hit and a miss, the neurons that learn (layer, unit, step, potential),
/// the convergence of each layer, then the largest decision potential of each
/// class after each step.
pub fn record(b: &mut Buf, phase: Phase, net: &Net) {
    b.begin(DCSNN);
    b.u8(phase.index() as u8);
    b.u8(net.depth as u8);
    for (x, y) in [net.rates[0], net.rates[1], net.reward, net.punish] {
        b.f32(x);
        b.f32(y);
    }
    b.u8(net.winners.len() as u8);
    for w in &net.winners {
        b.u8(net.depth as u8);
        b.u16(w.unit);
        b.u8(w.time);
        b.f32(w.pot);
    }
    for c in [&net.l1, &net.l2, &net.l3] {
        b.f32(c.convergence());
    }
    b.u8(net.leader.len() as u8);
    b.u8(net.classes as u8);
    net.evidence.iter().for_each(|&e| b.f32(e));
    b.end();
}

pub struct Tap {
    t: PopTap,
    phase: Phase,
    /// probed (population, unit): 1 L1, 2 L2, 3 L3
    probe: (usize, usize),
    trace: Vec<f32>,
    /// decaying spike counts of each L1 and L2 map
    usage: [Vec<f32>; 2],
    counts: [u32; 4],
    winner: [u16; 1],
}

impl Tap {
    pub fn new(hub: Arc<Hub>, ctl: Arc<Control>) -> Self {
        Self {
            t: PopTap::new(hub, ctl),
            phase: Phase::Layer1,
            probe: (1, 0),
            trace: Vec::new(),
            usage: [Vec::new(), Vec::new()],
            counts: [0; 4],
            winner: [0],
        }
    }

    /// Step scalars, probe variables and image scalars, as the dashboard is told them.
    pub fn series() -> ([&'static str; 1], [&'static str; 1], [&'static str; 3]) {
        (["lead"], ["V"], ["convergence", "L1 spikes", "L2 spikes"])
    }

    // a requested neuron of a layer that ran, else the first that learns
    fn pick_probe(&self, net: &Net) -> (usize, usize) {
        let sizes = [
            net.l1.output.len(),
            net.l2.output.len(),
            net.l3.output.len(),
        ];
        match self.t.probe_request() {
            Some((pop, u)) if (1..=net.depth).contains(&pop) && u < sizes[pop - 1] => (pop, u),
            _ => match net.winners.first() {
                Some(w) => (net.depth, w.unit as usize),
                None => (net.depth, 0),
            },
        }
    }

    // a population's layer and the wave it reads
    fn layer(net: &Net, pop: usize) -> (&Conv, &Wave) {
        match pop {
            1 => (&net.l1, &net.input),
            2 => (&net.l2, &net.pool1),
            _ => (&net.l3, &net.pool2),
        }
    }

    // the decision layer's lead of its first class over the second after step `t`
    fn lead(net: &Net, t: usize) -> f32 {
        let c = net.classes;
        let Some(ev) = net.evidence.get(t * c..(t + 1) * c) else {
            return 0.0;
        };
        let (mut a, mut b) = (0.0f32, 0.0f32);
        for &e in ev {
            if e > a {
                b = a;
                a = e;
            } else if e > b {
                b = e;
            }
        }
        a - b
    }

    // L1 features in phase 1, else the busiest L2 features tinted by the class
    // their decision weights favour
    fn fields(&self, b: &mut Buf, net: &Net) {
        let mut l1 = vec![Vec::new(); net.l1.output.maps];
        for (m, f) in l1.iter_mut().enumerate() {
            net.l1_feature(m, f);
        }
        let side1 = (l1[0].len() as f32).sqrt() as usize;
        let centre = |s: Shape, m: usize| s.unit(s.h / 2, s.w / 2, m) as u16;
        if self.phase == Phase::Layer1 {
            let use1 = norm(&self.usage[0]);
            let units: Vec<u16> = (0..l1.len()).map(|m| centre(net.l1.output, m)).collect();
            let f = frame::Fields {
                title: "Fig. 02 — L1 features through the DoG filters · bar = use",
                pop: 1,
                w: side1,
                h: side1,
                channels: 1,
                units: &units,
                labels: &vec![None; l1.len()],
                values: &use1,
                data: &l1.concat(),
            };
            return frame::fields(b, &f);
        }
        let mut maps: Vec<usize> = (0..net.l2.output.maps).collect();
        let usage = &self.usage[1];
        maps.sort_by(|&a, &c| usage[c].total_cmp(&usage[a]));
        maps.truncate(FIELDS);
        // the decision weights mean nothing before R-STDP has moved them
        let labels: Vec<Option<u8>> = if self.phase == Phase::Decision {
            let votes = votes(net);
            maps.iter().map(|&m| votes[m]).collect()
        } else {
            vec![None; maps.len()]
        };
        let use2 = norm(usage);
        let values: Vec<f32> = maps.iter().map(|&m| use2[m]).collect();
        let (mut data, mut g) = (Vec::new(), Vec::new());
        for &m in &maps {
            net.l2_feature(m, &l1, &mut g);
            data.extend_from_slice(&g);
        }
        let side2 = (g.len() as f32).sqrt() as usize;
        let units: Vec<u16> = maps.iter().map(|&m| centre(net.l2.output, m)).collect();
        let f = frame::Fields {
            title: if self.phase == Phase::Decision {
                "Fig. 02 — busiest L2 features · tint = class its decision weights favour"
            } else {
                "Fig. 02 — busiest L2 features through L1 and the filters · bar = use"
            },
            pop: 2,
            w: side2,
            h: side2,
            channels: 1,
            units: &units,
            labels: &labels,
            values: &values,
            data: &data,
        };
        frame::fields(b, &f);
    }
}

fn norm(v: &[f32]) -> Vec<f32> {
    let top = v.iter().copied().fold(f32::MIN_POSITIVE, f32::max);
    v.iter().map(|&u| u / top).collect()
}

/// For each L2 map, the class whose decision maps weigh it most above their
/// lower bound, if one does more than the mean.
fn votes(net: &Net) -> Vec<Option<u8>> {
    let (l3, per) = (&net.l3, net.p.per_class);
    let (ci, k) = (l3.input.maps, l3.kernel);
    let mut sum = vec![0.0f32; ci * net.classes];
    for ky in 0..k {
        for kx in 0..k {
            for c in 0..ci {
                for m in 0..l3.output.maps {
                    sum[c * net.classes + m / per] += l3.weight(m, ky, kx, c) - l3.bounds.0;
                }
            }
        }
    }
    sum.chunks_exact(net.classes)
        .map(|row| {
            let mean = row.iter().sum::<f32>() / row.len() as f32;
            let (k, &v) = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
            (v > mean).then_some(k as u8)
        })
        .collect()
}

impl Monitor for Tap {
    fn image_start(&mut self, info: &ImageInfo, phase: Phase, pixels: &[u8], net: &Net) {
        self.phase = phase;
        self.counts = [0; 4];
        self.probe = self.pick_probe(net);
        let (pop, unit) = self.probe;
        let (conv, input) = Self::layer(net, pop);
        let th = if conv.threshold.is_finite() {
            conv.threshold
        } else {
            0.0
        };
        if self.t.begin_image(info, pixels, (pop, unit, th)) {
            conv.trace(input, unit, net.p.steps, &mut self.trace);
            self.t.extra(|b| record(b, phase, net));
        }
    }

    fn step(&mut self, t: usize, net: &Net, s: &Spikes) {
        // decision neurons answer at the last step
        let last = t == net.p.steps && net.depth == 3;
        let l3: &[u16] = match net.winners.first() {
            Some(w) if last => {
                self.winner = [w.unit];
                &self.winner
            }
            _ => &[],
        };
        let lists = [s.input, s.l1, s.l2, l3];
        for (c, l) in self.counts.iter_mut().zip(lists) {
            *c += l.len() as u32;
        }
        if !self.t.streaming() {
            self.t.step(t, &[], &[], &[]);
            return;
        }
        let v = self.trace.get(t - 1).copied().unwrap_or(0.0);
        self.t.step(t, &lists, &[Self::lead(net, t - 1)], &[v]);
    }

    fn image_end(&mut self, r: &ImageResult, net: &Net) {
        let waves = [&net.fire1, &net.fire2];
        for ((usage, fire), ran) in self.usage.iter_mut().zip(waves).zip([1, 2]) {
            let maps = fire.out.shape.maps;
            if usage.len() != maps {
                *usage = vec![0.0; maps];
            }
            if net.depth < ran {
                continue;
            }
            usage.iter_mut().for_each(|u| *u *= 0.995);
            for t in 0..net.p.steps {
                for &u in fire.out.at(t) {
                    usage[u as usize % maps] += 1.0;
                }
            }
        }
        // potentials are large and close; the lead over the weakest class shows
        let c = net.classes;
        let mut scores = match net.evidence.len() {
            0 => vec![0.0; c],
            n => net.evidence[n - c..].to_vec(),
        };
        let lo = scores.iter().copied().fold(f32::INFINITY, f32::min);
        scores.iter_mut().for_each(|s| *s -= lo);
        let conv = [&net.l1, &net.l2, &net.l3][r.phase.index()].convergence();
        let e = End {
            predicted: r.predicted,
            correct: r.correct,
            running_accuracy: r.running_accuracy,
            scores: &scores,
            spikes: &self.counts,
            scalars: &[conv, self.counts[1] as f32, self.counts[2] as f32],
            activity: r.activity,
        };
        let fields = if self.t.fields_due() {
            let mut b = Buf::new();
            self.fields(&mut b, net);
            b.take()
        } else {
            Vec::new()
        };
        self.t.end_image(&r.info, &e, |_| {}, |b| b.raw(&fields));
    }

    fn epoch_end(&mut self, e: &EpochResult, _net: &Net) {
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
    fn streams_waves_records_and_features() {
        let hub = Arc::new(Hub::new());
        let mut v = hub.join();
        let ctl = Arc::new(Control::new(0));
        let mut tap = Tap::new(hub.clone(), ctl.clone());
        let data = synthetic(2, 1);
        let mut p = Params::default();
        p.l1.epochs = 1;
        p.l2.epochs = 0;
        let mut t = Trainer::new(p, 28, 28, 10, 1).unwrap();
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
        assert_eq!(count(DCSNN), count(frame::IMAGE));
        assert_eq!(count(frame::OUTCOME), count(frame::IMAGE));
        assert!(count(frame::SPIKES) >= 1 && count(frame::FIELDS) >= 1);
        // phase 1: 30 L1 features of 17 x 17
        let f = &recs.iter().find(|r| r.0 == frame::FIELDS).unwrap().1;
        let title = u16::from_le_bytes([f[0], f[1]]) as usize;
        assert_eq!(&f[2 + title..][..8], &[1, 30, 0, 17, 0, 17, 0, 1]);
        // the first record is phase 1, run one layer deep, with no evidence
        let d = &recs.iter().find(|r| r.0 == DCSNN).unwrap().1;
        assert_eq!((d[0], d[1]), (0, 1));
        let n = d[34] as usize;
        assert_eq!(d.len(), 35 + n * 8 + 12 + 2);
        let mut b = Buf::new();
        arch(&mut b, &t.net);
        let bytes = b.take();
        let a = records(&bytes)[0].1;
        assert_eq!(a.len(), 1 + 4 * 14 + 1);
        assert_eq!(&a[1..7], &[6, 0, 28, 0, 28, 0]);
        // L1: 5x5 windows, threshold 15, pooling 2, 5 winners within radius 3
        assert_eq!(a[21], 5);
        assert_eq!(f32::from_le_bytes(a[22..26].try_into().unwrap()), 15.0);
        assert_eq!(&a[26..29], &[2, 5, 3]);
    }
}
