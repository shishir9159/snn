//! Training loop: encodes spike times, runs the forward pass, replays it step
//! by step for monitors, then infers and learns from the label.

use crate::{Net, Params};
use snn_core::{
    Activity, Control, Dataset, EpochResult, ImageInfo, Pacer, Pass, Perturb, Recent, Rng,
};
use snn_live::app::Printer;
use std::time::Instant;

/// Images in the running-accuracy window.
const RECENT: usize = 500;
/// Paced steps each inference iteration is given, so it can be watched.
const HOLD: usize = 8;

/// Units of one layer bucketed by spike time, so a replay finds each step's spikes.
#[derive(Debug, Clone, Default)]
pub struct Schedule {
    units: Vec<u16>,
    start: Vec<u32>,
}

impl Schedule {
    /// Buckets `times` below `t_max`, skipping `skip`ped units.
    pub fn build(&mut self, times: &[f32], skip: Option<&[bool]>, t_max: usize) {
        let mut by_t: Vec<(u16, u16)> = times
            .iter()
            .enumerate()
            .filter(|&(i, &t)| (t as usize) < t_max && !skip.is_some_and(|s| s[i]))
            .map(|(i, &t)| (t as u16, i as u16))
            .collect();
        by_t.sort_unstable();
        self.units.clear();
        self.units.extend(by_t.iter().map(|&(_, i)| i));
        self.start.clear();
        self.start.resize(t_max + 1, 0);
        for &(t, _) in &by_t {
            self.start[t as usize + 1] += 1;
        }
        for t in 1..self.start.len() {
            self.start[t] += self.start[t - 1];
        }
    }

    /// Units spiking at time `t`.
    pub fn at(&self, t: usize) -> &[u16] {
        if t + 1 >= self.start.len() {
            return &[];
        }
        &self.units[self.start[t] as usize..self.start[t + 1] as usize]
    }
}

/// Spikes of the input, hidden and output layers at one replayed time.
pub struct Spikes<'a> {
    pub input: &'a [u16],
    pub hidden: &'a [u16],
    pub output: &'a [u16],
}

#[derive(Debug, Clone)]
pub struct ImageResult {
    pub info: ImageInfo,
    /// first output to fire, before learning from this image
    pub predicted: Option<u8>,
    pub correct: bool,
    pub running_accuracy: f32,
    /// time of the first output spike
    pub first_spike: f32,
    pub activity: Activity,
}

/// Receives training events. Every method defaults to a no-op.
pub trait Monitor {
    fn image_start(&mut self, _info: &ImageInfo, _pixels: &[u8], _net: &Net) {}
    /// Step `t` replays spike time `t - 1`.
    fn step(&mut self, _t: usize, _net: &Net, _s: &Spikes) {}
    /// Inference has settled; weights are not updated yet.
    fn inferred(&mut self, _net: &Net) {}
    fn image_end(&mut self, _r: &ImageResult, _net: &Net) {}
    fn epoch_end(&mut self, _e: &EpochResult) {}
}

impl Monitor for () {}

impl<A: Monitor, B: Monitor> Monitor for (A, B) {
    fn image_start(&mut self, info: &ImageInfo, pixels: &[u8], net: &Net) {
        self.0.image_start(info, pixels, net);
        self.1.image_start(info, pixels, net);
    }
    fn step(&mut self, t: usize, net: &Net, s: &Spikes) {
        self.0.step(t, net, s);
        self.1.step(t, net, s);
    }
    fn inferred(&mut self, net: &Net) {
        self.0.inferred(net);
        self.1.inferred(net);
    }
    fn image_end(&mut self, r: &ImageResult, net: &Net) {
        self.0.image_end(r, net);
        self.1.image_end(r, net);
    }
    fn epoch_end(&mut self, e: &EpochResult) {
        self.0.epoch_end(e);
        self.1.epoch_end(e);
    }
}

impl Monitor for Printer {
    fn image_end(&mut self, r: &ImageResult, _net: &Net) {
        self.image(&r.info, r.running_accuracy);
    }
    fn epoch_end(&mut self, e: &EpochResult) {
        self.epoch(e);
    }
}

pub struct Trainer {
    pub net: Net,
    /// damage done to test images, for robustness curves
    pub perturb: Option<Perturb>,
    rng: Rng,
    sched: [Schedule; 3],
    recent: Recent,
    pacer: Pacer,
}

impl Trainer {
    pub fn new(p: Params, n_in: usize, classes: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            net: Net::new(n_in, classes, p, &mut rng),
            perturb: None,
            rng,
            sched: Default::default(),
            recent: Recent::new(RECENT),
            pacer: Pacer::new(),
        }
    }

    pub fn running_accuracy(&self) -> f32 {
        self.recent.rate()
    }

    /// Presents one image; learning images run inference and update the weights.
    pub fn present<M: Monitor>(
        &mut self,
        data: &Dataset,
        i: usize,
        info: ImageInfo,
        mon: &mut M,
        ctl: &Control,
    ) -> ImageResult {
        let mut noisy = Vec::new();
        let mut img = data.image(i);
        let mut damage = self.perturb.as_mut().filter(|_| !info.learn);
        if let Some(d) = &mut damage {
            if d.pixels(img, &mut noisy) {
                img = &noisy;
            }
        }
        let net = &mut self.net;
        net.encode(img);
        if let Some(d) = damage {
            d.times(&mut net.t_in, net.p.t_max);
        }
        net.drop_hidden(info.learn.then_some(&mut self.rng));
        let predicted = net.forward();
        let activity = net.activity();
        mon.image_start(&info, img, net);

        // replay the forward pass in time for monitors and pacing
        let steps = net.p.steps();
        self.sched[0].build(&net.t_in, None, steps);
        self.sched[1].build(&net.hid_pred, Some(&net.dropped), steps);
        self.sched[2].build(&net.out_fwd, None, steps);
        for t in 1..=steps {
            let s = Spikes {
                input: self.sched[0].at(t - 1),
                hidden: self.sched[1].at(t - 1),
                output: self.sched[2].at(t - 1),
            };
            mon.step(t, net, &s);
            if info.learn {
                self.pacer.tick(ctl);
            } else {
                ctl.wait_while_paused();
            }
        }

        let correct = predicted == Some(info.label);
        if info.learn {
            net.infer(info.label);
            mon.inferred(net);
            for _ in 0..net.p.iterations * HOLD {
                self.pacer.tick(ctl);
            }
            net.learn();
            net.revive(&mut self.rng);
            self.recent.push(correct);
        }
        let r = ImageResult {
            info,
            activity,
            predicted,
            correct,
            running_accuracy: self.recent.rate(),
            first_spike: net.out_fwd.iter().copied().fold(net.p.t_max, f32::min),
        };
        mon.image_end(&r, net);
        r
    }

    /// One shuffled pass with learning.
    pub fn train_epoch<M: Monitor>(
        &mut self,
        data: &Dataset,
        epoch: usize,
        mon: &mut M,
        ctl: &Control,
    ) -> Pass {
        let mut order: Vec<usize> = (0..data.len()).collect();
        self.rng.shuffle(&mut order);
        let mut pass = Pass::new(self.net.classes);
        for (index, &i) in order.iter().enumerate() {
            if ctl.stopped() {
                break;
            }
            let info = ImageInfo {
                epoch,
                index,
                total: data.len(),
                label: data.label(i),
                learn: true,
            };
            let r = self.present(data, i, info, mon, ctl);
            pass.record(r.info.label, r.predicted, r.activity);
        }
        pass
    }

    /// A pass without learning or dropout.
    pub fn evaluate<M: Monitor>(
        &mut self,
        data: &Dataset,
        epoch: usize,
        mon: &mut M,
        ctl: &Control,
    ) -> Pass {
        let mut pass = Pass::new(self.net.classes);
        for i in 0..data.len() {
            if ctl.stopped() {
                break;
            }
            let info = ImageInfo {
                epoch,
                index: i,
                total: data.len(),
                label: data.label(i),
                learn: false,
            };
            let r = self.present(data, i, info, mon, ctl);
            pass.record(r.info.label, r.predicted, r.activity);
        }
        pass
    }

    /// Trains for `epochs`, evaluating on `test` after each.
    pub fn fit<M: Monitor>(
        &mut self,
        train: &Dataset,
        test: Option<&Dataset>,
        epochs: usize,
        mon: &mut M,
        ctl: &Control,
    ) -> Vec<EpochResult> {
        let mut out = Vec::new();
        for epoch in 0..epochs {
            let t0 = Instant::now();
            let trained = self.train_epoch(train, epoch, mon, ctl);
            let images_per_sec = train.len() as f32 / t0.elapsed().as_secs_f32().max(1e-9);
            let tested = test.map(|t| self.evaluate(t, epoch, mon, ctl));
            if ctl.stopped() {
                break;
            }
            let e = EpochResult {
                epoch,
                train_accuracy: trained.accuracy(),
                test_accuracy: tested.as_ref().map(Pass::accuracy),
                images_per_sec,
                activity: trained.per_image(),
                confusion: tested.map(|p| p.confusion),
            };
            mon.epoch_end(&e);
            out.push(e);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use snn_core::dataset::synthetic;

    #[test]
    fn schedule_buckets_by_time() {
        let mut s = Schedule::default();
        s.build(&[3.0, 0.0, 256.0, 3.0, 1.0], None, 256);
        assert_eq!(s.at(0), &[1]);
        assert_eq!(s.at(3), &[0, 3]);
        assert!(s.at(2).is_empty() && s.at(255).is_empty() && s.at(300).is_empty());
        s.build(&[3.0, 0.0], Some(&[true, false]), 256);
        assert!(s.at(3).is_empty());
    }

    struct Replay(usize, usize);
    impl Monitor for Replay {
        fn step(&mut self, _t: usize, net: &Net, s: &Spikes) {
            self.0 += 1;
            self.1 += s.hidden.len();
            assert!(s.hidden.iter().all(|&j| !net.dropped[j as usize]));
        }
    }

    #[test]
    fn replay_covers_every_step_and_hidden_spike() {
        let data = synthetic(2, 1);
        let mut t = Trainer::new(Params::default(), 784, 10, 1);
        let mut r = Replay(0, 0);
        let info = ImageInfo {
            epoch: 0,
            index: 0,
            total: 2,
            label: data.label(0),
            learn: false,
        };
        t.present(&data, 0, info, &mut r, &Control::new(0));
        let fired = t.net.hid_pred.iter().filter(|&&x| x < 256.0).count();
        assert_eq!((r.0, r.1), (256, fired));
    }

    #[test]
    fn learns_synthetic_digits_above_chance() {
        // synthetic digits light fewer pixels than MNIST, so learning starts slower
        let train = synthetic(3000, 1);
        let test = synthetic(500, 2);
        let mut t = Trainer::new(Params::default(), 784, 10, 7);
        let r = t.fit(&train, Some(&test), 6, &mut (), &Control::new(0));
        let acc = r.last().unwrap().test_accuracy.unwrap();
        assert!(acc > 0.3, "test accuracy {acc}");
        let a = r.last().unwrap().activity;
        assert!(a.spikes > 0.0 && a.synops > a.spikes, "{a:?}");
        let c = r.last().unwrap().confusion.as_ref().unwrap();
        assert_eq!((c.total(), c.classes()), (500, 10));
        assert_eq!(Some(c.accuracy()), r.last().unwrap().test_accuracy);
    }
}
