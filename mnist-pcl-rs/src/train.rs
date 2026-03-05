//! Training loop: turns each image into saccade events, runs the network on
//! them, reads the class from its spike counts and tracks accuracy.

use crate::input::Input;
use crate::{Net, Params};
use snn_core::events::{Events, Saccades};
use snn_core::readout::{argmax, Readout};
use snn_core::{Activity, Control, EpochResult, ImageInfo, Pacer, Pass, Recent, Rng};
use snn_live::app::Printer;
use std::time::Instant;

/// Images in the running-accuracy window.
const RECENT: usize = 500;

#[derive(Debug, Clone)]
pub struct ImageResult {
    pub info: ImageInfo,
    /// readout's class before it learned from this image
    pub predicted: Option<u8>,
    pub correct: bool,
    pub running_accuracy: f32,
    pub probs: Vec<f32>,
    /// share of the drive to simple cells that was inhibitory
    pub suppression: f32,
    pub activity: Activity,
}

/// Receives training events. Every method defaults to a no-op.
pub trait Monitor {
    fn image_start(&mut self, _info: &ImageInfo, _pixels: &[u8], _net: &Net) {}
    /// `events` arrived at step `t`; the net shows what they set off.
    fn step(&mut self, _t: usize, _net: &Net, _events: &[u16]) {}
    fn image_end(&mut self, _r: &ImageResult, _net: &Net) {}
    fn epoch_end(&mut self, _e: &EpochResult) {}
}

impl Monitor for () {}

impl<A: Monitor, B: Monitor> Monitor for (A, B) {
    fn image_start(&mut self, info: &ImageInfo, pixels: &[u8], net: &Net) {
        self.0.image_start(info, pixels, net);
        self.1.image_start(info, pixels, net);
    }
    fn step(&mut self, t: usize, net: &Net, events: &[u16]) {
        self.0.step(t, net, events);
        self.1.step(t, net, events);
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
    pub readout: Readout,
    pub saccades: Saccades,
    rng: Rng,
    events: Events,
    picture: Vec<u8>,
    code: Vec<f32>,
    probs: Vec<f32>,
    recent: Recent,
    pacer: Pacer,
}

impl Trainer {
    /// For images of `width` by `height`, which saccades turn into events.
    pub fn new(p: Params, width: usize, height: usize, classes: usize, seed: u64) -> Self {
        let side = width.max(height) + 2 * Saccades::default().pad;
        Self::with_retina(p, side, classes, seed)
    }

    /// For a retina of `side` by `side` pixels.
    pub fn with_retina(p: Params, side: usize, classes: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let saccades = Saccades::default();
        let net = Net::new(side, p.clone(), &mut rng);
        let readout = Readout::new(
            net.simple.len() + net.complex.len(),
            classes,
            p.readout_rate,
        );
        Self {
            net,
            readout,
            saccades,
            rng,
            events: Events::new(),
            picture: Vec::new(),
            code: Vec::new(),
            probs: Vec::new(),
            recent: Recent::new(RECENT),
            pacer: Pacer::new(),
        }
    }

    pub fn running_accuracy(&self) -> f32 {
        self.recent.rate()
    }

    /// Presents one sample; learning ones update the network and the readout.
    pub fn present<M: Monitor, D: Input + ?Sized>(
        &mut self,
        data: &D,
        i: usize,
        info: ImageInfo,
        mon: &mut M,
        ctl: &Control,
    ) -> ImageResult {
        data.load(i, &self.saccades, &mut self.events, &mut self.picture);
        self.net.reset();
        mon.image_start(&info, &self.picture, &self.net);
        let (mut exc, mut inh) = (0.0, 0.0);
        let mut events = 0u64;
        for t in 1..=self.events.steps() {
            let ev = self.events.at(t);
            events += ev.len() as u64;
            self.net.step(t as u32, ev, info.learn);
            exc += self.net.excitation;
            inh += self.net.inhibition;
            mon.step(t, &self.net, ev);
            if info.learn {
                self.pacer.tick(ctl);
            } else {
                ctl.wait_while_paused();
            }
        }
        let activity = self.net.activity(events);
        self.net.code(&mut self.code);
        self.readout.probs(&self.code, &mut self.probs);
        let predicted = argmax(&self.probs);
        let correct = predicted == Some(info.label);
        if info.learn {
            self.readout.learn(&self.code, &self.probs, info.label);
            self.recent.push(correct);
        }
        let r = ImageResult {
            info,
            activity,
            predicted,
            correct,
            running_accuracy: self.recent.rate(),
            probs: self.probs.clone(),
            suppression: inh / (exc + inh).max(f32::MIN_POSITIVE),
        };
        mon.image_end(&r, &self.net);
        r
    }

    /// One shuffled pass with learning.
    pub fn train_epoch<M: Monitor, D: Input + ?Sized>(
        &mut self,
        data: &D,
        epoch: usize,
        mon: &mut M,
        ctl: &Control,
    ) -> Pass {
        let mut order: Vec<usize> = (0..data.len()).collect();
        self.rng.shuffle(&mut order);
        let mut pass = Pass::new(self.readout.classes());
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

    /// A pass without learning.
    pub fn evaluate<M: Monitor, D: Input + ?Sized>(
        &mut self,
        data: &D,
        epoch: usize,
        mon: &mut M,
        ctl: &Control,
    ) -> Pass {
        let mut pass = Pass::new(self.readout.classes());
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
    pub fn fit<M: Monitor, D: Input + ?Sized>(
        &mut self,
        train: &D,
        test: Option<&D>,
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

    struct Count(usize, usize);
    impl Monitor for Count {
        fn step(&mut self, _t: usize, _net: &Net, events: &[u16]) {
            self.0 += 1;
            self.1 += events.len();
        }
    }

    #[test]
    fn every_step_and_event_is_reported() {
        let data = synthetic(2, 1);
        let mut t = Trainer::new(Params::default(), 28, 28, 10, 1);
        let mut c = Count(0, 0);
        t.fit(&data, None, 1, &mut c, &Control::new(0));
        assert_eq!(c.0, 2 * t.saccades.steps());
        assert!(c.1 > 100);
    }

    #[test]
    fn learns_synthetic_digits_above_chance() {
        let train = synthetic(600, 1);
        let test = synthetic(200, 2);
        let mut t = Trainer::new(Params::default(), 28, 28, 10, 7);
        let r = t.fit(&train, Some(&test), 1, &mut (), &Control::new(0));
        let acc = r[0].test_accuracy.unwrap();
        assert!(acc > 0.3, "test accuracy {acc}");
        let a = r[0].activity;
        assert!(a.spikes > 0.0 && a.synops > a.spikes, "{a:?}");
        let c = r[0].confusion.as_ref().unwrap();
        assert_eq!((c.total(), c.classes()), (200, 10));
        assert_eq!(Some(c.accuracy()), r[0].test_accuracy);
    }
}
