//! Training loop: Poisson-codes each image, runs the network, reads the class
//! from the spike code and tracks accuracy.

use crate::{Net, Params};
use snn_core::encode::{Coding, EncodeParams};
use snn_core::exit::{checkpoint, Partial, POINTS, SURE};
use snn_core::readout::{argmax, Readout};
use snn_core::{
    Activity, Control, Dataset, EpochResult, Exits, ImageInfo, Pacer, Pass, Perturb, Recent, Rng,
    SpikeTrain,
};
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
    /// share of the input the predictions left unexplained, over the presentation
    pub unexplained: f32,
    pub activity: Activity,
}

/// Receives training events. Every method defaults to a no-op.
pub trait Monitor {
    fn image_start(&mut self, _info: &ImageInfo, _pixels: &[u8], _net: &Net) {}
    fn step(&mut self, _t: usize, _net: &Net) {}
    fn image_end(&mut self, _r: &ImageResult, _net: &Net, _readout: &Readout) {}
    fn epoch_end(&mut self, _e: &EpochResult) {}
}

impl Monitor for () {}

impl<A: Monitor, B: Monitor> Monitor for (A, B) {
    fn image_start(&mut self, info: &ImageInfo, pixels: &[u8], net: &Net) {
        self.0.image_start(info, pixels, net);
        self.1.image_start(info, pixels, net);
    }
    fn step(&mut self, t: usize, net: &Net) {
        self.0.step(t, net);
        self.1.step(t, net);
    }
    fn image_end(&mut self, r: &ImageResult, net: &Net, readout: &Readout) {
        self.0.image_end(r, net, readout);
        self.1.image_end(r, net, readout);
    }
    fn epoch_end(&mut self, e: &EpochResult) {
        self.0.epoch_end(e);
        self.1.epoch_end(e);
    }
}

impl Monitor for Printer {
    fn image_end(&mut self, r: &ImageResult, _net: &Net, _readout: &Readout) {
        self.image(&r.info, r.running_accuracy);
    }
    fn epoch_end(&mut self, e: &EpochResult) {
        self.epoch(e);
    }
}

pub struct Trainer {
    pub net: Net,
    pub readout: Readout,
    /// damage done to test images, for robustness curves
    pub perturb: Option<Perturb>,
    /// totals of answers cut short, gathered while testing
    pub exits: Option<Exits>,
    rng: Rng,
    encode: EncodeParams,
    pot: Vec<f32>,
    train: SpikeTrain,
    code: Vec<f32>,
    probs: Vec<f32>,
    recent: Recent,
    pacer: Pacer,
}

impl Trainer {
    pub fn new(p: Params, n_in: usize, classes: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let net = Net::new(n_in, p.clone(), &mut rng);
        let readout = Readout::new(p.layers.iter().sum(), classes, p.readout_rate);
        // pixel brightness sets a Poisson rate up to max_rate
        let encode = EncodeParams {
            coding: Coding::Poisson,
            pot_min: 0.0,
            pot_max: 1.0,
            freq_min: 0.0,
            freq_max: p.max_rate,
            period_scale: 1000.0 / p.dt,
        };
        Self {
            net,
            readout,
            perturb: None,
            exits: None,
            rng,
            encode,
            pot: Vec::new(),
            train: SpikeTrain::new(),
            code: Vec::new(),
            probs: Vec::new(),
            recent: Recent::new(RECENT),
            pacer: Pacer::new(),
        }
    }

    /// Accuracy over the last few hundred training images.
    pub fn running_accuracy(&self) -> f32 {
        self.recent.rate()
    }

    // the softmax readout weighs every unit's count once per image
    fn readout_macs(&self) -> u64 {
        let units: usize = self.net.p.layers.iter().sum();
        (units * self.readout.classes()) as u64
    }

    /// MACs per image of a dense ANN with the same layers and readout.
    pub fn dense_macs(&self) -> u64 {
        self.net.dense_macs() + self.readout_macs()
    }

    /// Presents one image; learning images update the network and the readout.
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
        if let Some(d) = self.perturb.as_mut().filter(|_| !info.learn) {
            if d.pixels(img, &mut noisy) {
                img = &noisy;
            }
        }
        self.pot.clear();
        self.pot.extend(img.iter().map(|&v| v as f32 / 255.0));
        let steps = self.net.p.steps();
        self.train
            .encode(&self.pot, steps, &self.encode, &mut self.rng);
        if let Some(d) = self.perturb.as_mut().filter(|_| !info.learn) {
            d.train(&mut self.train);
        }
        self.net.reset();
        mon.image_start(&info, img, &self.net);
        let (mut err, mut input) = (0.0, 0.0);
        let mut inputs = 0u64;
        let tracking = self.exits.is_some() && !info.learn;
        let (mut at, mut next, mut spent) = ([Partial::default(); POINTS], 0, 0u64);
        for t in 1..=steps {
            self.net.step(t, &self.train, info.learn);
            inputs += self.net.input.len() as u64;
            if tracking {
                let latent: usize = self.net.lat.iter().map(|l| l.spikes.len()).sum();
                spent += (self.net.input.len() + latent) as u64;
                // the readout of the counts so far
                while next < POINTS && checkpoint(next, steps) == t {
                    self.net.code(&mut self.code);
                    self.readout.probs(&self.code, &mut self.probs);
                    let top = self.probs.iter().copied().fold(0.0, f32::max);
                    at[next] = Partial {
                        answer: argmax(&self.probs),
                        spikes: spent,
                        sure: top >= SURE,
                    };
                    next += 1;
                }
            }
            err += self.net.energy[0];
            input += self.net.input_energy;
            mon.step(t, &self.net);
            // only training is paced; evaluation runs flat out
            if info.learn {
                self.pacer.tick(ctl);
            } else {
                ctl.wait_while_paused();
            }
        }
        self.net.end(info.learn);
        let mut activity = self.net.activity(inputs);
        activity.macs += self.readout_macs();

        self.net.code(&mut self.code);
        self.readout.probs(&self.code, &mut self.probs);
        let predicted = argmax(&self.probs);
        let correct = predicted == Some(info.label);
        if let Some(e) = self.exits.as_mut().filter(|_| tracking) {
            e.record(info.label, &at);
        }
        if info.learn {
            self.readout.learn(&self.code, &self.probs, info.label);
            self.recent.push(correct);
        }
        let r = ImageResult {
            info,
            activity,
            predicted,
            correct,
            running_accuracy: self.running_accuracy(),
            probs: self.probs.clone(),
            unexplained: if input > 0.0 { err / input } else { 0.0 },
        };
        mon.image_end(&r, &self.net, &self.readout);
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
    pub fn evaluate<M: Monitor>(
        &mut self,
        data: &Dataset,
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

    struct Count(usize, usize);
    impl Monitor for Count {
        fn step(&mut self, _t: usize, _net: &Net) {
            self.0 += 1;
        }
        fn image_end(&mut self, _r: &ImageResult, _net: &Net, _ro: &Readout) {
            self.1 += 1;
        }
    }

    #[test]
    fn every_step_and_image_is_reported() {
        let data = synthetic(3, 1);
        let p = Params {
            layers: vec![30],
            ..Params::default()
        };
        let steps = p.steps();
        let mut t = Trainer::new(p, 784, 10, 1);
        let mut c = Count(0, 0);
        t.fit(&data, None, 1, &mut c, &Control::new(0));
        assert_eq!((c.0, c.1), (3 * steps, 3));
    }

    #[test]
    fn learns_synthetic_digits_above_chance() {
        let train = synthetic(1500, 1);
        let test = synthetic(300, 2);
        let p = Params {
            layers: vec![100],
            ..Params::default()
        };
        let mut t = Trainer::new(p, 784, 10, 7);
        let r = t.fit(&train, Some(&test), 1, &mut (), &Control::new(0));
        let acc = r[0].test_accuracy.unwrap();
        assert!(acc > 0.4, "test accuracy {acc}");
        let a = r[0].activity;
        // input spikes are the only spike deliveries, graded errors the bulk
        assert!(a.spikes > a.synops && a.macs > 100.0 * a.synops, "{a:?}");
        assert!(a.macs as u64 > t.dense_macs(), "{a:?}");
        let c = r[0].confusion.as_ref().unwrap();
        assert_eq!((c.total(), c.classes()), (300, 10));
        assert_eq!(Some(c.accuracy()), r[0].test_accuracy);
    }

    #[test]
    fn stop_ends_training_early() {
        let data = synthetic(20, 1);
        let mut t = Trainer::new(Params::default(), 784, 10, 7);
        let ctl = Control::new(0);
        ctl.stop();
        assert!(t.fit(&data, None, 2, &mut (), &ctl).is_empty());
    }
}
