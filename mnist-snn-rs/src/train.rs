//! Training loop: presents images, labels neurons by the classes they win, tracks accuracy.

use crate::layer::{Layer, Observer, Outcome, Reward, Step};
use crate::params::{Params, Rule};
use snn_core::encode::{receptive_field_offset, SpikeTrain};
use snn_core::exit::{checkpoint, Partial};
use snn_core::{
    Activity, Control, Dataset, EpochResult, Exits, ImageInfo, Pacer, Pass, Perturb, Recent, Rng,
};
use snn_live::app::Printer;
use std::time::Instant;

/// Images in the running-accuracy window.
const RECENT: usize = 500;

/// Per-neuron class counts with forgetting, so labels follow the weights as
/// they change; or a fixed class per neuron, as reward learning needs.
#[derive(Debug, Clone)]
pub struct Labels {
    classes: usize,
    counts: Vec<f32>,
    fixed: bool,
}

impl Labels {
    const DECAY: f32 = 0.99;

    pub fn new(n_out: usize, classes: usize) -> Self {
        Self {
            classes,
            counts: vec![0.0; n_out * classes],
            fixed: false,
        }
    }

    /// Neuron j stands for class `j % classes` whatever it wins.
    pub fn fixed(classes: usize) -> Self {
        Self {
            classes,
            counts: Vec::new(),
            fixed: true,
        }
    }

    pub fn record(&mut self, neuron: usize, class: u8) {
        if self.fixed {
            return;
        }
        let row = &mut self.counts[neuron * self.classes..(neuron + 1) * self.classes];
        row.iter_mut().for_each(|c| *c *= Self::DECAY);
        row[class as usize] += 1.0;
    }

    /// Class the neuron wins most, if it has won anything.
    pub fn of(&self, neuron: usize) -> Option<u8> {
        if self.fixed {
            return Some((neuron % self.classes) as u8);
        }
        let row = &self.counts[neuron * self.classes..(neuron + 1) * self.classes];
        let (best, &n) = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
        (n > 0.0).then_some(best as u8)
    }

    pub fn classes(&self) -> usize {
        self.classes
    }
}

#[derive(Debug, Clone)]
pub struct ImageResult {
    pub info: ImageInfo,
    pub outcome: Outcome,
    /// label of the winner before this image updated the labels
    pub predicted: Option<u8>,
    pub correct: bool,
    pub running_accuracy: f32,
    pub activity: Activity,
}

/// Receives training events. Every method defaults to a no-op.
pub trait Monitor {
    fn image_start(&mut self, _info: &ImageInfo, _pixels: &[u8], _train: &SpikeTrain) {}
    fn step(&mut self, _s: &Step) {}
    fn image_end(&mut self, _r: &ImageResult, _layer: &Layer, _labels: &Labels) {}
    fn epoch_end(&mut self, _e: &EpochResult) {}
}

impl Monitor for () {}

impl<A: Monitor, B: Monitor> Monitor for (A, B) {
    fn image_start(&mut self, info: &ImageInfo, pixels: &[u8], train: &SpikeTrain) {
        self.0.image_start(info, pixels, train);
        self.1.image_start(info, pixels, train);
    }
    fn step(&mut self, s: &Step) {
        self.0.step(s);
        self.1.step(s);
    }
    fn image_end(&mut self, r: &ImageResult, layer: &Layer, labels: &Labels) {
        self.0.image_end(r, layer, labels);
        self.1.image_end(r, layer, labels);
    }
    fn epoch_end(&mut self, e: &EpochResult) {
        self.0.epoch_end(e);
        self.1.epoch_end(e);
    }
}

impl Monitor for Printer {
    fn image_end(&mut self, r: &ImageResult, _layer: &Layer, _labels: &Labels) {
        self.image(&r.info, r.running_accuracy);
    }

    fn epoch_end(&mut self, e: &EpochResult) {
        self.epoch(e);
    }
}

/// Spikes so far after each step and the step the winner appeared at.
#[derive(Debug, Clone, Default)]
struct Track {
    spikes: Vec<u64>,
    decided: Option<usize>,
}

struct Hook<'a, M: Monitor> {
    mon: &'a mut M,
    pacer: &'a mut Pacer,
    ctl: &'a Control,
    learn: bool,
    track: Option<&'a mut Track>,
}

impl<M: Monitor> Observer for Hook<'_, M> {
    fn step(&mut self, s: &Step) {
        self.mon.step(s);
        if let Some(tr) = self.track.as_mut() {
            let before = tr.spikes.last().copied().unwrap_or(0);
            tr.spikes
                .push(before + (s.active.len() + s.fired.len()) as u64);
            if tr.decided.is_none() && s.winner.is_some() {
                tr.decided = Some(s.t);
            }
        }
        // only training is paced; evaluation runs flat out
        if self.learn {
            self.pacer.tick(self.ctl);
        } else {
            self.ctl.wait_while_paused();
        }
    }
}

pub struct Trainer {
    pub params: Params,
    pub layer: Layer,
    pub labels: Labels,
    /// damage done to test images, for robustness curves
    pub perturb: Option<Perturb>,
    /// totals of answers cut short, gathered while testing
    pub exits: Option<Exits>,
    track: Track,
    rng: Rng,
    pot: Vec<f32>,
    train: SpikeTrain,
    recent: Recent,
    pacer: Pacer,
}

impl Trainer {
    pub fn new(params: Params, n_in: usize, n_out: usize, classes: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let layer = Layer::new(n_in, n_out, &params, &mut rng);
        Self::with_layer(params, layer, classes, rng)
    }

    pub fn with_layer(params: Params, layer: Layer, classes: usize, rng: Rng) -> Self {
        let labels = if params.rule.rewarded() {
            Labels::fixed(classes)
        } else {
            Labels::new(layer.n_out, classes)
        };
        Self {
            params,
            layer,
            labels,
            perturb: None,
            exits: None,
            track: Track::default(),
            rng,
            pot: Vec::new(),
            train: SpikeTrain::new(),
            recent: Recent::new(RECENT),
            pacer: Pacer::new(),
        }
    }

    /// Accuracy over the last few hundred training images.
    pub fn running_accuracy(&self) -> f32 {
        self.recent.rate()
    }

    /// Encodes and presents one image. Learning images update weights and labels.
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
        let damage = self.perturb.as_mut().filter(|_| !info.learn);
        if let Some(d) = damage {
            if d.pixels(img, &mut noisy) {
                img = &noisy;
            }
        }
        let p = &self.params;
        receptive_field_offset(img, data.width, data.height, p.kernel_offset, &mut self.pot);
        self.train
            .encode(&self.pot, p.time_steps, &p.encode, &mut self.rng);
        if let Some(d) = self.perturb.as_mut().filter(|_| !info.learn) {
            d.train(&mut self.train);
        }
        mon.image_start(&info, img, &self.train);

        let tracking = self.exits.is_some() && !info.learn;
        self.track.spikes.clear();
        self.track.decided = None;
        let mut hook = Hook {
            mon,
            pacer: &mut self.pacer,
            ctl,
            learn: info.learn,
            track: tracking.then_some(&mut self.track),
        };
        let outcome = if info.learn && p.rule.rewarded() {
            // as in Mozafari et al., rewards scale with the miss rate and
            // punishments with the hit rate, so learning settles as answers improve
            let hit = self.recent.rate();
            let r = Reward {
                label: info.label,
                classes: self.labels.classes(),
                reward: 1.0 - hit,
                punish: hit,
            };
            let out = self.layer.present_rewarded(&self.train, p, r, &mut hook);
            if p.rule == Rule::Delayed {
                self.pacer.ticks(p.dopamine.gap as u64, ctl);
            }
            out
        } else {
            self.layer.present(&self.train, p, info.learn, &mut hook)
        };
        let activity = outcome.activity(self.layer.n_out);
        let predicted = outcome.winner.and_then(|w| self.labels.of(w));
        let correct = predicted == Some(info.label);
        if let Some(e) = self.exits.as_mut().filter(|_| tracking) {
            // the first neuron over threshold answers; nothing after changes it
            let (spikes, decided) = (&self.track.spikes, self.track.decided);
            let at = std::array::from_fn(|k| {
                let step = checkpoint(k, spikes.len());
                let sure = decided.is_some_and(|d| d <= step);
                Partial {
                    answer: predicted.filter(|_| sure),
                    spikes: spikes[step - 1],
                    sure,
                }
            });
            e.record(info.label, &at);
        }
        if info.learn {
            if let Some(w) = outcome.winner {
                self.labels.record(w, info.label);
            }
            self.recent.push(correct);
        }
        let r = ImageResult {
            info,
            outcome,
            predicted,
            correct,
            running_accuracy: self.running_accuracy(),
            activity,
        };
        mon.image_end(&r, &self.layer, &self.labels);
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
        let mut pass = Pass::new(self.labels.classes());
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
        let mut pass = Pass::new(self.labels.classes());
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

    /// Labels neurons from a pass over `data` without touching the weights.
    pub fn assign_labels(&mut self, data: &Dataset) {
        let ctl = Control::new(0);
        for i in 0..data.len() {
            let info = ImageInfo {
                epoch: 0,
                index: i,
                total: data.len(),
                label: data.label(i),
                learn: false,
            };
            if let Some(w) = self.present(data, i, info, &mut (), &ctl).outcome.winner {
                self.labels.record(w, data.label(i));
            }
        }
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
    use snn_core::exit::POINTS;

    #[test]
    fn labels_follow_recent_wins() {
        let mut l = Labels::new(2, 3);
        assert_eq!(l.of(0), None);
        l.record(0, 2);
        assert_eq!(l.of(0), Some(2));
        for _ in 0..3 {
            l.record(0, 1);
        }
        assert_eq!(l.of(0), Some(1));
        assert_eq!(l.of(1), None);
    }

    #[test]
    fn learns_synthetic_digits_above_chance() {
        let train = synthetic(2000, 1);
        let test = synthetic(300, 2);
        let mut t = Trainer::new(Params::default(), 784, 40, 10, 7);
        let ctl = Control::new(0);
        let r = t.fit(&train, Some(&test), 1, &mut (), &ctl);
        let acc = r[0].test_accuracy.unwrap();
        assert!(acc > 0.4, "test accuracy {acc}");
        let a = r[0].activity;
        assert!(a.spikes > 0.0 && a.synops > a.spikes, "{a:?}");
        let c = r[0].confusion.as_ref().unwrap();
        assert_eq!((c.total(), c.classes()), (300, 10));
        assert_eq!(Some(c.accuracy()), r[0].test_accuracy);
    }

    #[test]
    fn learns_synthetic_digits_with_few_bits_per_weight() {
        let train = synthetic(2000, 1);
        let test = synthetic(300, 2);
        for bits in [8, 6] {
            let p = Params {
                weight_bits: bits,
                ..Params::default()
            };
            let mut t = Trainer::new(p, 784, 40, 10, 7);
            let r = t.fit(&train, Some(&test), 1, &mut (), &Control::new(0));
            let acc = r[0].test_accuracy.unwrap();
            assert!(acc > 0.35, "{bits} bits: test accuracy {acc}");
        }
    }

    #[test]
    fn fixed_labels_follow_the_neuron_index() {
        let mut l = Labels::fixed(10);
        l.record(3, 7);
        assert_eq!((l.of(3), l.of(13), l.of(0)), (Some(3), Some(3), Some(0)));
    }

    #[test]
    fn every_rule_learns_synthetic_digits() {
        let train = synthetic(2000, 1);
        let test = synthetic(300, 2);
        for rule in [Rule::Triplet, Rule::Reward] {
            let p = Params {
                rule,
                ..Params::default()
            };
            let mut t = Trainer::new(p, 784, 40, 10, 7);
            let r = t.fit(&train, Some(&test), 1, &mut (), &Control::new(0));
            let acc = r[0].test_accuracy.unwrap();
            assert!(acc > 0.4, "{rule}: test accuracy {acc}");
        }
    }

    #[test]
    fn delayed_reward_needs_trials_apart() {
        // rewards 1-3 s late find the right tags only if the next image waits
        let train = synthetic(2000, 1);
        let test = synthetic(300, 2);
        let acc = |gap| {
            let mut p = Params {
                rule: Rule::Delayed,
                ..Params::default()
            };
            p.dopamine.gap = gap;
            let mut t = Trainer::new(p, 784, 40, 10, 7);
            let r = t.fit(&train, Some(&test), 1, &mut (), &Control::new(0));
            assert_eq!(t.layer.dopamine.images, 2000);
            r[0].test_accuracy.unwrap()
        };
        let (apart, back_to_back) = (acc(2800), acc(0));
        assert!(apart > 0.25, "{apart}");
        assert!(back_to_back < apart - 0.1, "{back_to_back} vs {apart}");
    }

    #[test]
    fn damage_touches_test_images_only() {
        use snn_core::perturb::Kind;
        let train = synthetic(600, 1);
        let test = synthetic(200, 2);
        let mut t = Trainer::new(Params::default(), 784, 20, 10, 7);
        let ctl = Control::new(0);
        t.fit(&train, None, 1, &mut (), &ctl);
        let clean = t.evaluate(&test, 0, &mut (), &ctl).accuracy();
        let mut acc = |kind, level| {
            t.perturb = Some(Perturb::new(kind, level, 3));
            t.evaluate(&test, 0, &mut (), &ctl).accuracy()
        };
        // the threshold follows the input's peak, so only losing everything hurts much
        assert_eq!(acc(Kind::Dropout, 1.0), 0.0);
        assert!(acc(Kind::Noise, 1.0) < clean - 0.1, "{clean}");
        let before = t.layer.weights().to_vec();
        t.perturb = Some(Perturb::new(Kind::Dropout, 0.9, 3));
        let w = t.fit(&train, None, 1, &mut (), &ctl)[0].train_accuracy;
        assert!(w > 0.3, "training images are not damaged: {w}");
        assert_ne!(t.layer.weights(), before.as_slice());
    }

    #[test]
    fn early_answers_cost_fewer_spikes() {
        let train = synthetic(800, 1);
        let test = synthetic(200, 2);
        let mut t = Trainer::new(Params::default(), 784, 20, 10, 7);
        let ctl = Control::new(0);
        t.fit(&train, None, 1, &mut (), &ctl);
        let full = t.evaluate(&test, 0, &mut (), &ctl).accuracy();
        t.exits = Some(Exits::default());
        t.evaluate(&test, 0, &mut (), &ctl);
        let e = t.exits.take().unwrap();
        let acc = e.accuracy();
        assert_eq!(
            acc[POINTS - 1],
            full,
            "the last checkpoint is the full answer"
        );
        assert!(acc[0] < acc[POINTS - 1] && acc.windows(2).all(|w| w[0] <= w[1]));
        let (sure, steps, spikes) = e.sure();
        // a winner never changes its mind, so stopping at it loses nothing
        assert_eq!(sure, full);
        assert!(steps < 0.8 && spikes < steps + 0.1, "{steps} {spikes}");
    }

    #[test]
    fn stop_ends_training_early() {
        let train = synthetic(50, 1);
        let mut t = Trainer::new(Params::default(), 784, 4, 10, 7);
        let ctl = Control::new(0);
        ctl.stop();
        assert!(t.fit(&train, None, 3, &mut (), &ctl).is_empty());
    }
}
