//! Training loop: presents images, labels neurons by the classes they win, tracks accuracy.

use crate::dataset::Dataset;
use crate::encode::{receptive_field, SpikeTrain};
use crate::layer::{Layer, Observer, Outcome, Step};
use crate::params::Params;
use crate::rng::Rng;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// Images in the running-accuracy window.
const RECENT: usize = 500;

/// Per-neuron class counts with forgetting, so labels follow the weights as they change.
#[derive(Debug, Clone)]
pub struct Labels {
    classes: usize,
    counts: Vec<f32>,
}

impl Labels {
    const DECAY: f32 = 0.99;

    pub fn new(n_out: usize, classes: usize) -> Self {
        Self {
            classes,
            counts: vec![0.0; n_out * classes],
        }
    }

    pub fn record(&mut self, neuron: usize, class: u8) {
        let row = &mut self.counts[neuron * self.classes..(neuron + 1) * self.classes];
        row.iter_mut().for_each(|c| *c *= Self::DECAY);
        row[class as usize] += 1.0;
    }

    /// Class the neuron wins most, if it has won anything.
    pub fn of(&self, neuron: usize) -> Option<u8> {
        let row = &self.counts[neuron * self.classes..(neuron + 1) * self.classes];
        let (best, &n) = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
        (n > 0.0).then_some(best as u8)
    }

    pub fn classes(&self) -> usize {
        self.classes
    }
}

/// Pause, speed and stop requests shared with other threads.
#[derive(Debug)]
pub struct Control {
    paused: AtomicBool,
    lock: Mutex<()>,
    wake: Condvar,
    /// simulation steps per second, 0 for unthrottled
    speed: AtomicU32,
    stop: AtomicBool,
    done: AtomicBool,
}

impl Control {
    pub fn new(speed: u32) -> Self {
        Self {
            paused: AtomicBool::new(false),
            lock: Mutex::new(()),
            wake: Condvar::new(),
            speed: AtomicU32::new(speed),
            stop: AtomicBool::new(false),
            done: AtomicBool::new(false),
        }
    }

    pub fn set_paused(&self, paused: bool) {
        let _g = self.lock.lock().unwrap();
        self.paused.store(paused, Ordering::Relaxed);
        self.wake.notify_all();
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    pub fn set_speed(&self, steps_per_sec: u32) {
        self.speed.store(steps_per_sec, Ordering::Relaxed);
    }

    pub fn speed(&self) -> u32 {
        self.speed.load(Ordering::Relaxed)
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
        self.set_paused(false);
    }

    pub fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Marks training as finished.
    pub fn finish(&self) {
        self.done.store(true, Ordering::Relaxed);
    }

    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Relaxed)
    }

    /// Blocks while paused; returns how long it waited.
    pub fn wait_while_paused(&self) -> Duration {
        if !self.is_paused() {
            return Duration::ZERO;
        }
        let t0 = Instant::now();
        let mut g = self.lock.lock().unwrap();
        while self.is_paused() && !self.stopped() {
            g = self.wake.wait(g).unwrap();
        }
        t0.elapsed()
    }
}

/// Holds the simulation to the control's step rate.
struct Pacer {
    origin: Instant,
    steps: u64,
    rate: u32,
}

impl Pacer {
    fn new() -> Self {
        Self {
            origin: Instant::now(),
            steps: 0,
            rate: 0,
        }
    }

    fn tick(&mut self, ctl: &Control) {
        if ctl.wait_while_paused() > Duration::ZERO || ctl.speed() != self.rate {
            *self = Self {
                rate: ctl.speed(),
                ..Self::new()
            };
        }
        if self.rate == 0 {
            return;
        }
        self.steps += 1;
        let due = self.origin + Duration::from_secs_f64(self.steps as f64 / self.rate as f64);
        let now = Instant::now();
        // sleep in chunks of at least 2 ms; coarse OS timers would overshoot finer waits
        if due > now + Duration::from_millis(2) {
            std::thread::sleep(due - now);
        } else if now > due + Duration::from_millis(250) {
            // fell far behind (e.g. a slow frame); don't sprint to catch up
            *self = Self {
                rate: self.rate,
                ..Self::new()
            };
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImageInfo {
    pub epoch: usize,
    /// position within the epoch
    pub index: usize,
    pub total: usize,
    pub label: u8,
    pub learn: bool,
}

#[derive(Debug, Clone)]
pub struct ImageResult {
    pub info: ImageInfo,
    pub outcome: Outcome,
    /// label of the winner before this image updated the labels
    pub predicted: Option<u8>,
    pub correct: bool,
    pub running_accuracy: f32,
}

#[derive(Debug, Clone)]
pub struct EpochResult {
    pub epoch: usize,
    pub train_accuracy: f32,
    pub test_accuracy: Option<f32>,
    pub images_per_sec: f32,
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

struct Hook<'a, M: Monitor> {
    mon: &'a mut M,
    pacer: &'a mut Pacer,
    ctl: &'a Control,
    learn: bool,
}

impl<M: Monitor> Observer for Hook<'_, M> {
    fn step(&mut self, s: &Step) {
        self.mon.step(s);
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
    rng: Rng,
    pot: Vec<f32>,
    train: SpikeTrain,
    recent: VecDeque<bool>,
    recent_hits: usize,
    pacer: Pacer,
}

impl Trainer {
    pub fn new(params: Params, n_in: usize, n_out: usize, classes: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let layer = Layer::new(n_in, n_out, &params, &mut rng);
        Self::with_layer(params, layer, classes, rng)
    }

    pub fn with_layer(params: Params, layer: Layer, classes: usize, rng: Rng) -> Self {
        let labels = Labels::new(layer.n_out, classes);
        Self {
            params,
            layer,
            labels,
            rng,
            pot: Vec::new(),
            train: SpikeTrain::new(),
            recent: VecDeque::with_capacity(RECENT),
            recent_hits: 0,
            pacer: Pacer::new(),
        }
    }

    /// Accuracy over the last few hundred training images.
    pub fn running_accuracy(&self) -> f32 {
        self.recent_hits as f32 / self.recent.len().max(1) as f32
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
        let img = data.image(i);
        receptive_field(img, data.width, data.height, &mut self.pot);
        let p = &self.params;
        self.train
            .encode(&self.pot, p.time_steps, &p.encode, &mut self.rng);
        mon.image_start(&info, img, &self.train);

        let mut hook = Hook {
            mon,
            pacer: &mut self.pacer,
            ctl,
            learn: info.learn,
        };
        let outcome = self.layer.present(&self.train, p, info.learn, &mut hook);
        let predicted = outcome.winner.and_then(|w| self.labels.of(w));
        let correct = predicted == Some(info.label);
        if info.learn {
            if let Some(w) = outcome.winner {
                self.labels.record(w, info.label);
            }
            if self.recent.len() == RECENT {
                self.recent_hits -= self.recent.pop_front().unwrap() as usize;
            }
            self.recent.push_back(correct);
            self.recent_hits += correct as usize;
        }
        let r = ImageResult {
            info,
            outcome,
            predicted,
            correct,
            running_accuracy: self.running_accuracy(),
        };
        mon.image_end(&r, &self.layer, &self.labels);
        r
    }

    /// One shuffled pass with learning; returns the epoch's accuracy.
    pub fn train_epoch<M: Monitor>(
        &mut self,
        data: &Dataset,
        epoch: usize,
        mon: &mut M,
        ctl: &Control,
    ) -> f32 {
        let mut order: Vec<usize> = (0..data.len()).collect();
        self.rng.shuffle(&mut order);
        let mut hits = 0;
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
            hits += self.present(data, i, info, mon, ctl).correct as usize;
        }
        hits as f32 / data.len().max(1) as f32
    }

    /// Accuracy without learning.
    pub fn evaluate<M: Monitor>(
        &mut self,
        data: &Dataset,
        epoch: usize,
        mon: &mut M,
        ctl: &Control,
    ) -> f32 {
        let mut hits = 0;
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
            hits += self.present(data, i, info, mon, ctl).correct as usize;
        }
        hits as f32 / data.len().max(1) as f32
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
            let train_accuracy = self.train_epoch(train, epoch, mon, ctl);
            let images_per_sec = train.len() as f32 / t0.elapsed().as_secs_f32().max(1e-9);
            let test_accuracy = test.map(|t| self.evaluate(t, epoch, mon, ctl));
            if ctl.stopped() {
                break;
            }
            let e = EpochResult {
                epoch,
                train_accuracy,
                test_accuracy,
                images_per_sec,
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
    use crate::dataset::synthetic;

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
    fn pacing_holds_the_step_rate() {
        let ctl = Control::new(2000);
        let mut p = Pacer::new();
        let t0 = Instant::now();
        for _ in 0..200 {
            p.tick(&ctl);
        }
        let dt = t0.elapsed().as_secs_f32();
        assert!((0.08..0.3).contains(&dt), "200 steps at 2000/s took {dt}s");
    }

    #[test]
    fn pause_blocks_until_resumed() {
        let ctl = std::sync::Arc::new(Control::new(0));
        ctl.set_paused(true);
        let c = ctl.clone();
        let h = std::thread::spawn(move || c.wait_while_paused());
        std::thread::sleep(Duration::from_millis(30));
        ctl.set_paused(false);
        assert!(h.join().unwrap() >= Duration::from_millis(20));
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
