//! Training loop, layer by layer as in the paper: STDP on L1, then on L2, then
//! R-STDP on the decision layer, each for its own epochs. Each image is coded,
//! run forward, replayed step by step for monitors, then learned from.

use crate::params::Phase;
use crate::wave::NEVER;
use crate::{Net, Params};
use snn_core::exit::{checkpoint, Partial, POINTS};
use snn_core::{
    Activity, Control, Dataset, EpochResult, Exits, ImageInfo, Pacer, Pass, Perturb, Recent, Rng,
};
use snn_live::app::Printer;
use std::time::Instant;

/// Images in the running-accuracy window.
const RECENT: usize = 500;

/// Spikes of S1, L1 and L2 at one replayed step.
pub struct Spikes<'a> {
    pub input: &'a [u16],
    pub l1: &'a [u16],
    pub l2: &'a [u16],
}

#[derive(Debug, Clone)]
pub struct ImageResult {
    pub info: ImageInfo,
    pub phase: Phase,
    /// the decision layer's answer before learning; none while features learn
    pub predicted: Option<u8>,
    pub correct: bool,
    /// NaN until the decision layer trains
    pub running_accuracy: f32,
    pub activity: Activity,
}

/// Receives training events. Every method defaults to a no-op.
pub trait Monitor {
    fn phase_start(&mut self, _phase: Phase, _net: &Net) {}
    fn image_start(&mut self, _info: &ImageInfo, _phase: Phase, _pixels: &[u8], _net: &Net) {}
    /// Step `t` replays time `t - 1`.
    fn step(&mut self, _t: usize, _net: &Net, _s: &Spikes) {}
    fn image_end(&mut self, _r: &ImageResult, _net: &Net) {}
    fn epoch_end(&mut self, _e: &EpochResult, _net: &Net) {}
}

impl Monitor for () {}

impl<A: Monitor, B: Monitor> Monitor for (A, B) {
    fn phase_start(&mut self, phase: Phase, net: &Net) {
        self.0.phase_start(phase, net);
        self.1.phase_start(phase, net);
    }
    fn image_start(&mut self, info: &ImageInfo, phase: Phase, pixels: &[u8], net: &Net) {
        self.0.image_start(info, phase, pixels, net);
        self.1.image_start(info, phase, pixels, net);
    }
    fn step(&mut self, t: usize, net: &Net, s: &Spikes) {
        self.0.step(t, net, s);
        self.1.step(t, net, s);
    }
    fn image_end(&mut self, r: &ImageResult, net: &Net) {
        self.0.image_end(r, net);
        self.1.image_end(r, net);
    }
    fn epoch_end(&mut self, e: &EpochResult, net: &Net) {
        self.0.epoch_end(e, net);
        self.1.epoch_end(e, net);
    }
}

impl Monitor for Printer {
    fn phase_start(&mut self, phase: Phase, net: &Net) {
        let p = &net.p;
        let line = |l: &crate::params::Layer| {
            format!(
                "{} maps of {k}x{k}, {} epochs",
                l.maps,
                l.epochs,
                k = l.kernel
            )
        };
        match phase {
            Phase::Layer1 => println!("layer 1: STDP on {}", line(&p.l1)),
            Phase::Layer2 => println!("layer 2: STDP on {}", line(&p.l2)),
            Phase::Decision => println!(
                "decision: R-STDP on {} maps, {} per class",
                net.l3.output.maps, p.per_class
            ),
        }
    }
    fn image_end(&mut self, r: &ImageResult, _net: &Net) {
        self.image(&r.info, r.running_accuracy);
    }
    fn epoch_end(&mut self, e: &EpochResult, net: &Net) {
        self.epoch(e);
        let c = [&net.l1, &net.l2, &net.l3].map(|l| l.convergence());
        println!(
            "  convergence L1 {:.4}  L2 {:.4}  L3 {:.4}",
            c[0], c[1], c[2]
        );
    }
}

pub struct Trainer {
    pub net: Net,
    /// damage done to test images, for robustness curves
    pub perturb: Option<Perturb>,
    /// totals of answers cut short, gathered while testing
    pub exits: Option<Exits>,
    rng: Rng,
    recent: Recent,
    pacer: Pacer,
    noisy: Vec<u8>,
    times: Vec<f32>,
    /// phase of the last image trained on
    phase: Option<Phase>,
}

impl Trainer {
    pub fn new(
        p: Params,
        width: usize,
        height: usize,
        classes: usize,
        seed: u64,
    ) -> Result<Self, String> {
        let mut rng = Rng::new(seed);
        Ok(Self {
            net: Net::new(width, height, classes, p, &mut rng)?,
            perturb: None,
            exits: None,
            rng,
            recent: Recent::new(RECENT),
            pacer: Pacer::new(),
            noisy: Vec::new(),
            times: Vec::new(),
            phase: None,
        })
    }

    /// Epochs in all: the feature layers' and then `decision` of R-STDP.
    pub fn epochs(&self, decision: usize) -> usize {
        self.net.p.feature_epochs() + decision
    }

    pub fn running_accuracy(&self) -> f32 {
        self.recent.rate()
    }

    /// Presents one image; training images learn in the phase of their epoch.
    pub fn present<M: Monitor>(
        &mut self,
        data: &Dataset,
        i: usize,
        info: ImageInfo,
        mon: &mut M,
        ctl: &Control,
    ) -> ImageResult {
        let phase = if info.learn {
            self.net.p.phase(info.epoch)
        } else {
            Phase::Decision
        };
        let mut img = data.image(i);
        let mut damage = self.perturb.as_mut().filter(|_| !info.learn);
        if let Some(d) = &mut damage {
            if d.pixels(img, &mut self.noisy) {
                img = &self.noisy;
            }
        }
        let net = &mut self.net;
        net.encode(img);
        let steps = net.p.steps;
        if let Some(d) = damage {
            let never = steps as f32;
            let time = &mut net.input.time;
            self.times.clear();
            self.times.extend(
                time.iter()
                    .map(|&t| if t == NEVER { never } else { t as f32 }),
            );
            d.times(&mut self.times, never);
            for (t, &v) in time.iter_mut().zip(&self.times) {
                *t = if v >= never { NEVER } else { v as u8 };
            }
            net.input.index(steps);
        }
        let predicted = net.forward(phase, info.learn);
        let activity = net.activity();
        mon.image_start(&info, phase, img, net);

        // replay the pass in time for monitors and pacing
        for t in 1..=steps {
            let s = Spikes {
                input: net.input.at(t - 1),
                l1: net.fire1.out.at(t - 1),
                l2: if net.depth >= 2 {
                    net.fire2.out.at(t - 1)
                } else {
                    &[]
                },
            };
            mon.step(t, net, &s);
            if info.learn {
                self.pacer.tick(ctl);
            } else {
                ctl.wait_while_paused();
            }
        }

        let correct = predicted == Some(info.label);
        if let Some(e) = self.exits.as_mut().filter(|_| !info.learn) {
            // the leading decision neuron answers; nothing fires before the end
            let at = std::array::from_fn(|k| {
                let end = checkpoint(k, steps) - 1;
                let spikes: usize = (0..=end)
                    .map(|t| {
                        net.input.at(t).len()
                            + net.fire1.out.at(t).len()
                            + net.fire2.out.at(t).len()
                    })
                    .sum();
                Partial {
                    answer: net
                        .leader
                        .get(end)
                        .copied()
                        .flatten()
                        .map(|u| net.class_of(u)),
                    spikes: spikes as u64,
                    sure: false,
                }
            });
            let at: [Partial; POINTS] = at;
            e.record(info.label, &at);
        }
        if info.learn {
            net.learn(phase, info.label, predicted);
            if phase == Phase::Decision {
                self.recent.push(correct);
            }
        }
        let decided = phase == Phase::Decision;
        let r = ImageResult {
            info,
            phase,
            predicted,
            correct,
            running_accuracy: if decided {
                self.recent.rate()
            } else {
                f32::NAN
            },
            activity,
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
        let phase = self.net.p.phase(epoch);
        if self.phase != Some(phase) {
            mon.phase_start(phase, &self.net);
            self.phase = Some(phase);
        }
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

    /// A pass of the whole network without learning.
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

    /// Trains the feature layers, then `epochs` of the decision layer,
    /// evaluating on `test` after each of those.
    pub fn fit<M: Monitor>(
        &mut self,
        train: &Dataset,
        test: Option<&Dataset>,
        epochs: usize,
        mon: &mut M,
        ctl: &Control,
    ) -> Vec<EpochResult> {
        let mut out = Vec::new();
        for epoch in 0..self.epochs(epochs) {
            let t0 = Instant::now();
            let trained = self.train_epoch(train, epoch, mon, ctl);
            let images_per_sec = train.len() as f32 / t0.elapsed().as_secs_f32().max(1e-9);
            let decided = self.net.p.phase(epoch) == Phase::Decision;
            let tested = test
                .filter(|_| decided)
                .map(|t| self.evaluate(t, epoch, mon, ctl));
            if ctl.stopped() {
                break;
            }
            let e = EpochResult {
                epoch,
                train_accuracy: if decided {
                    trained.accuracy()
                } else {
                    f32::NAN
                },
                test_accuracy: tested.as_ref().map(Pass::accuracy),
                images_per_sec,
                activity: trained.per_image(),
                confusion: tested.map(|p| p.confusion),
            };
            mon.epoch_end(&e, &self.net);
            out.push(e);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use snn_core::dataset::synthetic;

    fn small() -> Params {
        let mut p = Params::default();
        p.l1.epochs = 1;
        p.l2.epochs = 1;
        p.l2.maps = 64;
        p.per_class = 10;
        p.rate_every = 100;
        p.batch = 100;
        p
    }

    struct Replay(usize, usize, Vec<Phase>);
    impl Monitor for Replay {
        fn phase_start(&mut self, phase: Phase, _net: &Net) {
            self.2.push(phase);
        }
        fn step(&mut self, _t: usize, net: &Net, s: &Spikes) {
            self.0 += 1;
            self.1 += s.l1.len();
            assert!(s.l2.is_empty() || net.depth >= 2);
        }
    }

    #[test]
    fn replay_covers_every_step_and_spike() {
        let data = synthetic(2, 1);
        let mut t = Trainer::new(small(), 28, 28, 10, 1).unwrap();
        let mut r = Replay(0, 0, Vec::new());
        let info = ImageInfo {
            epoch: 0,
            index: 0,
            total: 2,
            label: data.label(0),
            learn: false,
        };
        let res = t.present(&data, 0, info, &mut r, &Control::new(0));
        assert_eq!((r.0, r.1), (15, t.net.fire1.out.spikes()));
        assert!(res.predicted.is_some() && !res.running_accuracy.is_nan());
    }

    #[test]
    fn trains_layer_by_layer_then_decides_above_chance() {
        // features need a couple of thousand images to settle
        let train = synthetic(2000, 1);
        let test = synthetic(200, 2);
        let mut t = Trainer::new(small(), 28, 28, 10, 7).unwrap();
        let mut r = Replay(0, 0, Vec::new());
        let res = t.fit(&train, Some(&test), 3, &mut r, &Control::new(0));
        assert_eq!(r.2, [Phase::Layer1, Phase::Layer2, Phase::Decision]);
        assert_eq!(res.len(), 5);
        assert!(res[0].train_accuracy.is_nan() && res[1].test_accuracy.is_none());
        let acc = res[4].test_accuracy.unwrap();
        assert!(acc > 0.5, "test accuracy {acc}");
        let a = res[4].activity;
        assert!(a.spikes > 0.0 && a.synops > a.spikes, "{a:?}");
        let c = res[4].confusion.as_ref().unwrap();
        assert_eq!((c.total(), c.classes()), (200, 10));
    }

    #[test]
    fn early_exit_ends_on_the_answer() {
        let data = synthetic(20, 3);
        let mut t = Trainer::new(small(), 28, 28, 10, 1).unwrap();
        t.exits = Some(Exits::default());
        let pass = t.evaluate(&data, 0, &mut (), &Control::new(0));
        let e = t.exits.take().unwrap();
        assert_eq!(e.accuracy()[POINTS - 1], pass.accuracy());
        assert_eq!(e.spent()[POINTS - 1], 1.0);
    }
}
