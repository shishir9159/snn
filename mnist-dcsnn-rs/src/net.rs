//! The deep convolutional network of Mozafari et al. (2019): rank-order input
//! from DoG filters, two convolutional layers that learn features by STDP one
//! after the other, each followed by max pooling, and a decision layer whose
//! maps belong to classes and learn by reward-modulated STDP. The largest final
//! potential in the decision layer answers.

use crate::conv::{pointwise_inhibition, winners, Conv, Fire, Winner};
use crate::input::{DOG, S1};
use crate::params::{Params, Phase};
use crate::wave::{Shape, Wave};
use snn_core::{Activity, Rng};

#[derive(Debug, Clone)]
pub struct Net {
    pub p: Params,
    pub classes: usize,
    pub s1: S1,
    pub l1: Conv,
    pub l2: Conv,
    pub l3: Conv,
    /// the image as a rank-order wave
    pub input: Wave,
    pub fire1: Fire,
    pub pool1: Wave,
    pub fire2: Fire,
    pub pool2: Wave,
    pub fire3: Fire,
    /// layers the last pass ran
    pub depth: usize,
    /// decision neurons that lead after each step
    pub leader: Vec<Option<u16>>,
    /// largest decision potential of each class after each step, step-major
    pub evidence: Vec<f32>,
    /// neurons that learn from this image
    pub winners: Vec<Winner>,
    /// STDP (before, after) of L1 and L2 as their schedules have them now
    pub rates: [(f32, f32); 2],
    /// R-STDP after a hit and after a miss, scaled by the last batch
    pub reward: (f32, f32),
    pub punish: (f32, f32),
    /// images each feature layer trained on
    trained: [usize; 2],
    /// hits, misses and images of the current batch
    batch: (usize, usize, usize),
}

impl Net {
    pub fn new(
        width: usize,
        height: usize,
        classes: usize,
        p: Params,
        rng: &mut Rng,
    ) -> Result<Self, String> {
        let [s0, s1, s2, s3] = p.shapes(width, height, classes)?.map(Shape::new);
        let pooled = |s: Shape, k: usize| Shape::new([s.maps, s.h / k, s.w / k]);
        let (p1, p2) = (pooled(s1, p.l1.pool), pooled(s2, p.l2.pool));
        let init = (p.init_mean, p.init_sd);
        let unit = (0.0, 1.0);
        let l1 = Conv::new(
            s0,
            s1,
            p.l1.kernel,
            p.l1.pad,
            p.l1.threshold,
            unit,
            init,
            rng,
        );
        let l2 = Conv::new(
            p1,
            s2,
            p.l2.kernel,
            p.l2.pad,
            p.l2.threshold,
            unit,
            init,
            rng,
        );
        let bounds = (p.w_min, p.w_max);
        let l3 = Conv::new(
            p2,
            s3,
            p.l3_kernel,
            p.l3_pad,
            f32::INFINITY,
            bounds,
            init,
            rng,
        );
        let feature = (p.ltp, -p.ltd);
        Ok(Self {
            s1: S1::new(width, height, p.dog_threshold, p.norm_radius, p.steps),
            input: Wave::new(s0),
            fire1: Fire::new(s1),
            pool1: Wave::new(p1),
            fire2: Fire::new(s2),
            pool2: Wave::new(p2),
            fire3: Fire::new(s3),
            depth: 0,
            leader: Vec::new(),
            evidence: Vec::new(),
            winners: Vec::new(),
            rates: [feature; 2],
            reward: (p.reward_ltp, -p.reward_ltd),
            punish: (-p.punish_ltp, p.punish_ltd),
            trained: [0; 2],
            batch: (0, 0, 0),
            l1,
            l2,
            l3,
            classes,
            p,
        })
    }

    /// Codes an image into the input wave.
    pub fn encode(&mut self, pixels: &[u8]) {
        self.s1.encode(pixels, &mut self.input);
    }

    /// Runs the network as far as `phase` needs and returns the answer. A
    /// feature layer learning from the image fires under pointwise inhibition
    /// and picks its winners; the layers above it do not run.
    pub fn forward(&mut self, phase: Phase, learn: bool) -> Option<u8> {
        let steps = self.p.steps;
        self.winners.clear();
        self.leader.clear();
        self.evidence.clear();
        self.l1
            .forward(&self.input, steps, &mut self.fire1, |_, _| {});
        if learn && phase == Phase::Layer1 {
            let l = &self.p.l1;
            pointwise_inhibition(&mut self.fire1, steps);
            winners(&self.fire1, steps, l.winners, l.radius, &mut self.winners);
            self.depth = 1;
            return None;
        }
        self.fire1
            .out
            .pool_into(self.p.l1.pool, &mut self.pool1, steps);
        self.l2
            .forward(&self.pool1, steps, &mut self.fire2, |_, _| {});
        if learn && phase == Phase::Layer2 {
            let l = &self.p.l2;
            pointwise_inhibition(&mut self.fire2, steps);
            winners(&self.fire2, steps, l.winners, l.radius, &mut self.winners);
            self.depth = 2;
            return None;
        }
        self.fire2
            .out
            .pool_into(self.p.l2.pool, &mut self.pool2, steps);
        let (shape, per, classes) = (self.l3.output, self.p.per_class, self.classes);
        let (leader, evidence) = (&mut self.leader, &mut self.evidence);
        self.l3
            .forward(&self.pool2, steps, &mut self.fire3, |_, pot| {
                leader.push(top(pot, shape));
                let at = evidence.len();
                evidence.resize(at + classes, 0.0);
                let ev = &mut evidence[at..];
                for maps in pot.chunks_exact(shape.maps) {
                    for (e, class) in ev.iter_mut().zip(maps.chunks_exact(per)) {
                        *e = class.iter().copied().fold(*e, f32::max);
                    }
                }
            });
        self.depth = 3;
        // decision neurons fire once, at the last step, if their potential is positive
        let win = *self.leader.last()?;
        let w = win?;
        self.winners.push(Winner {
            unit: w,
            time: (steps - 1) as u8,
            pot: self.fire3.pot[w as usize],
        });
        Some(self.class_of(w))
    }

    /// Class a decision neuron stands for.
    pub fn class_of(&self, unit: u16) -> u8 {
        (unit as usize % self.l3.output.maps / self.p.per_class) as u8
    }

    /// Learns from the last forward pass: STDP on the winners of a feature
    /// layer, or R-STDP on the decision neuron that answered.
    pub fn learn(&mut self, phase: Phase, label: u8, answer: Option<u8>) {
        match phase {
            Phase::Layer1 | Phase::Layer2 => {
                let l = phase.index();
                self.trained[l] += 1;
                // the rates double every `rate_every` images, up to `rate_max`
                if self.trained[l].is_multiple_of(self.p.rate_every.max(1)) {
                    let ltp = (2.0 * self.rates[l].0).min(self.p.rate_max);
                    self.rates[l] = (ltp, -ltp * self.p.ltd / self.p.ltp);
                }
                let (conv, input) = match phase {
                    Phase::Layer1 => (&mut self.l1, &self.input),
                    _ => (&mut self.l2, &self.pool1),
                };
                for w in &self.winners {
                    conv.stdp(input, w.unit as usize, w.time, self.rates[l], true);
                }
            }
            Phase::Decision => {
                let Some(w) = self.winners.first().copied() else {
                    self.count(None, label);
                    return;
                };
                let rate = if answer == Some(label) {
                    self.reward
                } else {
                    self.punish
                };
                self.l3
                    .stdp(&self.pool2, w.unit as usize, w.time, rate, false);
                self.count(answer, label);
            }
        }
    }

    // reward scales with the last batch's miss rate, punishment with its hit rate
    fn count(&mut self, answer: Option<u8>, label: u8) {
        let b = &mut self.batch;
        b.0 += (answer == Some(label)) as usize;
        b.1 += (answer.is_some() && answer != Some(label)) as usize;
        b.2 += 1;
        if b.2 >= self.p.batch.max(1) {
            let (hit, miss) = (b.0 as f32 / b.2 as f32, b.1 as f32 / b.2 as f32);
            let p = &self.p;
            self.reward = (p.reward_ltp * miss, -p.reward_ltd * miss);
            self.punish = (-p.punish_ltp * hit, p.punish_ltd * hit);
            self.batch = (0, 0, 0);
        }
    }

    /// Spikes of the last pass and the synapses they reached; the decision
    /// layer is read by its potentials and the filters by MACs.
    pub fn activity(&self) -> Activity {
        let mut spikes = (self.input.spikes() + self.fire1.out.spikes()) as u64;
        let mut synops = self.fire1.synops;
        if self.depth >= 2 {
            spikes += self.fire2.out.spikes() as u64;
            synops += self.fire2.synops;
        }
        if self.depth >= 3 {
            synops += self.fire3.synops;
        }
        Activity::new(spikes, synops).with_macs(self.s1.macs)
    }

    /// MACs per image of a dense ANN with the same filters and layers.
    pub fn dense_macs(&self) -> u64 {
        let conv = |c: &Conv| (c.output.len() * c.fan_in()) as u64;
        self.s1.dense_macs() + conv(&self.l1) + conv(&self.l2) + conv(&self.l3)
    }

    /// Feature `f` of L1 seen in pixels: its weights spread back through the
    /// DoG filters, a (kernel + 12)² image for 13-wide filters.
    pub fn l1_feature(&self, f: usize, out: &mut Vec<f32>) {
        let (k, c) = (self.l1.kernel, &self.l1);
        let r = DOG.iter().map(|d| d.0).max().unwrap_or(1) / 2;
        let side = k + 2 * r;
        out.clear();
        out.resize(side * side, 0.0);
        let kernels: Vec<Vec<f32>> = DOG
            .iter()
            .map(|&(n, a, b)| crate::input::dog(n, a, b))
            .collect();
        for ky in 0..k {
            for kx in 0..k {
                for (ch, dog) in kernels.iter().enumerate() {
                    let w = c.weight(f, ky, kx, ch);
                    let n = DOG[ch].0;
                    let off = r - n / 2;
                    for dy in 0..n {
                        for dx in 0..n {
                            out[(ky + off + dy) * side + kx + off + dx] += w * dog[dy * n + dx];
                        }
                    }
                }
            }
        }
    }

    /// Features of L2 seen in pixels, from the L1 features (one per input map,
    /// each `l1_feature`-sized): each weight spreads its map's feature over the
    /// pooling window it reads.
    pub fn l2_feature(&self, g: usize, l1: &[Vec<f32>], out: &mut Vec<f32>) {
        let (k, pool, ci) = (self.l2.kernel, self.p.l1.pool, self.l2.input.maps);
        let s1 = (l1[0].len() as f32).sqrt() as usize;
        let side = k * pool + s1 - 1;
        out.clear();
        out.resize(side * side, 0.0);
        let share = 1.0 / (pool * pool) as f32;
        for ky in 0..k {
            for kx in 0..k {
                for (c, feat) in l1.iter().enumerate().take(ci) {
                    let w = self.l2.weight(g, ky, kx, c) * share;
                    if w == 0.0 {
                        continue;
                    }
                    for sy in 0..pool {
                        for sx in 0..pool {
                            let (oy, ox) = (ky * pool + sy, kx * pool + sx);
                            for y in 0..s1 {
                                let row = &mut out[(oy + y) * side + ox..][..s1];
                                row.iter_mut()
                                    .zip(&feat[y * s1..][..s1])
                                    .for_each(|(o, &f)| *o += w * f);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Unit with the largest positive potential; ties go to the lowest map, row
/// and column, as SpykeTorch's winner search breaks them.
pub fn top(pot: &[f32], shape: Shape) -> Option<u16> {
    let max = pot.iter().copied().fold(0.0, f32::max);
    if max <= 0.0 {
        return None;
    }
    let positions = shape.h * shape.w;
    (0..shape.maps)
        .flat_map(|m| (0..positions).map(move |pos| pos * shape.maps + m))
        .find(|&u| pot[u] == max)
        .map(|u| u as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digit(d: u8, seed: u64) -> Vec<u8> {
        let mut img = vec![0u8; 784];
        snn_core::dataset::render_digit(d, 28, &mut Rng::new(seed), &mut img);
        img
    }

    fn net() -> Net {
        Net::new(28, 28, 10, Params::default(), &mut Rng::new(1)).unwrap()
    }

    #[test]
    fn phases_run_as_deep_as_they_need() {
        let mut n = net();
        n.encode(&digit(3, 1));
        assert_eq!(n.forward(Phase::Layer1, true), None);
        assert_eq!(n.depth, 1);
        assert!(!n.winners.is_empty() && n.winners.len() <= 5);
        // under pointwise inhibition at most one map fires per position
        let s = n.fire1.out.shape;
        for pos in 0..s.h * s.w {
            let fired = n.fire1.out.time[pos * s.maps..][..s.maps]
                .iter()
                .filter(|&&t| t != crate::wave::NEVER);
            assert!(fired.count() <= 1);
        }
        n.forward(Phase::Layer2, true);
        assert_eq!(n.depth, 2);
        assert!(n.winners.len() <= 8);
        let answer = n.forward(Phase::Layer1, false);
        assert_eq!(n.depth, 3);
        assert!(answer.is_some());
        assert_eq!((n.leader.len(), n.evidence.len()), (15, 150));
        let w = n.winners[0];
        assert_eq!(answer, Some(n.class_of(w.unit)));
        let k = answer.unwrap() as usize;
        assert_eq!(n.evidence[14 * 10 + k], w.pot);
        assert!(n.evidence[140..].iter().all(|&e| e <= w.pot));
    }

    #[test]
    fn activity_counts_what_ran() {
        let mut n = net();
        n.encode(&digit(5, 2));
        n.forward(Phase::Decision, false);
        let a = n.activity();
        let spikes = n.input.spikes() + n.fire1.out.spikes() + n.fire2.out.spikes();
        assert_eq!(a.spikes, spikes as u64);
        assert_eq!(a.synops, n.fire1.synops + n.fire2.synops + n.fire3.synops);
        assert!(a.macs > 0 && n.dense_macs() > 30_000_000);
    }

    #[test]
    fn feature_rates_double_on_schedule() {
        let p = Params {
            rate_every: 2,
            ..Params::default()
        };
        let mut n = Net::new(28, 28, 10, p, &mut Rng::new(1)).unwrap();
        let img = digit(1, 3);
        for _ in 0..4 {
            n.encode(&img);
            n.forward(Phase::Layer1, true);
            n.learn(Phase::Layer1, 1, None);
        }
        let (a, b) = n.rates[0];
        assert!(
            (a - 0.016).abs() < 1e-6 && (b + 0.012).abs() < 1e-6,
            "{a} {b}"
        );
        assert_eq!(n.rates[1], (0.004, -0.003), "layer 2 has not trained");
        assert!(n.l1.convergence() != net().l1.convergence());
    }

    #[test]
    fn r_stdp_rates_follow_the_batch() {
        let p = Params {
            batch: 4,
            ..Params::default()
        };
        let mut n = Net::new(28, 28, 10, p, &mut Rng::new(1)).unwrap();
        // three hits and one miss
        for (answer, label) in [(1, 1), (2, 2), (3, 3), (4, 0)] {
            n.count(Some(answer), label);
        }
        assert!((n.reward.0 - 0.001).abs() < 1e-7);
        assert!((n.punish.0 + 0.003).abs() < 1e-7 && (n.punish.1 - 0.000375).abs() < 1e-7);
    }

    #[test]
    fn reward_and_punishment_move_the_answering_map() {
        let mut n = net();
        n.encode(&digit(7, 4));
        let answer = n.forward(Phase::Decision, true).unwrap();
        let w = n.winners[0];
        let m = w.unit as usize % n.l3.output.maps;
        let sum = |n: &Net| n.l3.map(m).sum::<f32>();
        let before = sum(&n);
        n.learn(Phase::Decision, answer, Some(answer));
        let rewarded = sum(&n);
        n.learn(Phase::Decision, (answer + 1) % 10, Some(answer));
        assert!(rewarded != before && sum(&n) != rewarded);
        assert!(n.l3.map(m).all(|w| (0.2..=0.8).contains(&w)));
    }

    #[test]
    fn top_breaks_ties_by_map_first() {
        let s = Shape::new([2, 1, 2]);
        // units: (pos 0, map 0), (pos 0, map 1), (pos 1, map 0), (pos 1, map 1)
        assert_eq!(top(&[1.0, 3.0, 2.0, 3.0], s), Some(1));
        assert_eq!(top(&[0.5, 0.0, 0.5, 0.0], s), Some(0));
        assert_eq!(top(&[0.0, 0.0, 0.0, 0.5], s), Some(3));
        assert_eq!(top(&[0.0; 4], s), None);
    }

    #[test]
    fn features_reconstruct_to_pixel_patches() {
        let n = net();
        let mut f = Vec::new();
        n.l1_feature(0, &mut f);
        assert_eq!(f.len(), 17 * 17);
        let l1: Vec<Vec<f32>> = (0..30)
            .map(|i| {
                let mut v = Vec::new();
                n.l1_feature(i, &mut v);
                v
            })
            .collect();
        let mut g = Vec::new();
        n.l2_feature(0, &l1, &mut g);
        assert_eq!(g.len(), 22 * 22);
        assert!(g.iter().any(|&v| v != 0.0));
    }
}
