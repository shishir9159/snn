//! Convolutional layers of non-leaky integrate-and-fire neurons that fire at
//! most once. Spikes are cumulative, as in SpykeTorch's spike waves: an input
//! that spiked at step t adds its weight to every potential it reaches from t
//! on. Each layer runs event by event, step by step, and keeps for every neuron
//! the step it first went above threshold and its potential then.

use crate::wave::{Shape, Wave, NEVER};
use snn_core::Rng;

/// What a forward pass leaves in a layer.
#[derive(Debug, Clone)]
pub struct Fire {
    /// potentials after the last step; -inf for neurons that fired
    pub pot: Vec<f32>,
    /// potential at the step each neuron first went above threshold
    pub at_fire: Vec<f32>,
    pub out: Wave,
    /// spike deliveries: one per input spike per neuron it reached
    pub synops: u64,
    /// 1 where a potential is above threshold, padded to whole words
    over: Vec<u8>,
}

impl Fire {
    pub fn new(shape: Shape) -> Self {
        Self {
            pot: vec![0.0; shape.len()],
            at_fire: vec![0.0; shape.len()],
            out: Wave::new(shape),
            synops: 0,
            over: vec![0; shape.len().div_ceil(8) * 8],
        }
    }
}

/// A neuron chosen to learn: earliest spike first, then largest potential.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Winner {
    pub unit: u16,
    pub time: u8,
    pub pot: f32,
}

#[derive(Debug, Clone)]
pub struct Conv {
    pub input: Shape,
    pub output: Shape,
    pub kernel: usize,
    pub pad: usize,
    /// above this a neuron fires; infinite for a layer read by its potentials
    pub threshold: f32,
    /// weights stay in this range
    pub bounds: (f32, f32),
    /// window row, input map, window column from the right, then output map:
    /// an input reaches a row of outputs through one contiguous run
    pub w: Vec<f32>,
    /// sum of u(1 - u) over each map's weights, u the weight scaled to the bounds
    spread: Vec<f32>,
}

impl Conv {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: Shape,
        output: Shape,
        kernel: usize,
        pad: usize,
        threshold: f32,
        bounds: (f32, f32),
        (mean, sd): (f32, f32),
        rng: &mut Rng,
    ) -> Self {
        let mut c = Self {
            input,
            output,
            kernel,
            pad,
            threshold,
            bounds,
            w: vec![0.0; kernel * kernel * input.maps * output.maps],
            spread: vec![0.0; output.maps],
        };
        // drawn in window order, whatever the storage order
        for ky in 0..kernel {
            for kx in 0..kernel {
                for ch in 0..input.maps {
                    let at = c.at(ky, kx, ch);
                    for w in &mut c.w[at..at + output.maps] {
                        *w = mean + sd * rng.normal();
                    }
                }
            }
        }
        c.spread = (0..output.maps)
            .map(|f| c.map(f).map(|w| c.spread_of(w)).sum())
            .collect();
        c
    }

    /// Weights per map.
    pub fn fan_in(&self) -> usize {
        self.kernel * self.kernel * self.input.maps
    }

    /// Weights of map `f`, in storage order.
    pub fn map(&self, f: usize) -> impl Iterator<Item = f32> + '_ {
        self.w.iter().skip(f).step_by(self.output.maps).copied()
    }

    // where the weights from input map `c` at window (ky, kx) start
    fn at(&self, ky: usize, kx: usize, c: usize) -> usize {
        let k = self.kernel;
        ((ky * self.input.maps + c) * k + k - 1 - kx) * self.output.maps
    }

    /// Weight from input map `c` at window (ky, kx) into map `m`.
    pub fn weight(&self, m: usize, ky: usize, kx: usize, c: usize) -> f32 {
        self.w[self.at(ky, kx, c) + m]
    }

    fn spread_of(&self, w: f32) -> f32 {
        let (lo, hi) = self.bounds;
        let u = ((w - lo) / (hi - lo)).clamp(0.0, 1.0);
        u * (1.0 - u)
    }

    /// Kheradpisheh's convergence: mean u(1 - u) over the weights, 0 once every
    /// weight sits at a bound.
    pub fn convergence(&self) -> f32 {
        self.spread.iter().sum::<f32>() / self.w.len() as f32
    }

    /// Runs the input wave through the layer; `each_step` sees the potentials
    /// after each step.
    pub fn forward<F: FnMut(usize, &[f32])>(
        &self,
        input: &Wave,
        steps: usize,
        fire: &mut Fire,
        mut each_step: F,
    ) {
        let (k, pad) = (self.kernel, self.pad);
        let o = self.output;
        let nf = o.maps;
        fire.pot.fill(0.0);
        fire.at_fire.fill(0.0);
        fire.out.time.fill(NEVER);
        fire.synops = 0;
        for t in 0..steps {
            let spikes = input.at(t);
            for &u in spikes {
                let (iy, ix, c) = self.input.split(u as usize);
                // outputs whose window covers the input: oy = iy + pad - ky, ox likewise
                let ky_lo = (iy + pad + 1).saturating_sub(o.h);
                let (kx_lo, kx_hi) = ((ix + pad + 1).saturating_sub(o.w), (ix + pad).min(k - 1));
                if kx_lo > kx_hi {
                    continue;
                }
                let n = (kx_hi - kx_lo + 1) * nf;
                let ox = ix + pad - kx_hi;
                for ky in ky_lo..=(iy + pad).min(k - 1) {
                    let row = &self.w[self.at(ky, kx_hi, c)..][..n];
                    let p = &mut fire.pot[((iy + pad - ky) * o.w + ox) * nf..][..n];
                    p.iter_mut().zip(row).for_each(|(v, &w)| *v += w);
                    fire.synops += n as u64;
                }
            }
            if !spikes.is_empty() && self.threshold.is_finite() {
                // a neuron that fires drops to -inf and never fires again; the
                // test is a byte per neuron in one vector pass, read eight at a time
                let th = self.threshold;
                for (m, &v) in fire.over.iter_mut().zip(&fire.pot) {
                    *m = (v > th) as u8;
                }
                for (w, word) in fire.over.chunks_exact(8).enumerate() {
                    let mut bits = u64::from_le_bytes(word.try_into().unwrap());
                    while bits != 0 {
                        let i = w * 8 + bits.trailing_zeros() as usize / 8;
                        bits &= bits - 1;
                        fire.at_fire[i] = fire.pot[i];
                        fire.out.time[i] = t as u8;
                        fire.pot[i] = f32::NEG_INFINITY;
                    }
                }
            }
            each_step(t, &fire.pot);
        }
        fire.out.index(steps);
    }

    /// Potential of output `unit` after each step, from the input wave.
    pub fn trace(&self, input: &Wave, unit: usize, steps: usize, out: &mut Vec<f32>) {
        out.clear();
        out.resize(steps, 0.0);
        self.window(input, unit, |t_in, w| {
            if (t_in as usize) < steps {
                out[t_in as usize] += w;
            }
        });
        for t in 1..steps {
            out[t] += out[t - 1];
        }
    }

    // calls `f(input time, weight)` for each synapse of output `unit`
    fn window<F: FnMut(u8, f32)>(&self, input: &Wave, unit: usize, mut f: F) {
        let (oy, ox, m) = self.output.split(unit);
        for ky in 0..self.kernel {
            for kx in 0..self.kernel {
                let (iy, ix) = (
                    (oy + ky).wrapping_sub(self.pad),
                    (ox + kx).wrapping_sub(self.pad),
                );
                let inside = iy < self.input.h && ix < self.input.w;
                for c in 0..self.input.maps {
                    let t = if inside {
                        input.time[self.input.unit(iy, ix, c)]
                    } else {
                        NEVER
                    };
                    f(t, self.w[self.at(ky, kx, c) + m]);
                }
            }
        }
    }

    /// STDP on the map of the neuron `unit` that fired at `t_out`: a synapse
    /// whose input spiked at or before it changes by `before`, any other by
    /// `after`; with `stabilize` both scale by (w - lo)(hi - w).
    pub fn stdp(
        &mut self,
        input: &Wave,
        unit: usize,
        t_out: u8,
        (before, after): (f32, f32),
        stabilize: bool,
    ) {
        let (oy, ox, m) = self.output.split(unit);
        let (lo, hi) = self.bounds;
        let mut spread = 0.0;
        for ky in 0..self.kernel {
            for kx in 0..self.kernel {
                let (iy, ix) = (
                    (oy + ky).wrapping_sub(self.pad),
                    (ox + kx).wrapping_sub(self.pad),
                );
                let inside = iy < self.input.h && ix < self.input.w;
                for c in 0..self.input.maps {
                    let t = if inside {
                        input.time[self.input.unit(iy, ix, c)]
                    } else {
                        NEVER
                    };
                    let at = self.at(ky, kx, c) + m;
                    let w = &mut self.w[at];
                    let a = if t != NEVER && t <= t_out {
                        before
                    } else {
                        after
                    };
                    let s = if stabilize {
                        (*w - lo) * (hi - *w)
                    } else {
                        1.0
                    };
                    *w = (*w + a * s).clamp(lo, hi);
                    let u = (*w - lo) / (hi - lo);
                    spread += u * (1.0 - u);
                }
            }
        }
        self.spread[m] = spread;
    }
}

/// Lets at most one map fire at each position: the earliest, then the one
/// with the largest potential then. The others are silenced for this image.
pub fn pointwise_inhibition(fire: &mut Fire, steps: usize) {
    let nf = fire.out.shape.maps;
    for (time, pot) in fire
        .out
        .time
        .chunks_exact_mut(nf)
        .zip(fire.at_fire.chunks_exact_mut(nf))
    {
        let first = time.iter().copied().fold(NEVER, u8::min);
        if first == NEVER {
            continue;
        }
        let mut best = 0;
        for f in (0..nf).filter(|&f| time[f] == first) {
            if time[best] != first || pot[f] > pot[best] {
                best = f;
            }
        }
        for f in (0..nf).filter(|&f| f != best) {
            time[f] = NEVER;
            pot[f] = 0.0;
        }
    }
    fire.out.index(steps);
}

/// Up to `k` neurons that learn, at most one per map, each silencing every map
/// within `radius` of it (none for radius 0); earliest spike first, then
/// largest potential, then lowest map, row and column.
pub fn winners(fire: &Fire, steps: usize, k: usize, radius: usize, out: &mut Vec<Winner>) {
    out.clear();
    let s = fire.out.shape;
    let mut used = vec![false; s.maps];
    let mut bucket: Vec<Winner> = Vec::new();
    for t in 0..steps {
        bucket.clear();
        bucket.extend(fire.out.at(t).iter().map(|&u| Winner {
            unit: u,
            time: t as u8,
            pot: fire.at_fire[u as usize],
        }));
        let key = |w: &Winner| {
            let (y, x, m) = s.split(w.unit as usize);
            (m, y, x)
        };
        bucket.sort_by(|a, b| b.pot.total_cmp(&a.pot).then_with(|| key(a).cmp(&key(b))));
        for &c in &bucket {
            if out.len() == k {
                return;
            }
            let (y, x, m) = s.split(c.unit as usize);
            let near = |w: &Winner| {
                let (wy, wx, _) = s.split(w.unit as usize);
                y.abs_diff(wy) <= radius && x.abs_diff(wx) <= radius
            };
            if used[m] || (radius > 0 && out.iter().any(near)) {
                continue;
            }
            used[m] = true;
            out.push(c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_wave(shape: Shape, steps: usize, p: f32, rng: &mut Rng) -> Wave {
        let mut w = Wave::new(shape);
        for t in w.time.iter_mut() {
            if rng.f32() < p {
                *t = rng.below(steps) as u8;
            }
        }
        w.index(steps);
        w
    }

    #[test]
    fn forward_matches_a_dense_convolution() {
        let mut rng = Rng::new(4);
        let (input, steps) = (Shape::new([3, 7, 6]), 6);
        let output = Shape::new([4, 7, 6]);
        let conv = Conv::new(input, output, 3, 1, 1.5, (0.0, 1.0), (0.5, 0.3), &mut rng);
        let wave = random_wave(input, steps, 0.4, &mut rng);
        let mut fire = Fire::new(output);
        let mut seen = Vec::new();
        conv.forward(&wave, steps, &mut fire, |t, pot| seen.push((t, pot[5])));
        assert_eq!(seen.len(), steps);
        for u in 0..output.len() {
            let mut tr = Vec::new();
            conv.trace(&wave, u, steps, &mut tr);
            let first = tr.iter().position(|&v| v > conv.threshold);
            match first {
                Some(t) => {
                    assert_eq!(fire.out.time[u], t as u8, "unit {u}");
                    assert!((fire.at_fire[u] - tr[t]).abs() < 1e-4);
                    assert_eq!(fire.pot[u], f32::NEG_INFINITY);
                }
                None => {
                    assert_eq!(fire.out.time[u], NEVER);
                    assert!((tr[steps - 1] - fire.pot[u]).abs() < 1e-4);
                }
            }
            if u == 5 && first.is_none() {
                assert!(seen.iter().all(|&(t, v)| (tr[t] - v).abs() < 1e-4));
            }
        }
        assert!(fire.out.spikes() > 0 && fire.out.spikes() < output.len());
        let fan: u64 = (0..output.len())
            .map(|u| {
                let mut n = 0;
                conv.window(&wave, u, |t, _| n += (t != NEVER) as u64);
                n
            })
            .sum();
        assert_eq!(fire.synops, fan);
    }

    fn fire_of(shape: Shape, spikes: &[(usize, u8, f32)]) -> Fire {
        let mut f = Fire::new(shape);
        for &(u, t, v) in spikes {
            f.out.time[u] = t;
            f.at_fire[u] = v;
        }
        f.out.index(15);
        f
    }

    #[test]
    fn pointwise_inhibition_keeps_the_earliest_then_strongest() {
        let s = Shape::new([3, 1, 2]);
        let mut f = fire_of(
            s,
            &[
                (0, 4, 20.0),
                (1, 2, 16.0),
                (2, 2, 18.0),
                (3, 7, 30.0),
                (5, 7, 30.0),
            ],
        );
        pointwise_inhibition(&mut f, 15);
        assert_eq!(f.out.time, [NEVER, NEVER, 2, 7, NEVER, NEVER]);
        assert_eq!(f.out.spikes(), 2);
    }

    #[test]
    fn winners_differ_in_map_and_keep_their_distance() {
        let s = Shape::new([3, 5, 5]);
        let f = fire_of(
            s,
            &[
                (s.unit(2, 2, 0), 1, 16.0),
                (s.unit(0, 0, 1), 1, 20.0),
                (s.unit(4, 4, 0), 0, 15.5),
                (s.unit(2, 3, 2), 3, 40.0),
                (s.unit(4, 0, 2), 5, 40.0),
            ],
        );
        let mut w = Vec::new();
        winners(&f, 15, 5, 1, &mut w);
        let units: Vec<u16> = w.iter().map(|w| w.unit).collect();
        // (2,2,0) is map 0, taken by (4,4,0); (2,3,2) sits by nobody yet
        let want = [s.unit(4, 4, 0), s.unit(0, 0, 1), s.unit(2, 3, 2)];
        assert_eq!(units, want.map(|u| u as u16));
        winners(&f, 15, 2, 1, &mut w);
        assert_eq!(w.len(), 2);
        // radius 3 around (4,4) silences (2,3)
        winners(&f, 15, 5, 3, &mut w);
        assert_eq!(
            w.iter().map(|w| w.unit).collect::<Vec<_>>(),
            [s.unit(4, 4, 0), s.unit(0, 0, 1), s.unit(4, 0, 2)].map(|u| u as u16)
        );
    }

    #[test]
    fn stdp_follows_the_order_of_spikes() {
        let mut rng = Rng::new(1);
        let (input, output) = (Shape::new([1, 3, 3]), Shape::new([2, 1, 1]));
        let mut c = Conv::new(input, output, 3, 0, 1.0, (0.0, 1.0), (0.5, 0.0), &mut rng);
        let mut wave = Wave::new(input);
        wave.time[0] = 2;
        wave.time[1] = 3;
        wave.time[2] = 4;
        wave.index(15);
        let before = c.convergence();
        c.stdp(&wave, 1, 3, (0.1, -0.1), true);
        let row = |c: &Conv, m: usize, ky: usize| [0, 1, 2].map(|kx| c.weight(m, ky, kx, 0));
        // two inputs at or before 3 grow by 0.1 * 0.25, the rest shrink
        let near = |w: f32, v: f32| (w - v).abs() < 1e-6;
        let m = row(&c, 1, 0);
        assert!(near(m[0], 0.525) && near(m[1], 0.525) && near(m[2], 0.475));
        assert!(row(&c, 1, 2).iter().all(|&w| near(w, 0.475)));
        assert!(c.map(0).all(|w| w == 0.5), "other maps keep their weights");
        assert!(c.convergence() < before);
        // without the stabiliser the step is plain and clamped
        c.bounds = (0.2, 0.8);
        c.stdp(&wave, 0, 3, (0.5, -0.5), false);
        assert_eq!(row(&c, 0, 0), [0.8, 0.8, 0.2]);
    }
}
