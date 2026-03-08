//! Integer-only inference for the winner-take-all layer, the arithmetic a
//! chip would do. Weights are levels on a grid of `bits` bits over
//! [w_min, w_max]; potentials are integers in units of a level step / 2^FRAC;
//! a neuron's threshold is the image's base times an integer gain. The rules
//! are those of [`Layer::present`] without learning. This model's leak is
//! linear, so it stays a subtraction rather than becoming a shift.
//!
//! [`Fixed::write_config`] and [`Fixed::write_image`] export test vectors a
//! hardware design can be checked against; `scripts/check_vectors.py` replays
//! them on its own.

use crate::layer::Layer;
use crate::params::Params;
use snn_core::encode::SpikeTrain;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// Fractional bits of a potential below one weight level.
pub const FRAC: u32 = 8;
/// Fractional bits of a threshold gain.
pub const GAIN: u32 = 8;
/// Fractional bits of the threshold base.
pub const BASE: u32 = 16;
/// Most weight bits the 32-bit potentials allow.
pub const MAX_BITS: u32 = 12;

#[derive(Debug, Clone)]
pub struct Fixed {
    pub n_in: usize,
    pub n_out: usize,
    pub bits: u32,
    /// weight levels, input-major, so an input spike reads one run
    levels: Vec<u16>,
    /// w_min in potential units, added once per input spike
    offset: i32,
    leak: i32,
    rest: i32,
    floor: i32,
    t_ref: i32,
    /// per neuron: threshold for a base of 1, in potential units << GAIN
    gain: Vec<i64>,
    /// 1 / threshold_div, << BASE
    div: i64,
    p: Vec<i32>,
    t_rest: Vec<i32>,
    drive: Vec<i32>,
    thr: Vec<i32>,
    fired: Vec<u16>,
}

/// What a presentation ended with.
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub winner: Option<usize>,
    pub spikes: Vec<u32>,
    /// largest potential magnitude, for sizing registers
    pub peak: i32,
}

impl Fixed {
    /// The layer's weights snapped to `bits` levels; weights learned on that
    /// grid (`weight_bits`) keep their values. Above 12 bits the floor would
    /// not fit a 32-bit potential.
    pub fn new(layer: &Layer, p: &Params, bits: u32) -> Self {
        assert!(
            (1..=MAX_BITS).contains(&bits),
            "1 to {MAX_BITS} weight bits"
        );
        let (lo, hi) = (p.stdp.w_min, p.stdp.w_max);
        let top = ((1u32 << bits) - 1) as f32;
        let step = (hi - lo) / top;
        let unit = step / (1 << FRAC) as f32;
        let units = |v: f32| (v / unit).round() as i32;
        let (n_in, n_out) = (layer.n_in, layer.n_out);
        let mut levels = vec![0u16; n_in * n_out];
        for j in 0..n_out {
            for (i, &w) in layer.row(j).iter().enumerate() {
                levels[i * n_out + j] = ((w - lo) / step).round().clamp(0.0, top) as u16;
            }
        }
        let gain = layer
            .theta
            .iter()
            .map(|&th| ((1.0 + th) / unit * (1 << GAIN) as f32).round() as i64)
            .collect();
        Self {
            n_in,
            n_out,
            bits,
            levels,
            offset: units(lo),
            leak: units(p.lif.leak),
            rest: units(p.lif.rest),
            floor: units(p.lif.floor),
            t_ref: p.lif.t_ref as i32,
            gain,
            div: ((1 << BASE) as f32 / p.threshold_div).round() as i64,
            p: vec![0; n_out],
            t_rest: vec![-1; n_out],
            drive: vec![0; n_out],
            thr: vec![0; n_out],
            fired: Vec::with_capacity(n_out),
        }
    }

    /// Weight level of input `i` onto output `j`.
    pub fn level(&self, i: usize, j: usize) -> u16 {
        self.levels[i * self.n_out + j]
    }

    /// Runs a presentation; `step` sees each step's potentials and the outputs
    /// that fired.
    pub fn present<F>(&mut self, train: &SpikeTrain, mut step: F) -> Run
    where
        F: FnMut(usize, &[i32], &[u16]),
    {
        assert_eq!(train.n_inputs(), self.n_in);
        // the base is the peak count of simultaneous inputs over threshold_div, at least 1
        let base = (train.max_simultaneous() as i64 * self.div).max(1 << BASE);
        let round = 1i64 << (BASE + GAIN - 1);
        for (th, &g) in self.thr.iter_mut().zip(&self.gain) {
            *th = ((base * g + round) >> (BASE + GAIN)).min(i32::MAX as i64) as i32;
        }
        self.p.fill(self.rest);
        self.t_rest.fill(-1);
        let mut run = Run {
            winner: None,
            spikes: vec![0; self.n_out],
            peak: 0,
        };
        let n = self.n_out;
        for t in 1..=train.steps() {
            let ti = t as i32;
            let active = train.active(t);
            self.drive.fill(self.offset * active.len() as i32);
            for &i in active {
                let col = &self.levels[i as usize * n..][..n];
                for (d, &q) in self.drive.iter_mut().zip(col) {
                    *d += (q as i32) << FRAC;
                }
            }
            for j in 0..n {
                if ti <= self.t_rest[j] {
                    continue;
                }
                let mut p = self.p[j] + self.drive[j];
                if p > self.rest {
                    p = (p - self.leak).max(self.rest);
                }
                self.p[j] = p.max(self.floor);
                run.peak = run.peak.max(self.p[j].abs());
            }
            if run.winner.is_none() {
                run.winner = self.pick_winner(ti);
                if let Some(w) = run.winner {
                    for (j, p) in self.p.iter_mut().enumerate() {
                        if j != w {
                            *p = self.floor;
                        }
                    }
                }
            }
            self.fired.clear();
            for j in 0..n {
                if self.p[j] >= self.thr[j] {
                    self.p[j] = self.rest;
                    self.t_rest[j] = ti + self.t_ref;
                    run.spikes[j] += 1;
                    self.fired.push(j as u16);
                }
            }
            step(t, &self.p, &self.fired);
        }
        run
    }

    // free neuron furthest past its own threshold, if any reached it
    fn pick_winner(&self, t: i32) -> Option<usize> {
        let (mut best, mut margin) = (None, 0);
        for j in 0..self.n_out {
            if t <= self.t_rest[j] {
                continue;
            }
            let m = self.p[j] - self.thr[j];
            if m >= margin {
                margin = m;
                best = Some(j);
            }
        }
        best
    }

    /// Bits a signed register needs to hold `peak`.
    pub fn register_bits(peak: i32) -> u32 {
        33 - peak.max(1).leading_zeros()
    }

    /// `config.txt` (one `name value` per line), `weights.txt` (a row of input
    /// levels per output neuron) and `gains.txt` (one threshold gain per output).
    pub fn write_config(&self, dir: &Path, steps: usize) -> io::Result<()> {
        fs::create_dir_all(dir)?;
        let mut f = BufWriter::new(fs::File::create(dir.join("config.txt"))?);
        let rows = [
            ("inputs", self.n_in as i64),
            ("outputs", self.n_out as i64),
            ("steps", steps as i64),
            ("weight_bits", self.bits as i64),
            ("frac_bits", FRAC as i64),
            ("gain_bits", GAIN as i64),
            ("base_bits", BASE as i64),
            ("offset", self.offset as i64),
            ("leak", self.leak as i64),
            ("rest", self.rest as i64),
            ("floor", self.floor as i64),
            ("t_ref", self.t_ref as i64),
            ("div", self.div),
        ];
        for (name, v) in rows {
            writeln!(f, "{name} {v}")?;
        }
        f.flush()?;
        let mut f = BufWriter::new(fs::File::create(dir.join("weights.txt"))?);
        for j in 0..self.n_out {
            let row: Vec<String> = (0..self.n_in)
                .map(|i| self.level(i, j).to_string())
                .collect();
            writeln!(f, "{}", row.join(" "))?;
        }
        f.flush()?;
        let gains: Vec<String> = self.gain.iter().map(|g| g.to_string()).collect();
        fs::write(dir.join("gains.txt"), gains.join("\n") + "\n")
    }

    /// Image `k`'s vectors: `NNNN.in` (`label L`, then `t i i ...` per step
    /// with input spikes), `NNNN.out` (`winner W`, -1 for none, then `t j j ...`
    /// per step with output spikes) and `NNNN.trace` (every potential after
    /// each step, a line per step).
    pub fn write_image(
        &mut self,
        dir: &Path,
        k: usize,
        label: u8,
        train: &SpikeTrain,
    ) -> io::Result<Run> {
        let open = |ext: &str| -> io::Result<BufWriter<fs::File>> {
            Ok(BufWriter::new(fs::File::create(
                dir.join(format!("{k:04}.{ext}")),
            )?))
        };
        let mut input = open("in")?;
        writeln!(input, "label {label}")?;
        for t in 1..=train.steps() {
            let active = train.active(t);
            if !active.is_empty() {
                writeln!(input, "{t} {}", join(active))?;
            }
        }
        input.flush()?;
        let (mut out, mut trace) = (Vec::new(), String::new());
        let run = self.present(train, |t, p, fired| {
            if !fired.is_empty() {
                out.push(format!("{t} {}", join(fired)));
            }
            trace += &join(p);
            trace.push('\n');
        });
        let winner = run.winner.map_or(-1, |w| w as i64);
        let mut o = open("out")?;
        writeln!(o, "winner {winner}")?;
        for line in out {
            writeln!(o, "{line}")?;
        }
        o.flush()?;
        fs::write(dir.join(format!("{k:04}.trace")), trace)?;
        Ok(run)
    }
}

fn join<T: ToString>(v: &[T]) -> String {
    let s: Vec<String> = v.iter().map(T::to_string).collect();
    s.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use snn_core::encode::receptive_field;
    use snn_core::{dataset, Rng};

    fn encode(img: &[u8], p: &Params) -> SpikeTrain {
        let mut pot = Vec::new();
        receptive_field(img, 28, 28, &mut pot);
        let mut tr = SpikeTrain::new();
        tr.encode(&pot, p.time_steps, &p.encode, &mut Rng::new(0));
        tr
    }

    /// A layer trained a little, so thresholds and weights differ per neuron.
    fn trained(p: &Params) -> Layer {
        let data = dataset::synthetic(300, 4);
        let mut l = Layer::new(784, 10, p, &mut Rng::new(5));
        for i in 0..data.len() {
            l.present(&encode(data.image(i), p), p, true, &mut ());
        }
        l
    }

    #[test]
    fn levels_hold_the_grid_the_layer_learned_on() {
        let p = Params {
            weight_bits: 6,
            ..Params::default()
        };
        let l = trained(&p);
        let f = Fixed::new(&l, &p, 6);
        let step = (p.stdp.w_max - p.stdp.w_min) / 63.0;
        for j in 0..10 {
            for (i, &w) in l.row(j).iter().enumerate() {
                assert!((f.level(i, j) as f32 * step + p.stdp.w_min - w).abs() < 1e-4);
            }
        }
        assert_eq!(f.leak, (0.15 / step * 256.0).round() as i32);
    }

    #[test]
    fn integer_runs_pick_the_float_winner() {
        let p = Params::default();
        let mut l = trained(&p);
        let mut f = Fixed::new(&l, &p, 12);
        let test = dataset::synthetic(100, 9);
        let mut same = 0;
        for i in 0..test.len() {
            let tr = encode(test.image(i), &p);
            let float = l.present(&tr, &p, false, &mut ());
            let int = f.present(&tr, |_, _, _| {});
            same += (float.winner == int.winner) as usize;
            if float.winner == int.winner {
                let w = float.winner.map_or(0, |w| float.spikes[w]);
                let v = int.winner.map_or(0, |w| int.spikes[w]);
                assert!(w.abs_diff(v) <= 1, "winner spikes {w} vs {v}");
            }
        }
        assert!(same >= 97, "{same} of 100 winners agree");
    }

    #[test]
    fn a_blank_image_reaches_no_threshold() {
        let p = Params::default();
        let mut f = Fixed::new(&trained(&p), &p, 8);
        let run = f.present(&encode(&[0; 784], &p), |_, _, fired| {
            assert!(fired.is_empty())
        });
        assert_eq!((run.winner, run.peak), (None, 0));
    }

    #[test]
    fn register_bits_cover_the_peak() {
        assert_eq!(Fixed::register_bits(0), 2);
        assert_eq!(Fixed::register_bits(1), 2);
        assert_eq!(Fixed::register_bits(255), 9);
        assert_eq!(Fixed::register_bits(256), 10);
    }

    #[test]
    fn vectors_describe_the_run() {
        let p = Params::default();
        let mut f = Fixed::new(&trained(&p), &p, 8);
        let dir = std::env::temp_dir().join(format!("snn-vectors-{}", std::process::id()));
        fs::remove_dir_all(&dir).ok();
        f.write_config(&dir, p.time_steps).unwrap();
        let img = dataset::synthetic(1, 2);
        let tr = encode(img.image(0), &p);
        let run = f.write_image(&dir, 0, img.label(0), &tr).unwrap();
        let read = |name: &str| fs::read_to_string(dir.join(name)).unwrap();
        let (config, weights, out, trace, input) = (
            read("config.txt"),
            read("weights.txt"),
            read("0000.out"),
            read("0000.trace"),
            read("0000.in"),
        );
        fs::remove_dir_all(&dir).ok();
        assert!(config.contains("inputs 784\n") && config.contains("weight_bits 8\n"));
        assert_eq!(weights.lines().count(), 10);
        assert!(weights.lines().all(|l| l.split(' ').count() == 784));
        let winner = run.winner.map_or(-1, |w| w as i64);
        assert_eq!(out.lines().next().unwrap(), format!("winner {winner}"));
        let fired: usize = out.lines().skip(1).map(|l| l.split(' ').count() - 1).sum();
        assert_eq!(fired as u32, run.spikes.iter().sum::<u32>());
        assert_eq!(trace.lines().count(), p.time_steps);
        assert_eq!(input.lines().next().unwrap(), "label 0");
        let sent: usize = input
            .lines()
            .skip(1)
            .map(|l| l.split(' ').count() - 1)
            .sum();
        assert_eq!(sent, tr.total());
    }
}
