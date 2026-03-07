//! What an image costs an event-driven chip: spikes emitted and the synaptic
//! operations they trigger (a spike reaching one synapse is one operation).
//! These counts, not the arithmetic a CPU simulation does, are the usual
//! proxy for energy, since silent neurons and idle synapses cost nothing.
//!
//! A spike only adds its weight to the target (an accumulate); a graded value
//! sent through a synapse is multiplied first (a multiply-accumulate). Energy
//! prices both per operation. Neuron updates and memory traffic, which a chip
//! also pays for, are left out, as in most SNN energy estimates.

use std::ops::AddAssign;

/// Totals for one presentation or a run of them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Activity {
    pub spikes: u64,
    /// spike deliveries, one accumulate each
    pub synops: u64,
    /// multiply-accumulates on graded values
    pub macs: u64,
}

/// Mean activity per image.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PerImage {
    pub spikes: f32,
    pub synops: f32,
    pub macs: f32,
}

impl Activity {
    pub fn new(spikes: u64, synops: u64) -> Self {
        Self {
            spikes,
            synops,
            macs: 0,
        }
    }

    /// `spikes` that each reach `fan_out` synapses.
    pub fn fan(spikes: u64, fan_out: u64) -> Self {
        Self::new(spikes, spikes * fan_out)
    }

    pub fn with_macs(self, macs: u64) -> Self {
        Self { macs, ..self }
    }

    pub fn per_image(self, images: usize) -> PerImage {
        let n = images.max(1) as f64;
        PerImage {
            spikes: (self.spikes as f64 / n) as f32,
            synops: (self.synops as f64 / n) as f32,
            macs: (self.macs as f64 / n) as f32,
        }
    }
}

impl AddAssign for Activity {
    fn add_assign(&mut self, o: Self) {
        self.spikes += o.spikes;
        self.synops += o.synops;
        self.macs += o.macs;
    }
}

/// Picojoules per operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Costs {
    pub ac: f32,
    pub mac: f32,
}

impl Costs {
    /// 32-bit floats in 45 nm CMOS (Horowitz, ISSCC 2014): an add 0.9 pJ, a
    /// multiply 3.7 pJ.
    pub const FP32: Costs = Costs { ac: 0.9, mac: 4.6 };
    /// 8-bit integers, same source: an add 0.03 pJ, a multiply 0.2 pJ.
    pub const INT8: Costs = Costs {
        ac: 0.03,
        mac: 0.23,
    };

    /// Nanojoules for `a`'s operations.
    pub fn nanojoules(&self, a: PerImage) -> f32 {
        (a.synops * self.ac + a.macs * self.mac) / 1e3
    }

    /// Nanojoules a dense ANN with the same layers spends per image: every
    /// weight is one MAC, whatever the input.
    pub fn dense(&self, macs: u64) -> f32 {
        macs as f32 * self.mac / 1e3
    }
}

/// 12.3K, 4.5M: three significant digits for a count.
pub fn compact(n: f32) -> String {
    match n {
        n if n >= 1e9 => format!("{:.1}G", n / 1e9),
        n if n >= 1e6 => format!("{:.1}M", n / 1e6),
        n if n >= 1e4 => format!("{:.0}K", n / 1e3),
        n if n >= 1e3 => format!("{:.1}K", n / 1e3),
        n => format!("{n:.0}"),
    }
}

/// 830 nJ, 12 µJ, 1.5 mJ, from nanojoules.
pub fn joules(nj: f32) -> String {
    let (v, unit) = match nj {
        n if n >= 1e6 => (n / 1e6, "mJ"),
        n if n >= 1e3 => (n / 1e3, "µJ"),
        n => (n, "nJ"),
    };
    if v >= 10.0 {
        format!("{v:.0} {unit}")
    } else {
        format!("{v:.2} {unit}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fan_out_multiplies_spikes() {
        assert_eq!(Activity::fan(3, 40), Activity::new(3, 120));
        let mut a = Activity::fan(1, 10);
        a += Activity::new(2, 5).with_macs(7);
        assert_eq!(a, Activity::new(3, 15).with_macs(7));
    }

    #[test]
    fn means_divide_by_images() {
        let a = Activity::new(10, 1_000).with_macs(40);
        assert_eq!(
            a.per_image(4),
            PerImage {
                spikes: 2.5,
                synops: 250.0,
                macs: 10.0
            }
        );
        assert_eq!(
            a.per_image(0).spikes,
            10.0,
            "no images is not a division by zero"
        );
    }

    #[test]
    fn energy_prices_accumulates_and_macs() {
        let a = Activity::new(5, 1_000).with_macs(100).per_image(1);
        let nj = Costs::FP32.nanojoules(a);
        assert!((nj - (1_000.0 * 0.9 + 100.0 * 4.6) / 1e3).abs() < 1e-6);
        // a dense layer of 784 x 100 is 78 400 MACs
        assert!((Costs::FP32.dense(78_400) - 360.64).abs() < 1e-3);
        assert!(Costs::INT8.nanojoules(a) < nj / 10.0);
    }

    #[test]
    fn compact_counts() {
        let got: Vec<String> = [0.0, 950.0, 1_234.0, 56_000.0, 7.8e6, 2e9]
            .iter()
            .map(|&n| compact(n))
            .collect();
        assert_eq!(got, ["0", "950", "1.2K", "56K", "7.8M", "2.0G"]);
    }

    #[test]
    fn joules_pick_a_unit() {
        let got: Vec<String> = [3.456, 830.0, 1_500.0, 12_000.0, 1.5e6]
            .iter()
            .map(|&n| joules(n))
            .collect();
        assert_eq!(got, ["3.46 nJ", "830 nJ", "1.50 µJ", "12 µJ", "1.50 mJ"]);
    }
}
