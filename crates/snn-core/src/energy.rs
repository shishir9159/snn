//! What an image costs an event-driven chip: spikes emitted and the synaptic
//! operations they trigger (a spike reaching one synapse is one operation).
//! These counts, not the arithmetic a CPU simulation does, are the usual
//! proxy for energy, since silent neurons and idle synapses cost nothing.

use std::ops::AddAssign;

/// Totals for one presentation or a run of them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Activity {
    pub spikes: u64,
    pub synops: u64,
}

/// Mean activity per image.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PerImage {
    pub spikes: f32,
    pub synops: f32,
}

impl Activity {
    pub fn new(spikes: u64, synops: u64) -> Self {
        Self { spikes, synops }
    }

    /// `spikes` that each reach `fan_out` synapses.
    pub fn fan(spikes: u64, fan_out: u64) -> Self {
        Self::new(spikes, spikes * fan_out)
    }

    pub fn per_image(self, images: usize) -> PerImage {
        let n = images.max(1) as f64;
        PerImage {
            spikes: (self.spikes as f64 / n) as f32,
            synops: (self.synops as f64 / n) as f32,
        }
    }
}

impl AddAssign for Activity {
    fn add_assign(&mut self, o: Self) {
        self.spikes += o.spikes;
        self.synops += o.synops;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fan_out_multiplies_spikes() {
        assert_eq!(Activity::fan(3, 40), Activity::new(3, 120));
        let mut a = Activity::fan(1, 10);
        a += Activity::new(2, 5);
        assert_eq!(a, Activity::new(3, 15));
    }

    #[test]
    fn means_divide_by_images() {
        let a = Activity::new(10, 1_000);
        assert_eq!(
            a.per_image(4),
            PerImage {
                spikes: 2.5,
                synops: 250.0
            }
        );
        assert_eq!(
            a.per_image(0).spikes,
            10.0,
            "no images is not a division by zero"
        );
    }

    #[test]
    fn compact_counts() {
        let got: Vec<String> = [0.0, 950.0, 1_234.0, 56_000.0, 7.8e6, 2e9]
            .iter()
            .map(|&n| compact(n))
            .collect();
        assert_eq!(got, ["0", "950", "1.2K", "56K", "7.8M", "2.0G"]);
    }
}
