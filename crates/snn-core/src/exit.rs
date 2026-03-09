//! Answering early: what accuracy a model would have, and how many of its
//! spikes it would have spent, if each test presentation were cut short.
//! Checkpoints split a presentation into tenths and the model gives the answer
//! it has at each. A sure exit stops at the first checkpoint the model is sure
//! of: a winner that has spiked, or a readout at least [`SURE`] confident.

/// Checkpoints per presentation.
pub const POINTS: usize = 10;
/// Readout probability a model needs to exit early.
pub const SURE: f32 = 0.9;

/// What a model has by a checkpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Partial {
    pub answer: Option<u8>,
    /// spikes so far, input spikes included
    pub spikes: u64,
    /// whether it would stop here
    pub sure: bool,
}

/// Step at which checkpoint `k` (0-based) of a `steps`-long presentation falls.
pub fn checkpoint(k: usize, steps: usize) -> usize {
    ((k + 1) * steps).div_ceil(POINTS).max(1)
}

/// Totals over the test images.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Exits {
    images: u32,
    hits: [u32; POINTS],
    spikes: [u64; POINTS],
    sure_hits: u32,
    sure_spikes: u64,
    /// checkpoints reached before the sure exit, summed
    sure_points: u64,
}

impl Exits {
    pub fn record(&mut self, label: u8, at: &[Partial; POINTS]) {
        self.images += 1;
        for (k, p) in at.iter().enumerate() {
            self.hits[k] += (p.answer == Some(label)) as u32;
            self.spikes[k] += p.spikes;
        }
        let k = at.iter().position(|p| p.sure).unwrap_or(POINTS - 1);
        self.sure_hits += (at[k].answer == Some(label)) as u32;
        self.sure_spikes += at[k].spikes;
        self.sure_points += k as u64 + 1;
    }

    /// Accuracy at each checkpoint.
    pub fn accuracy(&self) -> [f32; POINTS] {
        self.hits.map(|h| h as f32 / self.images.max(1) as f32)
    }

    /// Spikes by each checkpoint as a share of the whole presentation's.
    pub fn spent(&self) -> [f32; POINTS] {
        let all = self.spikes[POINTS - 1].max(1) as f32;
        self.spikes.map(|s| s as f32 / all)
    }

    /// The sure exit's (accuracy, share of steps, share of spikes).
    pub fn sure(&self) -> (f32, f32, f32) {
        let n = self.images.max(1) as f32;
        (
            self.sure_hits as f32 / n,
            self.sure_points as f32 / n / POINTS as f32,
            self.sure_spikes as f32 / self.spikes[POINTS - 1].max(1) as f32,
        )
    }

    /// `early exit: steps 10% 20% ...`, then accuracy and spikes spent at each
    /// checkpoint, then the sure exit.
    pub fn lines(&self) -> Vec<String> {
        let pct = |v: &[f32]| -> String {
            let cells: Vec<String> = v.iter().map(|x| format!("{:.1}%", x * 100.0)).collect();
            cells.join(" ")
        };
        let steps: Vec<f32> = (1..=POINTS).map(|k| k as f32 / POINTS as f32).collect();
        let (acc, at, spent) = self.sure();
        vec![
            format!("early exit: steps {}", pct(&steps)),
            format!("early exit: accuracy {}", pct(&self.accuracy())),
            format!("early exit: spikes {}", pct(&self.spent())),
            format!(
                "early exit: sure {:.1}% at {:.1}% of the steps and {:.1}% of the spikes",
                acc * 100.0,
                at * 100.0,
                spent * 100.0
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn partials(answer_from: usize, sure_from: usize, answer: u8) -> [Partial; POINTS] {
        std::array::from_fn(|k| Partial {
            answer: (k >= answer_from).then_some(answer),
            spikes: 10 * (k as u64 + 1),
            sure: k >= sure_from,
        })
    }

    #[test]
    fn checkpoints_cover_the_presentation() {
        let steps: Vec<usize> = (0..POINTS).map(|k| checkpoint(k, 200)).collect();
        assert_eq!(steps, [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]);
        assert_eq!(checkpoint(0, 5), 1);
        assert_eq!(checkpoint(POINTS - 1, 7), 7);
    }

    #[test]
    fn accuracy_spikes_and_the_sure_exit() {
        let mut e = Exits::default();
        // right from the 3rd checkpoint, sure from the 5th
        e.record(4, &partials(2, 4, 4));
        // wrong throughout, never sure: it exits at the end
        e.record(1, &partials(0, POINTS, 7));
        assert_eq!(e.accuracy()[1], 0.0);
        assert_eq!(e.accuracy()[2], 0.5);
        assert_eq!(e.spent()[4], 0.5);
        let (acc, steps, spikes) = e.sure();
        assert_eq!(acc, 0.5);
        assert!((steps - (5.0 + 10.0) / 2.0 / 10.0).abs() < 1e-6);
        assert!((spikes - (50.0 + 100.0) / 200.0).abs() < 1e-6);
        let lines = e.lines();
        assert!(lines[0].starts_with("early exit: steps 10.0% 20.0%"));
        assert!(lines[1].ends_with("50.0% 50.0%"));
        assert_eq!(
            lines[3],
            "early exit: sure 50.0% at 75.0% of the steps and 75.0% of the spikes"
        );
    }
}
