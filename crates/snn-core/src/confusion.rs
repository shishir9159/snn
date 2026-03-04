//! Which class each image was taken for, the detail an accuracy figure hides.

/// Counts per (true class, answer). A row is a true class; its columns are the
/// classes plus one for "no answer".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Confusion {
    classes: usize,
    counts: Vec<u32>,
}

impl Confusion {
    pub fn new(classes: usize) -> Self {
        Self {
            classes,
            counts: vec![0; classes * (classes + 1)],
        }
    }

    /// Counts one image; an answer outside the classes counts as no answer.
    pub fn record(&mut self, label: u8, answer: Option<u8>) {
        let col = answer.map_or(self.classes, |a| (a as usize).min(self.classes));
        self.counts[label as usize * (self.classes + 1) + col] += 1;
    }

    pub fn classes(&self) -> usize {
        self.classes
    }

    /// Row-major, `classes + 1` columns.
    pub fn counts(&self) -> &[u32] {
        &self.counts
    }

    pub fn total(&self) -> u32 {
        self.counts.iter().sum()
    }

    pub fn accuracy(&self) -> f32 {
        let hits: u32 = (0..self.classes).map(|c| self.at(c, c)).sum();
        hits as f32 / self.total().max(1) as f32
    }

    /// Share of a class's images that were answered correctly, if it has any.
    pub fn recall(&self, class: usize) -> Option<f32> {
        let row: u32 = (0..=self.classes).map(|a| self.at(class, a)).sum();
        (row > 0).then(|| self.at(class, class) as f32 / row as f32)
    }

    fn at(&self, label: usize, answer: usize) -> u32 {
        self.counts[label * (self.classes + 1) + answer]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_answers_per_class() {
        let mut c = Confusion::new(3);
        for (label, answer) in [
            (0, Some(0)),
            (0, Some(1)),
            (1, Some(1)),
            (1, None),
            (2, Some(2)),
        ] {
            c.record(label, answer);
        }
        c.record(2, Some(9));
        assert_eq!(c.total(), 6);
        // per row: classes 0, 1, 2 and "no answer"
        let want: [u32; 12] = [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1];
        assert_eq!(c.counts(), want);
    }

    #[test]
    fn accuracy_and_recall_read_the_diagonal() {
        let mut c = Confusion::new(3);
        assert_eq!((c.accuracy(), c.recall(0)), (0.0, None));
        for (label, answer) in [
            (0, Some(0)),
            (0, Some(1)),
            (1, Some(1)),
            (1, None),
            (2, Some(2)),
        ] {
            c.record(label, answer);
        }
        assert!((c.accuracy() - 3.0 / 5.0).abs() < 1e-6);
        assert_eq!(c.recall(0), Some(0.5));
        assert_eq!(c.recall(2), Some(1.0));
    }

    #[test]
    #[should_panic]
    fn a_label_outside_the_classes_is_a_bug() {
        Confusion::new(2).record(5, None);
    }
}
