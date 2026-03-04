//! What every trainer reports: the image being presented, finished epochs and a
//! running hit rate.

use crate::confusion::Confusion;
use crate::energy::{Activity, PerImage};
use std::collections::VecDeque;

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
pub struct EpochResult {
    pub epoch: usize,
    pub train_accuracy: f32,
    pub test_accuracy: Option<f32>,
    pub images_per_sec: f32,
    /// per training image of the epoch
    pub activity: PerImage,
    /// the test pass, if there was one
    pub confusion: Option<Confusion>,
}

/// What one pass over a dataset adds up to.
#[derive(Debug, Clone)]
pub struct Pass {
    pub images: usize,
    pub hits: usize,
    pub activity: Activity,
    pub confusion: Confusion,
}

impl Pass {
    pub fn new(classes: usize) -> Self {
        Self {
            images: 0,
            hits: 0,
            activity: Activity::default(),
            confusion: Confusion::new(classes),
        }
    }

    /// Adds an image with its true class and the class the model answered.
    pub fn record(&mut self, label: u8, answer: Option<u8>, activity: Activity) {
        self.images += 1;
        self.hits += (answer == Some(label)) as usize;
        self.activity += activity;
        self.confusion.record(label, answer);
    }

    pub fn accuracy(&self) -> f32 {
        self.hits as f32 / self.images.max(1) as f32
    }

    pub fn per_image(&self) -> PerImage {
        self.activity.per_image(self.images)
    }
}

/// Hit rate over the most recent `cap` results.
#[derive(Debug, Clone)]
pub struct Recent {
    cap: usize,
    window: VecDeque<bool>,
    hits: usize,
}

impl Recent {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            window: VecDeque::with_capacity(cap),
            hits: 0,
        }
    }

    pub fn push(&mut self, hit: bool) {
        if self.window.len() == self.cap {
            self.hits -= self.window.pop_front().unwrap() as usize;
        }
        self.window.push_back(hit);
        self.hits += hit as usize;
    }

    pub fn rate(&self) -> f32 {
        self.hits as f32 / self.window.len().max(1) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_adds_up_hits_and_activity() {
        let mut p = Pass::new(3);
        assert_eq!(p.accuracy(), 0.0);
        p.record(1, Some(1), Activity::new(10, 100));
        p.record(2, Some(0), Activity::new(30, 300));
        p.record(0, Some(0), Activity::new(20, 200));
        assert!((p.accuracy() - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(p.confusion.accuracy(), p.accuracy());
        assert_eq!(p.confusion.recall(2), Some(0.0));
        assert_eq!(
            p.per_image(),
            PerImage {
                spikes: 20.0,
                synops: 200.0
            }
        );
    }

    #[test]
    fn recent_forgets_old_results() {
        let mut r = Recent::new(3);
        assert_eq!(r.rate(), 0.0);
        for hit in [true, true, false, false, false] {
            r.push(hit);
        }
        assert_eq!(r.rate(), 0.0);
        r.push(true);
        assert!((r.rate() - 1.0 / 3.0).abs() < 1e-6);
    }
}
