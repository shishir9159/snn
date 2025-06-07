//! What every trainer reports: the image being presented, finished epochs and a
//! running hit rate.

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
