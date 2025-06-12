//! Online softmax regression from a spike code to a class, the readout of the
//! unsupervised models. Papers usually fit such a classifier on spike counts
//! after training; this one learns alongside, one image at a time, and never
//! touches the network.

#[derive(Debug, Clone)]
pub struct Readout {
    classes: usize,
    n: usize,
    /// one row of `n` weights plus a bias per class
    w: Vec<f32>,
    rate: f32,
}

impl Readout {
    pub fn new(n: usize, classes: usize, rate: f32) -> Self {
        Self {
            classes,
            n,
            w: vec![0.0; classes * (n + 1)],
            rate,
        }
    }

    /// Class probabilities for a code.
    pub fn probs(&self, x: &[f32], out: &mut Vec<f32>) {
        assert_eq!(x.len(), self.n);
        out.clear();
        for row in self.w.chunks_exact(self.n + 1) {
            let z: f32 = row[..self.n].iter().zip(x).map(|(w, v)| w * v).sum();
            out.push(z + row[self.n]);
        }
        let max = out.iter().copied().fold(f32::MIN, f32::max);
        out.iter_mut().for_each(|z| *z = (*z - max).exp());
        let sum: f32 = out.iter().sum();
        out.iter_mut().for_each(|z| *z /= sum);
    }

    /// One gradient step of the log loss toward `label`, given its probabilities.
    pub fn learn(&mut self, x: &[f32], probs: &[f32], label: u8) {
        for (c, row) in self.w.chunks_exact_mut(self.n + 1).enumerate() {
            let g = self.rate * ((c == label as usize) as u8 as f32 - probs[c]);
            row[..self.n]
                .iter_mut()
                .zip(x)
                .for_each(|(w, &v)| *w += g * v);
            row[self.n] += g;
        }
    }

    pub fn classes(&self) -> usize {
        self.classes
    }

    /// Class that feature `k` speaks for most, if it speaks for any.
    pub fn favourite(&self, k: usize) -> Option<u8> {
        let w = (0..self.classes).map(|c| self.w[c * (self.n + 1) + k]);
        let (c, v) = w.enumerate().max_by(|a, b| a.1.total_cmp(&b.1))?;
        (v > 0.0).then_some(c as u8)
    }
}

/// Index of the largest value.
pub fn argmax(v: &[f32]) -> Option<u8> {
    let (i, _) = v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
    Some(i as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untrained_readout_is_uniform() {
        let r = Readout::new(3, 4, 0.1);
        let mut p = Vec::new();
        r.probs(&[1.0, 2.0, 3.0], &mut p);
        assert!(p.iter().all(|&x| (x - 0.25).abs() < 1e-6));
    }

    #[test]
    fn separates_two_codes() {
        let mut r = Readout::new(2, 2, 0.5);
        let (a, b) = ([1.0, 0.0], [0.0, 1.0]);
        let mut p = Vec::new();
        for _ in 0..50 {
            r.probs(&a, &mut p);
            r.learn(&a, &p.clone(), 0);
            r.probs(&b, &mut p);
            r.learn(&b, &p.clone(), 1);
        }
        r.probs(&a, &mut p);
        assert_eq!(argmax(&p), Some(0));
        assert!(p[0] > 0.9);
        r.probs(&b, &mut p);
        assert_eq!(argmax(&p), Some(1));
    }
}
