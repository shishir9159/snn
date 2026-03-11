//! Constants of the deep convolutional spiking network of Mozafari et al.,
//! "Bio-inspired digit recognition using reward-modulated spike-timing-dependent
//! plasticity in deep convolutional networks" (Pattern Recognition 2019,
//! arXiv 1804.00227), as set in its SpykeTorch reimplementation (MozafariDeep.py).
//! The feature layers learn as in Kheradpisheh et al., "STDP-based spiking
//! deep convolutional neural networks for object recognition" (Neural Networks 2018).

/// One convolutional layer and the pooling after it.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub maps: usize,
    /// square window over the input maps
    pub kernel: usize,
    /// zeros around the input maps
    pub pad: usize,
    /// a neuron fires when its potential goes above this
    pub threshold: f32,
    /// neurons that learn per image, and the square radius each one silences
    pub winners: usize,
    pub radius: usize,
    /// max-pooling window and stride over its spikes, 1 for none
    pub pool: usize,
    /// epochs of STDP on this layer alone
    pub epochs: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// time steps of the rank-order code
    pub steps: usize,
    /// filter responses below this are dropped (pixels are 0..255)
    pub dog_threshold: f32,
    /// radius of the local mean each response is divided by
    pub norm_radius: usize,
    pub l1: Layer,
    pub l2: Layer,
    /// decision layer: no threshold, the largest final potential answers; its
    /// maps are `per_class` times the classes
    pub l3_kernel: usize,
    pub l3_pad: usize,
    pub per_class: usize,
    /// initial weights are normal around this
    pub init_mean: f32,
    pub init_sd: f32,
    /// STDP of the feature layers, before the schedule raises them
    pub ltp: f32,
    pub ltd: f32,
    /// the rates double every this many images of a layer's training
    pub rate_every: usize,
    pub rate_max: f32,
    /// R-STDP of the decision layer: after a right answer, pre-before-post
    /// synapses grow and the rest shrink
    pub reward_ltp: f32,
    pub reward_ltd: f32,
    /// after a wrong one, pre-before-post synapses shrink and the rest grow
    pub punish_ltp: f32,
    pub punish_ltd: f32,
    /// decision weights stay in this range
    pub w_min: f32,
    pub w_max: f32,
    /// images after which reward is scaled by the miss rate and punishment by
    /// the hit rate
    pub batch: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            steps: 15,
            dog_threshold: 50.0,
            norm_radius: 8,
            l1: Layer {
                maps: 30,
                kernel: 5,
                pad: 2,
                threshold: 15.0,
                winners: 5,
                radius: 3,
                pool: 2,
                epochs: 2,
            },
            l2: Layer {
                maps: 250,
                kernel: 3,
                pad: 1,
                threshold: 10.0,
                winners: 8,
                radius: 1,
                pool: 3,
                epochs: 4,
            },
            l3_kernel: 5,
            l3_pad: 2,
            per_class: 20,
            init_mean: 0.8,
            init_sd: 0.05,
            ltp: 0.004,
            ltd: 0.003,
            rate_every: 500,
            rate_max: 0.15,
            reward_ltp: 0.004,
            reward_ltd: 0.003,
            punish_ltp: 0.004,
            punish_ltd: 0.0005,
            w_min: 0.2,
            w_max: 0.8,
            batch: 1000,
        }
    }
}

/// Phase of the layer-wise training an image belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// STDP on the first convolutional layer
    Layer1,
    /// STDP on the second, the first fixed
    Layer2,
    /// R-STDP on the decision layer, the features fixed
    Decision,
}

impl Phase {
    pub fn index(self) -> usize {
        self as usize
    }
}

impl Params {
    /// Epochs of feature learning before the decision layer trains.
    pub fn feature_epochs(&self) -> usize {
        self.l1.epochs + self.l2.epochs
    }

    pub fn phase(&self, epoch: usize) -> Phase {
        if epoch < self.l1.epochs {
            Phase::Layer1
        } else if epoch < self.feature_epochs() {
            Phase::Layer2
        } else {
            Phase::Decision
        }
    }

    /// Map shapes (maps, height, width) of S1, L1, L2 and L3 for images of
    /// `width` by `height`, or why they do not fit.
    pub fn shapes(
        &self,
        width: usize,
        height: usize,
        classes: usize,
    ) -> Result<[[usize; 3]; 4], String> {
        if self.steps == 0 || self.steps >= u8::MAX as usize {
            return Err(format!("steps must be 1..{}", u8::MAX - 1));
        }
        let conv = |name: &str, (kernel, pad): (usize, usize), h: usize, w: usize| {
            let (h, w) = (h + 2 * pad, w + 2 * pad);
            if kernel == 0 || kernel > h || kernel > w {
                return Err(format!("{name}: a {kernel} window does not fit {h}x{w}"));
            }
            Ok((h - kernel + 1, w - kernel + 1))
        };
        let pool = |name: &str, k: usize, (h, w): (usize, usize)| {
            if k == 0 || k > h || k > w {
                return Err(format!("{name}: pooling {k} does not fit {h}x{w}"));
            }
            Ok((h / k, w / k))
        };
        let (l1, l2) = (&self.l1, &self.l2);
        let s1 = [crate::input::DOG.len(), height, width];
        let (h1, w1) = conv("l1", (l1.kernel, l1.pad), height, width)?;
        let (p1h, p1w) = pool("l1", l1.pool, (h1, w1))?;
        let (h2, w2) = conv("l2", (l2.kernel, l2.pad), p1h, p1w)?;
        let (p2h, p2w) = pool("l2", l2.pool, (h2, w2))?;
        let (h3, w3) = conv("l3", (self.l3_kernel, self.l3_pad), p2h, p2w)?;
        let shapes = [
            s1,
            [self.l1.maps, h1, w1],
            [self.l2.maps, h2, w2],
            [classes * self.per_class, h3, w3],
        ];
        // spikes reach the dashboard as 16-bit unit numbers
        for s in &shapes {
            let n = s[0] * s[1] * s[2];
            if n == 0 || n > u16::MAX as usize + 1 {
                return Err(format!("a layer of {n} neurons; 1..=65536 fit"));
            }
        }
        Ok(shapes)
    }
}

snn_core::tunable!(Params {
    "steps" => steps,
    "dog_threshold" => dog_threshold,
    "norm_radius" => norm_radius,
    "l1_maps" => l1.maps,
    "l1_kernel" => l1.kernel,
    "l1_pad" => l1.pad,
    "l1_threshold" => l1.threshold,
    "l1_winners" => l1.winners,
    "l1_radius" => l1.radius,
    "l1_pool" => l1.pool,
    "l1_epochs" => l1.epochs,
    "l2_maps" => l2.maps,
    "l2_kernel" => l2.kernel,
    "l2_pad" => l2.pad,
    "l2_threshold" => l2.threshold,
    "l2_winners" => l2.winners,
    "l2_radius" => l2.radius,
    "l2_pool" => l2.pool,
    "l2_epochs" => l2.epochs,
    "l3_kernel" => l3_kernel,
    "l3_pad" => l3_pad,
    "per_class" => per_class,
    "init_mean" => init_mean,
    "init_sd" => init_sd,
    "ltp" => ltp,
    "ltd" => ltd,
    "rate_every" => rate_every,
    "rate_max" => rate_max,
    "reward_ltp" => reward_ltp,
    "reward_ltd" => reward_ltd,
    "punish_ltp" => punish_ltp,
    "punish_ltd" => punish_ltd,
    "w_min" => w_min,
    "w_max" => w_max,
    "batch" => batch,
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_name_reaches_a_field() {
        let mut names = Params::KEYS.to_vec();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), Params::KEYS.len(), "a name appears twice");
        for key in Params::KEYS {
            Params::default().set(key, "1").unwrap();
        }
        let mut p = Params::default();
        p.set("l2_maps", "64").unwrap();
        p.set("l1_threshold", "12.5").unwrap();
        assert_eq!((p.l2.maps, p.l1.threshold), (64, 12.5));
        assert!(p.set("l1_epochs", "-1").is_err());
    }

    #[test]
    fn mnist_shapes_match_the_paper() {
        let s = Params::default().shapes(28, 28, 10).unwrap();
        assert_eq!(s, [[6, 28, 28], [30, 28, 28], [250, 14, 14], [200, 4, 4]]);
        let p = Params {
            per_class: 500,
            ..Params::default()
        };
        assert!(p.shapes(28, 28, 10).unwrap_err().contains("80000"));
        let mut p = Params::default();
        p.l1.kernel = 40;
        assert!(p.shapes(28, 28, 10).unwrap_err().starts_with("l1:"));
    }

    #[test]
    fn epochs_split_into_phases() {
        let p = Params::default();
        let phases: Vec<Phase> = (0..8).map(|e| p.phase(e)).collect();
        use Phase::*;
        assert_eq!(
            phases,
            [Layer1, Layer1, Layer2, Layer2, Layer2, Layer2, Decision, Decision]
        );
        assert_eq!(p.feature_epochs(), 6);
    }
}
