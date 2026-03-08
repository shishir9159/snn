//! Damage to test inputs, for robustness curves: Gaussian noise on the pixels
//! before encoding, dropped spikes or events, and jittered spike times.

use crate::encode::SpikeTrain;
use crate::events::Events;
use crate::rng::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Noise,
    Dropout,
    Jitter,
}

impl Kind {
    pub const ALL: [Kind; 3] = [Kind::Noise, Kind::Dropout, Kind::Jitter];

    /// Levels a curve runs over: noise standard deviation as a share of full
    /// brightness, share of spikes dropped, jitter standard deviation in steps.
    pub fn levels(self) -> &'static [f32] {
        match self {
            Kind::Noise => &[0.1, 0.2, 0.3, 0.5],
            Kind::Dropout => &[0.1, 0.3, 0.5, 0.7],
            Kind::Jitter => &[1.0, 3.0, 10.0, 30.0],
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Kind::Noise => "pixel noise",
            Kind::Dropout => "event dropout",
            Kind::Jitter => "timing jitter",
        }
    }
}

/// One kind of damage at one level, with its own random stream.
#[derive(Debug, Clone)]
pub struct Perturb {
    pub kind: Kind,
    pub level: f32,
    rng: Rng,
}

impl Perturb {
    pub fn new(kind: Kind, level: f32, seed: u64) -> Self {
        Self {
            kind,
            level,
            rng: Rng::new(seed),
        }
    }

    /// Writes `pixels` plus noise into `out` and returns true, if this is
    /// pixel noise; otherwise leaves `out` alone.
    pub fn pixels(&mut self, pixels: &[u8], out: &mut Vec<u8>) -> bool {
        if self.kind != Kind::Noise {
            return false;
        }
        let sd = self.level * 255.0;
        out.clear();
        out.extend(pixels.iter().map(|&p| {
            (p as f32 + sd * self.rng.normal())
                .round()
                .clamp(0.0, 255.0) as u8
        }));
        true
    }

    // where a spike at step `t` goes, or None if it is dropped
    fn moved(&mut self, t: usize) -> Option<usize> {
        match self.kind {
            Kind::Noise => Some(t),
            Kind::Dropout => (self.rng.f32() >= self.level).then_some(t),
            Kind::Jitter => {
                let to = t as f32 + self.level * self.rng.normal();
                Some(to.round().max(1.0) as usize)
            }
        }
    }

    pub fn train(&mut self, tr: &mut SpikeTrain) {
        if self.kind != Kind::Noise {
            tr.retime(|t| self.moved(t as usize));
        }
    }

    pub fn events(&mut self, ev: &mut Events) {
        if self.kind != Kind::Noise {
            ev.retime(|t| self.moved(t));
        }
    }

    /// A latency code: one spike time per input from 0, `t_max` for none.
    pub fn times(&mut self, times: &mut [f32], t_max: f32) {
        for t in times.iter_mut().filter(|t| **t < t_max) {
            match self.kind {
                Kind::Noise => {}
                Kind::Dropout if self.rng.f32() < self.level => *t = t_max,
                Kind::Dropout => {}
                Kind::Jitter => {
                    let to = *t + self.level * self.rng.normal();
                    *t = to.round().clamp(0.0, t_max - 1.0);
                }
            }
        }
    }
}

/// Test accuracy clean and at each level of each kind of damage.
#[derive(Debug, Clone, PartialEq)]
pub struct Curves {
    pub clean: f32,
    pub curves: Vec<(Kind, Vec<(f32, f32)>)>,
}

/// Runs `eval` without damage and then under every kind and level; `eval`
/// returns the accuracy, or None for damage the input cannot take (pixel noise
/// on a recording).
pub fn sweep<F>(seed: u64, mut eval: F) -> Curves
where
    F: FnMut(Option<Perturb>) -> Option<f32>,
{
    let clean = eval(None).unwrap_or(0.0);
    let mut curves = Vec::new();
    for kind in Kind::ALL {
        let mut points = Vec::new();
        for &level in kind.levels() {
            match eval(Some(Perturb::new(kind, level, seed))) {
                Some(acc) => points.push((level, acc)),
                None => break,
            }
        }
        if !points.is_empty() {
            curves.push((kind, points));
        }
    }
    Curves { clean, curves }
}

impl Curves {
    /// `robustness: clean 71.3%`, then e.g.
    /// `robustness: event dropout 0.1 70.2%  0.3 66.0% ...` per curve.
    pub fn lines(&self) -> Vec<String> {
        let mut out = vec![format!("robustness: clean {:.1}%", self.clean * 100.0)];
        for (kind, points) in &self.curves {
            let cells: Vec<String> = points
                .iter()
                .map(|(level, acc)| format!("{level} {:.1}%", acc * 100.0))
                .collect();
            out.push(format!("robustness: {} {}", kind.name(), cells.join("  ")));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::EncodeParams;

    fn train() -> SpikeTrain {
        let mut tr = SpikeTrain::new();
        tr.encode(
            &[10.0; 200],
            200,
            &EncodeParams::default(),
            &mut Rng::new(0),
        );
        tr
    }

    #[test]
    fn noise_spreads_pixels_and_only_noise_touches_them() {
        let (img, mut out) = (vec![128u8; 4000], Vec::new());
        assert!(!Perturb::new(Kind::Dropout, 0.5, 1).pixels(&img, &mut out));
        assert!(Perturb::new(Kind::Noise, 0.2, 1).pixels(&img, &mut out));
        let mean = out.iter().map(|&p| p as f32).sum::<f32>() / 4000.0;
        let sd = (out.iter().map(|&p| (p as f32 - mean).powi(2)).sum::<f32>() / 4000.0).sqrt();
        assert!(
            (mean - 128.0).abs() < 2.0 && (sd - 51.0).abs() < 3.0,
            "{mean} {sd}"
        );
    }

    #[test]
    fn dropout_drops_its_share_and_jitter_keeps_every_spike() {
        let full = train().total();
        let mut tr = train();
        Perturb::new(Kind::Dropout, 0.3, 2).train(&mut tr);
        let kept = tr.total() as f32 / full as f32;
        assert!((kept - 0.7).abs() < 0.03, "{kept}");
        let mut tr = train();
        Perturb::new(Kind::Jitter, 5.0, 3).train(&mut tr);
        assert_eq!(tr.total(), full);
        assert_ne!(tr.times(0), train().times(0));
        let mut tr = train();
        Perturb::new(Kind::Noise, 0.5, 3).train(&mut tr);
        assert_eq!(tr.times(7), train().times(7), "noise is for pixels only");
    }

    #[test]
    fn latency_codes_lose_or_shift_spikes_within_range() {
        let mut t = vec![0.0, 100.0, 255.0, 256.0];
        Perturb::new(Kind::Dropout, 1.0, 1).times(&mut t, 256.0);
        assert_eq!(t, [256.0; 4]);
        let mut t = vec![0.0, 100.0, 255.0, 256.0];
        Perturb::new(Kind::Jitter, 50.0, 2).times(&mut t, 256.0);
        assert!(t[..3].iter().all(|&x| (0.0..=255.0).contains(&x)));
        assert_eq!(t[3], 256.0, "a silent input stays silent");
    }

    #[test]
    fn sweep_visits_every_level_and_skips_what_does_not_apply() {
        let mut seen = Vec::new();
        let c = sweep(1, |d| {
            let (kind, level) = d.map_or((None, 0.0), |d| (Some(d.kind), d.level));
            seen.push(kind);
            (kind != Some(Kind::Noise)).then_some(1.0 - level / 100.0)
        });
        assert_eq!(c.clean, 1.0);
        assert_eq!(c.curves.len(), 2, "pixel noise does not apply");
        assert_eq!(c.curves[0].1[0], (0.1, 0.999));
        assert_eq!(seen.len(), 1 + 1 + 4 + 4);
        let lines = c.lines();
        assert_eq!(lines[0], "robustness: clean 100.0%");
        assert!(lines[2].starts_with("robustness: timing jitter 1 99.0%  3 97.0%"));
    }
}
