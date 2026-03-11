//! Static images as an event camera would see them during N-MNIST style
//! saccades: the padded image moves along a triangle and each pixel emits an ON
//! or OFF event whenever its log brightness has changed by a contrast step.
//!
//! Path, polarity and scale follow the real recordings: on 300 N-MNIST
//! samples of known MNIST digits, these defaults give per-leg ON and OFF
//! event maps that correlate 0.51 with the recorded ones (0.37 before) and
//! about as many events (4.2K against 4.1K).

/// The eye movement and the camera's sensitivity.
#[derive(Debug, Clone, PartialEq)]
pub struct Saccades {
    /// border added around the image, pixels
    pub pad: usize,
    /// steps (ms) per leg of the triangle
    pub leg: usize,
    /// side of the triangle, pixels
    pub amplitude: f32,
    /// log-brightness change that fires an event
    pub contrast: f32,
    /// added to brightness before the log, so black is finite
    pub floor: f32,
    /// the digit is dark on a bright screen: N-MNIST's ON events fire on the
    /// trailing edges of MNIST's bright strokes, as for an inverted digit
    pub dark: bool,
}

impl Default for Saccades {
    fn default() -> Self {
        Self {
            pad: 3,
            leg: 100,
            amplitude: 2.5,
            contrast: 0.2,
            floor: 0.1,
            dark: true,
        }
    }
}

impl Saccades {
    pub fn steps(&self) -> usize {
        3 * self.leg
    }

    /// Image offset at step `t`, y down: down-right, up-right, then left back
    /// home, the triangle of Orchard et al. 2015 (table 1) as the recordings
    /// show it.
    pub fn offset(&self, t: usize) -> (f32, f32) {
        let a = self.amplitude;
        let corners = [(0.0, 0.0), (a / 2.0, a), (a, 0.0), (0.0, 0.0)];
        let t = t.min(self.steps());
        let (leg, f) = (
            (t / self.leg).min(2),
            (t % self.leg) as f32 / self.leg as f32,
        );
        let f = if t == self.steps() { 1.0 } else { f };
        let ((x0, y0), (x1, y1)) = (corners[leg], corners[leg + 1]);
        (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)
    }
}

/// Events of one image, bucketed by step. An event id is
/// `polarity * side² + y * side + x`, polarity 0 for ON and 1 for OFF.
#[derive(Debug, Clone, Default)]
pub struct Events {
    pub side: usize,
    start: Vec<u32>,
    ids: Vec<u16>,
    /// the image inside a blank margin, so a shifted lookup never leaves it
    img: Vec<f32>,
    /// pixels that can change: their id and their index in `img`
    active: Vec<(u16, usize)>,
    /// log brightness at each active pixel's last event, and the brightness
    /// that fires its next ON or OFF event
    level: Vec<f32>,
    on: Vec<f32>,
    off: Vec<f32>,
}

impl Events {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn generate(&mut self, pixels: &[u8], w: usize, h: usize, s: &Saccades) {
        assert_eq!(pixels.len(), w * h);
        let side = w.max(h) + 2 * s.pad;
        assert!(2 * side * side <= u16::MAX as usize + 1);
        self.side = side;
        let margin = s.amplitude.ceil() as usize + 1;
        let wide = side + 2 * margin;
        self.img.clear();
        self.img.resize(wide * wide, 0.0);
        for y in 0..h {
            for x in 0..w {
                let at = (y + s.pad + margin) * wide + x + s.pad + margin;
                self.img[at] = pixels[y * w + x] as f32 / 255.0;
            }
        }
        // only pixels the moving strokes reach ever change
        let reach = s.amplitude.ceil() as usize + 1;
        let img = &self.img;
        let lit = |x: usize, y: usize| img[(y + margin) * wide + x + margin] > 0.0;
        self.active.clear();
        for y in 0..side {
            for x in 0..side {
                let (x0, y0) = (x.saturating_sub(reach), y.saturating_sub(reach));
                let near = (y0..=(y + 1).min(side - 1))
                    .any(|sy| (x0..=(x + 1).min(side - 1)).any(|sx| lit(sx, sy)));
                if near {
                    self.active
                        .push(((y * side + x) as u16, (y + margin) * wide + x + margin));
                }
            }
        }
        let shade = |v: f32| if s.dark { 1.0 - v } else { v };
        // log brightness at the start; an event fires once the brightness has
        // moved a contrast step from it, compared without taking logs
        let (c, floor) = (s.contrast, s.floor);
        let fires = |level: f32| ((level + c).exp() - floor, (level - c).exp() - floor);
        self.level.clear();
        self.on.clear();
        self.off.clear();
        for &(_, k) in &self.active {
            let level = (shade(self.img[k]) + floor).ln();
            let (on, off) = fires(level);
            self.level.push(level);
            self.on.push(on);
            self.off.push(off);
        }
        self.ids.clear();
        self.start.clear();
        self.start.push(0);
        self.start.push(0);
        for t in 1..=s.steps() {
            // every pixel sees the image at the same sub-pixel offset
            let (ox, oy) = s.offset(t);
            let (sx, sy) = ((-ox).floor(), (-oy).floor());
            let (fx, fy) = (-ox - sx, -oy - sy);
            let shift = sy as isize * wide as isize + sx as isize;
            for (a, &(id, k)) in self.active.iter().enumerate() {
                let k = k.wrapping_add_signed(shift);
                let img = &self.img;
                let top = img[k] * (1.0 - fx) + img[k + 1] * fx;
                let bottom = img[k + wide] * (1.0 - fx) + img[k + wide + 1] * fx;
                let b = shade(top * (1.0 - fy) + bottom * fy);
                let moved = if b >= self.on[a] {
                    self.ids.push(id);
                    c
                } else if b <= self.off[a] {
                    self.ids.push((side * side) as u16 + id);
                    -c
                } else {
                    continue;
                };
                self.level[a] += moved;
                (self.on[a], self.off[a]) = fires(self.level[a]);
            }
            self.start.push(self.ids.len() as u32);
        }
    }

    /// Takes a recording instead of generating events: `start` has one offset per
    /// step plus two leading ones (see [`Events::at`]), `ids` follow in step order.
    pub fn replace(&mut self, side: usize, start: &[u32], ids: &[u16]) {
        self.side = side;
        self.start.clear();
        self.start.extend_from_slice(start);
        self.ids.clear();
        self.ids.extend_from_slice(ids);
    }

    /// Moves or drops every event: `f` maps an event's step to its new one
    /// (clamped to the recording) or to None to drop it.
    pub fn retime<F: FnMut(usize) -> Option<usize>>(&mut self, mut f: F) {
        let steps = self.steps();
        if steps == 0 {
            return;
        }
        let mut moved: Vec<(usize, u16)> = Vec::with_capacity(self.ids.len());
        for t in 1..=steps {
            for &id in self.at(t) {
                if let Some(n) = f(t) {
                    moved.push((n.clamp(1, steps), id));
                }
            }
        }
        moved.sort_by_key(|&(t, _)| t);
        self.start.clear();
        self.start.resize(steps + 2, 0);
        for &(t, _) in &moved {
            self.start[t + 1] += 1;
        }
        for t in 1..self.start.len() {
            self.start[t] += self.start[t - 1];
        }
        self.ids.clear();
        self.ids.extend(moved.into_iter().map(|(_, id)| id));
    }

    pub fn steps(&self) -> usize {
        self.start.len().saturating_sub(2)
    }

    /// Events at step `t`, 1-based.
    pub fn at(&self, t: usize) -> &[u16] {
        if t + 1 >= self.start.len() {
            return &[];
        }
        &self.ids[self.start[t] as usize..self.start[t + 1] as usize]
    }

    pub fn total(&self) -> usize {
        self.ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar() -> Vec<u8> {
        let mut img = vec![0u8; 28 * 28];
        for y in 6..22 {
            for x in 12..16 {
                img[y * 28 + x] = 255;
            }
        }
        img
    }

    #[test]
    fn triangle_starts_and_ends_at_home() {
        let s = Saccades::default();
        assert_eq!(s.offset(0), (0.0, 0.0));
        assert_eq!(s.offset(100), (1.25, 2.5));
        assert_eq!(s.offset(200), (2.5, 0.0));
        assert_eq!(s.offset(300), (0.0, 0.0));
    }

    #[test]
    fn a_dark_digit_flips_the_polarity_at_each_edge() {
        // the first leg moves the image down-right: ON events lead the bar when
        // it is bright and trail it when it is dark
        let lead = |dark: bool| {
            let (mut e, s) = (
                Events::new(),
                Saccades {
                    dark,
                    ..Saccades::default()
                },
            );
            e.generate(&bar(), 28, 28, &s);
            let n = 34 * 34;
            let mut sum = [(0.0, 0.0); 2];
            for &i in (1..=100).flat_map(|t| e.at(t)) {
                let (pol, pix) = (i as usize / n, i as usize % n);
                let (along, count) = &mut sum[pol];
                // position along the motion (1.25, 2.5)
                *along += (pix % 34) as f32 + 2.0 * (pix / 34) as f32;
                *count += 1.0;
            }
            sum[0].0 / sum[0].1 - sum[1].0 / sum[1].1
        };
        assert!(lead(false) > 2.0, "{}", lead(false));
        assert!(lead(true) < -2.0, "{}", lead(true));
    }

    #[test]
    fn black_image_sends_nothing() {
        let mut e = Events::new();
        e.generate(&[0; 784], 28, 28, &Saccades::default());
        assert_eq!(e.total(), 0);
        assert_eq!(e.steps(), 300);
        assert_eq!(e.side, 34);
    }

    #[test]
    fn moving_bar_sends_balanced_on_and_off() {
        let (mut e, s) = (Events::new(), Saccades::default());
        e.generate(&bar(), 28, 28, &s);
        let n = 34 * 34;
        let all: Vec<u16> = (1..=e.steps()).flat_map(|t| e.at(t).to_vec()).collect();
        let on = all.iter().filter(|&&i| (i as usize) < n).count();
        let off = all.len() - on;
        assert!(on > 100 && off > 100, "{on} {off}");
        // the eye returns home, so every brightening is undone
        assert!((on as f32 - off as f32).abs() < 0.1 * all.len() as f32);
        // events stay near the bar's edges
        assert!(all.iter().all(|&i| (i as usize % n % 34) >= 13));
    }

    #[test]
    fn retime_keeps_the_length_and_moves_events() {
        let mut e = Events::new();
        e.replace(34, &[0, 0, 2, 2, 3], &[5, 9, 7]);
        assert_eq!((e.steps(), e.at(1), e.at(3)), (3, &[5, 9][..], &[7][..]));
        // the first event moves to the end, the second is dropped
        let mut k = 0;
        e.retime(|t| {
            k += 1;
            match k {
                1 => Some(t + 10),
                2 => None,
                _ => Some(t),
            }
        });
        assert_eq!((e.steps(), e.total()), (3, 2));
        assert!(e.at(1).is_empty());
        assert_eq!(e.at(3), &[5, 7]);
    }

    #[test]
    fn generation_is_deterministic() {
        let (mut a, mut b) = (Events::new(), Events::new());
        a.generate(&bar(), 28, 28, &Saccades::default());
        b.generate(&bar(), 28, 28, &Saccades::default());
        assert_eq!(a.ids, b.ids);
        assert_eq!(a.start, b.start);
    }
}
