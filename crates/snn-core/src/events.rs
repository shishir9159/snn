//! Static images as an event camera would see them during N-MNIST style
//! saccades: the padded image moves along a triangle and each pixel emits an ON
//! or OFF event whenever its log brightness has changed by a contrast step.

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
}

impl Default for Saccades {
    fn default() -> Self {
        Self {
            pad: 3,
            leg: 100,
            amplitude: 3.0,
            contrast: 0.15,
            floor: 0.1,
        }
    }
}

impl Saccades {
    pub fn steps(&self) -> usize {
        3 * self.leg
    }

    /// Image offset at step `t`: right, then down-left, then back home.
    pub fn offset(&self, t: usize) -> (f32, f32) {
        let a = self.amplitude;
        let corners = [(0.0, 0.0), (a, 0.0), (a / 2.0, a), (0.0, 0.0)];
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
    img: Vec<f32>,
    level: Vec<f32>,
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
        self.img.clear();
        self.img.resize(side * side, 0.0);
        for y in 0..h {
            for x in 0..w {
                self.img[(y + s.pad) * side + x + s.pad] = pixels[y * w + x] as f32 / 255.0;
            }
        }
        // only pixels the moving image can light ever change
        let reach = s.amplitude.ceil() as i32 + 1;
        let lit = |x: i32, y: i32| {
            x >= 0 && y >= 0 && (x as usize) < side && (y as usize) < side && {
                self.img[y as usize * side + x as usize] > 0.0
            }
        };
        let active: Vec<usize> = (0..side * side)
            .filter(|&i| {
                let (x, y) = ((i % side) as i32, (i / side) as i32);
                (x - reach..=x + 1).any(|sx| (y - reach..=y + 1).any(|sy| lit(sx, sy)))
            })
            .collect();
        // brightness each active pixel saw last, its log, and the log at its last event
        let mut seen: Vec<f32> = active
            .iter()
            .map(|&i| self.bright_at(i, (0.0, 0.0)))
            .collect();
        let mut log: Vec<f32> = seen.iter().map(|&b| (b + s.floor).ln()).collect();
        let mut level = std::mem::take(&mut self.level);
        level.clear();
        level.extend_from_slice(&log);
        self.ids.clear();
        self.start.clear();
        self.start.push(0);
        self.start.push(0);
        for t in 1..=s.steps() {
            let off = s.offset(t);
            for (a, &i) in active.iter().enumerate() {
                let b = self.bright_at(i, off);
                if b != seen[a] {
                    seen[a] = b;
                    log[a] = (b + s.floor).ln();
                }
                let d = log[a] - level[a];
                if d >= s.contrast {
                    level[a] += s.contrast;
                    self.ids.push(i as u16);
                } else if d <= -s.contrast {
                    level[a] -= s.contrast;
                    self.ids.push((side * side + i) as u16);
                }
            }
            self.start.push(self.ids.len() as u32);
        }
        self.level = level;
    }

    // brightness pixel i sees with the image shifted by `off`, bilinear
    fn bright_at(&self, i: usize, off: (f32, f32)) -> f32 {
        let side = self.side;
        let (x, y) = ((i % side) as f32 - off.0, (i / side) as f32 - off.1);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let at = |xx: f32, yy: f32| {
            if xx < 0.0 || yy < 0.0 || xx >= side as f32 || yy >= side as f32 {
                0.0
            } else {
                self.img[yy as usize * side + xx as usize]
            }
        };
        let top = at(x0, y0) * (1.0 - fx) + at(x0 + 1.0, y0) * fx;
        let bottom = at(x0, y0 + 1.0) * (1.0 - fx) + at(x0 + 1.0, y0 + 1.0) * fx;
        top * (1.0 - fy) + bottom * fy
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
        assert_eq!(s.offset(100), (3.0, 0.0));
        assert_eq!(s.offset(200), (1.5, 3.0));
        assert_eq!(s.offset(300), (0.0, 0.0));
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
