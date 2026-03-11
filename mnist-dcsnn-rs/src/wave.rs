//! Spike waves: the step each neuron of a layer first fired at, as in
//! SpykeTorch's cumulative spike-wave tensors, and max pooling over them.
//!
//! Maps are stored position-major: unit `(y * w + x) * maps + m`, so the maps
//! an input spike reaches at one position are contiguous.

/// Time of a neuron that never fired.
pub const NEVER: u8 = u8::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub maps: usize,
    pub h: usize,
    pub w: usize,
}

impl Shape {
    pub fn new([maps, h, w]: [usize; 3]) -> Self {
        Self { maps, h, w }
    }

    pub fn len(&self) -> usize {
        self.maps * self.h * self.w
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn unit(&self, y: usize, x: usize, m: usize) -> usize {
        (y * self.w + x) * self.maps + m
    }

    /// (row, column, map) of a unit.
    pub fn split(&self, u: usize) -> (usize, usize, usize) {
        let pos = u / self.maps;
        (pos / self.w, pos % self.w, u % self.maps)
    }
}

/// First-spike time of every neuron of a layer, and its neurons bucketed by it.
#[derive(Debug, Clone)]
pub struct Wave {
    pub shape: Shape,
    pub time: Vec<u8>,
    units: Vec<u16>,
    start: Vec<u32>,
}

impl Wave {
    pub fn new(shape: Shape) -> Self {
        assert!(shape.len() <= u16::MAX as usize + 1, "units are 16-bit");
        Self {
            shape,
            time: vec![NEVER; shape.len()],
            units: Vec::new(),
            start: Vec::new(),
        }
    }

    /// Buckets the neurons by `time` after it changed.
    pub fn index(&mut self, steps: usize) {
        self.start.clear();
        self.start.resize(steps + 1, 0);
        for &t in &self.time {
            if (t as usize) < steps {
                self.start[t as usize + 1] += 1;
            }
        }
        for t in 1..self.start.len() {
            self.start[t] += self.start[t - 1];
        }
        self.units.clear();
        self.units.resize(self.start[steps] as usize, 0);
        let mut fill = self.start.clone();
        for (u, &t) in self.time.iter().enumerate() {
            if (t as usize) < steps {
                self.units[fill[t as usize] as usize] = u as u16;
                fill[t as usize] += 1;
            }
        }
    }

    /// Neurons that first fired at step `t`, ascending.
    pub fn at(&self, t: usize) -> &[u16] {
        if t + 1 >= self.start.len() {
            return &[];
        }
        &self.units[self.start[t] as usize..self.start[t + 1] as usize]
    }

    pub fn spikes(&self) -> usize {
        self.units.len()
    }

    /// Max pooling of spike waves: each `k`×`k` window, stride `k`, fires with
    /// its earliest neuron.
    pub fn pool_into(&self, k: usize, out: &mut Wave, steps: usize) {
        let (s, o) = (self.shape, out.shape);
        debug_assert_eq!((o.maps, o.h, o.w), (s.maps, s.h / k, s.w / k));
        out.time.fill(NEVER);
        for oy in 0..o.h {
            for ox in 0..o.w {
                let dst = &mut out.time[o.unit(oy, ox, 0)..][..o.maps];
                for y in oy * k..oy * k + k {
                    for x in ox * k..ox * k + k {
                        let src = &self.time[s.unit(y, x, 0)..][..s.maps];
                        dst.iter_mut().zip(src).for_each(|(d, &t)| *d = (*d).min(t));
                    }
                }
            }
        }
        out.index(steps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units_split_back() {
        let s = Shape::new([3, 4, 5]);
        for u in 0..s.len() {
            let (y, x, m) = s.split(u);
            assert_eq!(s.unit(y, x, m), u);
        }
    }

    #[test]
    fn wave_buckets_and_pools_the_earliest() {
        let s = Shape::new([2, 4, 4]);
        let mut w = Wave::new(s);
        w.time[s.unit(0, 1, 0)] = 5;
        w.time[s.unit(1, 0, 0)] = 3;
        w.time[s.unit(3, 3, 1)] = 0;
        w.time[s.unit(2, 2, 1)] = 20;
        w.index(15);
        assert_eq!(w.at(3), &[s.unit(1, 0, 0) as u16]);
        assert_eq!(w.spikes(), 3, "past the last step is no spike");
        let mut p = Wave::new(Shape::new([2, 2, 2]));
        w.pool_into(2, &mut p, 15);
        assert_eq!(p.time, [3, NEVER, NEVER, NEVER, NEVER, NEVER, NEVER, 0]);
        assert_eq!(p.at(0), &[7]);
    }
}
