use crate::params::LifParams;

/// Leaky integrate-and-fire neuron with a hard refractory period.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neuron {
    /// membrane potential
    pub p: f32,
    /// last step of the refractory period, -1 when free
    pub t_rest: i32,
}

impl Neuron {
    pub fn new(lif: &LifParams) -> Self {
        Self {
            p: lif.rest,
            t_rest: -1,
        }
    }

    pub fn reset(&mut self, lif: &LifParams) {
        self.p = lif.rest;
        self.t_rest = -1;
    }

    pub fn is_refractory(&self, t: i32) -> bool {
        t <= self.t_rest
    }

    /// Adds `input` at step `t` unless refractory, then leaks toward rest.
    /// Returns whether the neuron integrated.
    pub fn integrate(&mut self, t: i32, input: f32, lif: &LifParams) -> bool {
        if self.is_refractory(t) {
            return false;
        }
        self.p += input;
        if self.p > lif.rest {
            self.p = (self.p - lif.leak).max(lif.rest);
        }
        if self.p < lif.floor {
            self.p = lif.floor;
        }
        true
    }

    pub fn inhibit(&mut self, lif: &LifParams) {
        self.p = lif.floor;
    }

    /// Spikes if the potential reached `threshold`: resets to rest and goes refractory.
    pub fn fire(&mut self, t: i32, threshold: f32, lif: &LifParams) -> bool {
        if self.p < threshold {
            return false;
        }
        self.p = lif.rest;
        self.t_rest = t + lif.t_ref as i32;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lif() -> LifParams {
        LifParams {
            rest: 0.0,
            floor: -10.0,
            leak: 0.5,
            t_ref: 3,
        }
    }

    #[test]
    fn integrates_and_leaks_once() {
        let l = lif();
        let mut n = Neuron::new(&l);
        assert!(n.integrate(1, 2.0, &l));
        assert_eq!(n.p, 1.5);
        n.integrate(2, 0.0, &l);
        assert_eq!(n.p, 1.0);
    }

    #[test]
    fn leak_stops_at_rest() {
        let l = lif();
        let mut n = Neuron::new(&l);
        n.integrate(1, 0.2, &l);
        assert_eq!(n.p, l.rest);
    }

    #[test]
    fn negative_input_is_not_leaked_and_floors() {
        let l = lif();
        let mut n = Neuron::new(&l);
        n.integrate(1, -3.0, &l);
        assert_eq!(n.p, -3.0);
        n.integrate(2, -30.0, &l);
        assert_eq!(n.p, l.floor);
    }

    #[test]
    fn fires_at_threshold_and_resets() {
        let l = lif();
        let mut n = Neuron::new(&l);
        n.p = 4.9;
        assert!(!n.fire(1, 5.0, &l));
        n.p = 5.0;
        assert!(n.fire(1, 5.0, &l));
        assert_eq!(n.p, l.rest);
        assert_eq!(n.t_rest, 1 + l.t_ref as i32);
    }

    #[test]
    fn refractory_blocks_integration_for_t_ref_steps() {
        let l = lif();
        let mut n = Neuron::new(&l);
        n.p = 10.0;
        assert!(n.fire(5, 1.0, &l));
        for t in 6..=8 {
            assert!(!n.integrate(t, 100.0, &l));
            assert_eq!(n.p, l.rest);
        }
        assert!(n.integrate(9, 2.0, &l));
        assert_eq!(n.p, 1.5);
    }

    #[test]
    fn inhibit_and_reset() {
        let l = lif();
        let mut n = Neuron::new(&l);
        n.inhibit(&l);
        assert_eq!(n.p, l.floor);
        n.t_rest = 40;
        n.reset(&l);
        assert_eq!(n, Neuron::new(&l));
    }
}
