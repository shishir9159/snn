const D: f64 = 0.5;
const PTH: f64 = 140.0;
const PREF: f64 = 0.0;
const PREST: f64 = 0.0;
const PMIN: f64 = -1.0;

#[derive(Debug, Clone)]
pub struct Neuron {
    pub p: f64,
    pub d: f64,
    pub pth: f64,
    pub pmin: f64,
    pub prest: f64,
    pub t_ref: i32,
    pub t_rest: i32,
}

impl Neuron {
    pub fn new() -> Self {
        Neuron {
            pth: PTH,
            t_ref: 4,
            t_rest: -1,
            p: PREST,
            d: D,
            pmin: PMIN,
            prest: PREST,
        }
    }

    pub fn new_with_params(pth: f64, t_ref: i32, d: f64, pmin: f64, prest: f64) -> Self {
        Neuron {
            pth,
            t_ref,
            t_rest: -1,
            p: prest,
            d,
            pmin,
            prest,
        }
    }

    pub fn check(&mut self) -> i32 {
        if self.p >= self.pth {
            self.p = PREF;
            1
        } else if self.p < self.pmin {
            self.p = self.prest;
            0
        } else {
            0
        }
    }

    pub fn inhibit(&mut self) {
        self.p = self.pmin;
    }

    pub fn initial(&mut self) {
        self.t_rest = -1;
        self.p = self.prest;
    }

    pub fn update_potential(&mut self, input: f64) {
        if self.t_rest < 0 || self.p > self.pmin {
            self.p += input;
            if self.p > self.prest {
                self.p -= self.d;
            }
        }
    }

    pub fn simulate(&mut self, spike_train: &[f64], time_steps: usize) -> Vec<i32> {
        let mut spikes = Vec::new();
        
        for t in 0..time_steps {

            if self.t_rest < t as i32 {
                self.update_potential(spike_train[t]);
            }
            
            let spike = self.check();
            if spike == 1 {
                self.t_rest = t as i32 + self.t_ref;
            }
            
            spikes.push(spike);
        }
        
        spikes
    }
}

impl Default for Neuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_creation() {
        let neuron = Neuron::new();
        assert_eq!(neuron.pth, PTH);
        assert_eq!(neuron.t_ref, 4);
        assert_eq!(neuron.p, PREST);
    }

    #[test]
    fn test_neuron_initialization() {
        let mut neuron = Neuron::new();
        neuron.initial();
        assert_eq!(neuron.t_rest, -1);
        assert_eq!(neuron.p, PREST);
    }

    #[test]
    fn test_spike_detection() {
        let mut neuron = Neuron::new();
        neuron.p = neuron.pth + 1.0;
        assert_eq!(neuron.check(), 1);
        assert_eq!(neuron.p, PREF);
    }
}