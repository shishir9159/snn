//! SpNCN constants. Time constants, threshold and rates follow Ororbia's
//! spiking neural coding network (Neurocomputing 2023); the scales the paper
//! leaves open were tuned on MNIST.

#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// latent layer sizes, bottom up
    pub layers: Vec<usize>,
    /// integration step, ms (paper: 0.25)
    pub dt: f32,
    /// presentation time per image, ms
    pub present: f32,
    /// membrane time constant, ms
    pub tau_m: f32,
    /// input current time constant, ms
    pub tau_j: f32,
    /// spike trace time constant, ms
    pub tau_tr: f32,
    /// leak of the input current
    pub kappa: f32,
    /// leak of the membrane
    pub gamma: f32,
    /// membrane resistance; with the weight norm it sets how much error makes a unit fire
    pub resistance: f32,
    /// initial firing threshold
    pub v_thr: f32,
    /// absolute refractory period, ms
    pub t_ref: f32,
    /// ST-LRA step size
    pub alpha: f32,
    /// error-synapse rate relative to alpha
    pub beta: f32,
    /// every unit's prediction and error weights are held at this L2 norm
    pub norm: f32,
    /// input rate of a white pixel, Hz
    pub max_rate: f32,
    /// average rate homeostasis aims each latent unit at, Hz (0 disables)
    pub target_rate: f32,
    /// how fast thresholds follow it
    pub theta_rate: f32,
    /// step size of the softmax readout
    pub readout_rate: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            // paper: three layers of 3000 at dt 0.25; sized here for real time
            layers: vec![400, 100],
            dt: 1.0,
            present: 200.0,
            tau_m: 20.0,
            tau_j: 10.0,
            tau_tr: 30.0,
            kappa: 0.25,
            gamma: 1.0,
            resistance: 0.03,
            v_thr: 0.4,
            t_ref: 1.0,
            // paper: 0.055 with weights bounded at norm 20
            alpha: 0.005,
            beta: 1.0,
            norm: 5.0,
            max_rate: 63.75,
            target_rate: 1.0,
            theta_rate: 0.02,
            readout_rate: 0.02,
        }
    }
}

impl Params {
    /// Simulation steps per image.
    pub fn steps(&self) -> usize {
        (self.present / self.dt).round().max(1.0) as usize
    }

    /// Per-step decay of a spike trace.
    pub fn trace_decay(&self) -> f32 {
        1.0 - self.dt / self.tau_tr
    }
}

snn_core::tunable!(Params {
    "dt" => dt,
    "present" => present,
    "tau_m" => tau_m,
    "tau_j" => tau_j,
    "tau_tr" => tau_tr,
    "kappa" => kappa,
    "gamma" => gamma,
    "resistance" => resistance,
    "v_thr" => v_thr,
    "t_ref" => t_ref,
    "alpha" => alpha,
    "beta" => beta,
    "norm" => norm,
    "max_rate" => max_rate,
    "target_rate" => target_rate,
    "theta_rate" => theta_rate,
    "readout_rate" => readout_rate,
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
        p.set("alpha", "0.01").unwrap();
        p.set("present", "100").unwrap();
        assert_eq!((p.alpha, p.steps()), (0.01, 100));
        assert!(p.set("layers", "10").is_err(), "layers has its own flag");
    }
}
