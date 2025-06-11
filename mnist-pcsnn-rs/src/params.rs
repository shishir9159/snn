//! PC-SNN constants, from Lan et al., "PC-SNN: supervised learning with local
//! Hebbian synaptic plasticity based on predictive coding in spiking neural
//! networks" (arXiv 2211.15386v1), Table II for MNIST.

#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    pub hidden: usize,
    /// latest spike time; also a silent neuron's time
    pub t_max: f32,
    /// firing threshold of the non-leaky integrate-and-fire neurons
    pub threshold: f32,
    /// how far behind the first spike wrong classes are pushed
    pub gamma: f32,
    /// slope dV/dt linking potential and spike-time errors
    pub alpha: f32,
    /// variance of the spike-time prediction errors
    pub sigma: f32,
    /// learning rates of the hidden and output weights
    pub eta_hidden: f32,
    pub eta_output: f32,
    /// inference iterations per image (the paper iterates to convergence)
    pub iterations: usize,
    /// step size of each inference iteration (unspecified in the paper)
    pub infer_rate: f32,
    /// share of hidden neurons silenced while training
    pub dropout: f32,
    /// training images between checks for hidden neurons that stopped firing
    pub revive_every: usize,
    /// share of those images a neuron must fire in to keep its weights
    pub revive_below: f32,
    /// initial weights are uniform in [0, init]
    pub init_hidden: f32,
    pub init_output: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            hidden: 200,
            t_max: 256.0,
            threshold: 100.0,
            gamma: 20.0,
            alpha: 1.0,
            sigma: 10.0,
            eta_hidden: 0.06,
            eta_output: 0.02,
            iterations: 10,
            infer_rate: 0.1,
            dropout: 0.5,
            revive_every: 2000,
            revive_below: 0.001,
            init_hidden: 5.0,
            init_output: 10.0,
        }
    }
}

impl Params {
    /// Simulation steps per image: one per time unit up to t_max.
    pub fn steps(&self) -> usize {
        self.t_max as usize
    }
}
