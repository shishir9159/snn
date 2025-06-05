//! Model parameters. Defaults follow the reference Python model except where
//! tuning on MNIST showed otherwise (noted inline).

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LifParams {
    pub rest: f32,
    /// lowest reachable potential; inhibited neurons are pushed here
    pub floor: f32,
    /// subtracted every step while above rest
    pub leak: f32,
    /// steps a neuron stays silent after a spike
    pub t_ref: u32,
}

/// How pixel potentials become input spike trains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coding {
    /// evenly spaced spikes, rate set by the potential
    Regular,
    /// Bernoulli spikes with the same mean rate
    Poisson,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncodeParams {
    pub coding: Coding,
    /// potential range mapped onto the frequency range (clamped)
    pub pot_min: f32,
    pub pot_max: f32,
    pub freq_min: f32,
    pub freq_max: f32,
    /// spike period is ceil(period_scale / freq) steps
    pub period_scale: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StdpParams {
    /// depression amplitude when the pre spike follows the post spike
    pub a_plus: f32,
    /// potentiation amplitude when the pre spike leads
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    /// window around a post spike, in steps
    pub t_back: u32,
    pub t_fore: u32,
    /// learning rate
    pub sigma: f32,
    pub w_min: f32,
    pub w_max: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// steps an image is presented for
    pub time_steps: usize,
    pub lif: LifParams,
    pub encode: EncodeParams,
    pub stdp: StdpParams,
    /// per-image threshold = peak simultaneous input spikes / threshold_div
    pub threshold_div: f32,
    /// depression of the winner's synapses from inputs that never spiked
    pub silent_penalty: f32,
    /// L2 norm every weight row is rescaled to after learning, so a broad template
    /// can't outbid a specific one (0 disables)
    pub w_norm: f32,
    /// homeostasis rate: thresholds rise for neurons winning more than their
    /// share of images and fall for the rest
    pub theta_rate: f32,
    /// initial weights are uniform in this range
    pub w_init: (f32, f32),
}

impl Default for LifParams {
    fn default() -> Self {
        Self {
            rest: 0.0,
            floor: -500.0,
            leak: 0.15,
            t_ref: 30,
        }
    }
}

impl Default for EncodeParams {
    fn default() -> Self {
        Self {
            coding: Coding::Regular,
            pot_min: -1.069,
            pot_max: 2.781,
            freq_min: 1.0,
            freq_max: 20.0,
            period_scale: 600.0,
        }
    }
}

impl Default for StdpParams {
    fn default() -> Self {
        Self {
            a_plus: 0.8,
            a_minus: 0.3,
            tau_plus: 8.0,
            tau_minus: 5.0,
            t_back: 20,
            t_fore: 20,
            sigma: 0.1,
            // reference: -1.2; negative weights made most images drive no neuron
            w_min: 0.0,
            w_max: 2.0,
        }
    }
}

impl Default for Params {
    fn default() -> Self {
        Self {
            time_steps: 200,
            lif: LifParams::default(),
            encode: EncodeParams::default(),
            stdp: StdpParams::default(),
            // reference: 3, 0.06; no normalisation
            threshold_div: 1.0,
            silent_penalty: 0.02,
            w_norm: 8.0,
            theta_rate: 0.05,
            w_init: (0.0, 0.4),
        }
    }
}
