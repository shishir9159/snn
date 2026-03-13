//! Model parameters. Defaults follow the reference Python model except where
//! tuning on MNIST showed otherwise (noted inline).

use snn_core::encode::EncodeParams;
use std::fmt;
use std::str::FromStr;

/// How the winner's weights learn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// pair STDP around every post spike
    Pair,
    /// minimal triplet STDP (Pfister & Gerstner 2006)
    Triplet,
    /// pair STDP gated by reward: neuron j answers class j mod classes, a right
    /// first spike is rewarded and a wrong one punished (Mozafari et al. 2018)
    Reward,
}

impl FromStr for Rule {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "pair" => Ok(Rule::Pair),
            "triplet" => Ok(Rule::Triplet),
            "reward" => Ok(Rule::Reward),
            _ => Err(format!("{s} is not pair, triplet or reward")),
        }
    }
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let name = match self {
            Rule::Pair => "pair",
            Rule::Triplet => "triplet",
            Rule::Reward => "reward",
        };
        f.write_str(name)
    }
}

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
    /// triplet rule: time constants of the pre trace and of the fast and slow
    /// post traces, steps
    pub tau_r: f32,
    pub tau_o: f32,
    pub tau_y: f32,
    /// triplet rule: potentiation per pre trace at a post spike, alone and
    /// times the slow post trace; depression per fast post trace at a pre spike
    pub a2_ltp: f32,
    pub a3_ltp: f32,
    pub a2_ltd: f32,
}

/// Delayed reward: eligibility traces and dopamine, in steps of 1 ms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DopamineParams {
    /// eligibility trace time constant
    pub tau_c: f32,
    /// dopamine time constant
    pub tau_d: f32,
    /// steps from an answer to its reward, plus up to `jitter` more
    pub delay: u32,
    pub jitter: u32,
    /// dopamine per unit of reward: a tag made as its reward arrives moves by
    /// `gain` times the immediate rule's step
    pub gain: f32,
    /// silent steps after each training image, as between conditioning
    /// trials; rewards still arrive in them. Without one, a reward due 1-3 s
    /// late lands on the tags of the images after it and learns nothing
    pub gap: u32,
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
    /// taken off every receptive-field weight; 0.1 makes the kernel sum to
    /// zero, so only edges drive. Filled shapes need it: on Fashion-MNIST the
    /// broadest template otherwise wins every image (12.9% test against 54.7%),
    /// while MNIST digits do better without (71.3% against 57.3%)
    pub kernel_offset: f32,
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
    /// bits per stored weight, on a grid over [w_min, w_max]; 0 keeps full precision
    pub weight_bits: u32,
    /// round learned weights up with the probability of the step they gained,
    /// instead of to the nearest level, so updates below a step still add up
    pub stochastic_rounding: bool,
    pub rule: Rule,
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
            // Pfister & Gerstner's minimal visual-cortex fit; amplitudes tuned on MNIST
            tau_r: 16.8,
            tau_o: 33.7,
            tau_y: 114.0,
            a2_ltp: 0.0,
            a3_ltp: 0.4,
            a2_ltd: 0.01,
        }
    }
}

impl Default for DopamineParams {
    fn default() -> Self {
        // Izhikevich 2007: traces of 1 s, dopamine of 0.2 s, rewards 1-3 s
        // late; trials 3 s apart so each lands before the next image. A 1-3 s
        // old tag keeps a sixth of itself on average, which the gain makes up
        Self {
            tau_c: 1000.0,
            tau_d: 200.0,
            delay: 1000,
            jitter: 2000,
            gain: 8.0,
            gap: 2800,
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
            kernel_offset: 0.0,
            silent_penalty: 0.02,
            w_norm: 8.0,
            theta_rate: 0.05,
            w_init: (0.0, 0.4),
            weight_bits: 0,
            stochastic_rounding: true,
            rule: Rule::Pair,
        }
    }
}

snn_core::tunable!(Params {
    "time_steps" => time_steps,
    "threshold_div" => threshold_div,
    "kernel_offset" => kernel_offset,
    "silent_penalty" => silent_penalty,
    "w_norm" => w_norm,
    "theta_rate" => theta_rate,
    "weight_bits" => weight_bits,
    "stochastic_rounding" => stochastic_rounding,
    "rule" => rule,
    "rest" => lif.rest,
    "floor" => lif.floor,
    "leak" => lif.leak,
    "t_ref" => lif.t_ref,
    "a_plus" => stdp.a_plus,
    "a_minus" => stdp.a_minus,
    "tau_plus" => stdp.tau_plus,
    "tau_minus" => stdp.tau_minus,
    "t_back" => stdp.t_back,
    "t_fore" => stdp.t_fore,
    "sigma" => stdp.sigma,
    "w_min" => stdp.w_min,
    "w_max" => stdp.w_max,
    "tau_r" => stdp.tau_r,
    "tau_o" => stdp.tau_o,
    "tau_y" => stdp.tau_y,
    "a2_ltp" => stdp.a2_ltp,
    "a3_ltp" => stdp.a3_ltp,
    "a2_ltd" => stdp.a2_ltd,
    "pot_min" => encode.pot_min,
    "pot_max" => encode.pot_max,
    "freq_min" => encode.freq_min,
    "freq_max" => encode.freq_max,
    "period_scale" => encode.period_scale,
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
            let value = match *key {
                "stochastic_rounding" => "false",
                "rule" => "triplet",
                _ => "1",
            };
            Params::default().set(key, value).unwrap();
        }
        let mut p = Params::default();
        p.set("leak", "0.4").unwrap();
        p.set("a_plus", "0.9").unwrap();
        p.set("freq_max", "30").unwrap();
        assert_eq!(
            (p.lif.leak, p.stdp.a_plus, p.encode.freq_max),
            (0.4, 0.9, 30.0)
        );
        p.set("weight_bits", "4").unwrap();
        assert_eq!(p.weight_bits, 4);
        p.set("rule", "reward").unwrap();
        assert_eq!(p.rule, Rule::Reward);
        assert!(p.set("rule", "hebb").is_err());
        assert_eq!(Rule::Triplet.to_string().parse(), Ok(Rule::Triplet));
        assert!(p.set("leek", "1").is_err());
        assert!(p.set("t_ref", "1.5").is_err());
    }
}
