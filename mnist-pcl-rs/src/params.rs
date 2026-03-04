//! Predictive Coding Light constants, from N'dri, Barbier, Teulière and
//! Triesch (CVPR workshops 2023), Tables 1 and 2, with the layout scaled from a
//! 66×66 retina to MNIST's 34×34 event frame. Rates marked "tuned" differ.

#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// simple-cell receptive field side, pixels
    pub rf: usize,
    /// step between neighbouring simple-cell locations, pixels
    pub stride: usize,
    /// simple-cell features per location, weights shared across locations
    pub features: usize,
    /// complex-cell receptive field side, in simple-cell locations
    pub complex_rf: usize,
    pub complex_features: usize,
    /// lateral inhibition reaches this many locations each way
    pub lateral_range: usize,

    /// membrane time constants, ms
    pub tau_m: f32,
    pub complex_tau_m: f32,
    pub v_thr: f32,
    pub complex_thr: f32,
    pub v_reset: f32,
    pub v_min: f32,
    /// relative refractory trace: amplitude and time constant
    pub eta_rp: f32,
    pub tau_rp: f32,
    /// fixed inhibition among cells at one location
    pub w_static: f32,
    pub complex_w_static: f32,

    /// excitatory STDP of simple cells (exponential windows)
    pub ltp: f32,
    pub ltd: f32,
    pub tau_ltp: f32,
    pub tau_ltd: f32,
    /// inhibitory STDP, same windows
    pub i_ltp: f32,
    pub i_ltd: f32,
    /// excitatory STDP of complex cells (rectangular windows, ms)
    pub c_ltp: f32,
    pub c_ltd: f32,
    pub c_win: f32,
    /// L2 norm each kind of weight vector is held at
    pub norm_exc: f32,
    pub norm_lateral: f32,
    pub norm_top_down: f32,
    pub norm_complex: f32,

    pub readout_rate: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            rf: 10,
            stride: 6,
            features: 32,
            complex_rf: 2,
            complex_features: 16,
            lateral_range: 1,
            tau_m: 20.0,
            complex_tau_m: 200.0,
            // tuned: MNIST saccades send fewer events than the paper's camera
            v_thr: 15.0,
            complex_thr: 3.0,
            v_reset: -20.0,
            v_min: -80.0,
            eta_rp: 1.0,
            tau_rp: 30.0,
            w_static: 20.0,
            complex_w_static: 15.0,
            // tuned: the paper's 77e-6 / 21e-6 suit hours of camera input
            ltp: 0.002,
            ltd: 0.0006,
            tau_ltp: 14.0,
            tau_ltd: 7.0,
            i_ltp: 0.77,
            i_ltd: 0.21,
            c_ltp: 0.02,
            c_ltd: 0.02,
            c_win: 20.0,
            norm_exc: 4.0,
            norm_lateral: 40.0,
            norm_top_down: 200.0,
            norm_complex: 10.0,
            readout_rate: 0.02,
        }
    }
}

snn_core::tunable!(Params {
    "rf" => rf,
    "stride" => stride,
    "features" => features,
    "complex_rf" => complex_rf,
    "complex_features" => complex_features,
    "lateral_range" => lateral_range,
    "tau_m" => tau_m,
    "complex_tau_m" => complex_tau_m,
    "v_thr" => v_thr,
    "complex_thr" => complex_thr,
    "v_reset" => v_reset,
    "v_min" => v_min,
    "eta_rp" => eta_rp,
    "tau_rp" => tau_rp,
    "w_static" => w_static,
    "complex_w_static" => complex_w_static,
    "ltp" => ltp,
    "ltd" => ltd,
    "tau_ltp" => tau_ltp,
    "tau_ltd" => tau_ltd,
    "i_ltp" => i_ltp,
    "i_ltd" => i_ltd,
    "c_ltp" => c_ltp,
    "c_ltd" => c_ltd,
    "c_win" => c_win,
    "norm_exc" => norm_exc,
    "norm_lateral" => norm_lateral,
    "norm_top_down" => norm_top_down,
    "norm_complex" => norm_complex,
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
        p.set("v_thr", "12.5").unwrap();
        p.set("features", "8").unwrap();
        assert_eq!((p.v_thr, p.features), (12.5, 8));
        assert!(p.set("features", "many").is_err());
    }
}
