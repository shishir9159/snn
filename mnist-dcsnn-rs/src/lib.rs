//! Deep convolutional spiking network (Mozafari et al. 2019): DoG filters and
//! rank-order coding, two convolutional layers that learn features by STDP
//! (Kheradpisheh et al. 2018), and a decision layer trained by
//! reward-modulated STDP.

pub mod conv;
pub mod input;
pub mod net;
pub mod params;
pub mod train;
pub mod wave;

pub use net::Net;
pub use params::{Params, Phase};
