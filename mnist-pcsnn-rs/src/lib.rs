//! PC-SNN (Lan et al. 2022): supervised learning in a time-to-first-spike
//! network by predictive coding, with only local Hebbian weight updates.

pub mod net;
pub mod params;
pub mod tap;
pub mod train;

pub use net::Net;
pub use params::Params;
