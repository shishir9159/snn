//! Predictive Coding Light (N'dri et al. 2023): an event-driven spiking
//! hierarchy whose plastic lateral and top-down inhibition learns to suppress
//! predictable spikes, on MNIST turned into saccade events.

pub mod net;
pub mod params;

pub use net::Net;
pub use params::Params;
