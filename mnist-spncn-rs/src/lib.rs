//! Spiking neural coding network (SpNCN, Ororbia 2023): LIF latent layers
//! predict the layer below, error units compare, and spike-triggered local
//! representation alignment learns online.

pub mod lif;
pub mod net;
pub mod params;
pub mod readout;

pub use net::Net;
pub use params::Params;
pub use readout::Readout;
