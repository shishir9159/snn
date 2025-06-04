//! Spiking neural network: LIF neurons, receptive-field rate coding, STDP.

pub mod encode;
pub mod neuron;
pub mod params;
pub mod rng;

pub use encode::SpikeTrain;
pub use neuron::Neuron;
pub use params::Params;
pub use rng::Rng;
