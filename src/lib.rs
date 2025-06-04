//! Spiking neural network: LIF neurons, receptive-field rate coding, STDP.

pub mod encode;
pub mod layer;
pub mod neuron;
pub mod params;
pub mod rng;
pub mod stdp;
pub mod weights;

pub use encode::SpikeTrain;
pub use layer::{Layer, Observer, Outcome, Step};
pub use neuron::Neuron;
pub use params::Params;
pub use rng::Rng;
