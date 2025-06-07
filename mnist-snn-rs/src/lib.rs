//! Diehl & Cook style MNIST network: LIF neurons, receptive-field rate coding,
//! STDP and a winner-take-all output layer.

pub mod layer;
pub mod neuron;
pub mod params;
pub mod stdp;
pub mod train;
pub mod viz;

pub use layer::{Layer, Observer, Outcome, Step};
pub use neuron::Neuron;
pub use params::Params;
