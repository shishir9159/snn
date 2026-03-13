//! Diehl & Cook style MNIST network: LIF neurons, receptive-field rate coding,
//! STDP and a winner-take-all output layer.

pub mod dopamine;
pub mod fixed;
pub mod layer;
pub mod neuron;
pub mod params;
pub mod stdp;
pub mod tap;
pub mod train;

pub use fixed::Fixed;
pub use layer::{Layer, Observer, Outcome, Step};
pub use neuron::Neuron;
pub use params::Params;
