//! Shared building blocks for the spiking models: datasets, input coding,
//! seeded randomness, weight files and run control.

pub mod control;
pub mod dataset;
pub mod encode;
pub mod events;
pub mod rng;
pub mod run;
pub mod weights;

pub use control::{Control, Pacer};
pub use dataset::Dataset;
pub use encode::SpikeTrain;
pub use rng::Rng;
pub use run::{EpochResult, ImageInfo, Recent};
