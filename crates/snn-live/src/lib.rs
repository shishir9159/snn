//! Live view shared by every model: binary records, a broadcast hub and the
//! web server.

pub mod frame;
pub mod hub;
pub mod server;

pub use hub::Hub;
