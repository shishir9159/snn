//! Live view shared by every model: binary records, a broadcast hub, the web
//! server and a stream helper for trainer taps.

pub mod frame;
pub mod hub;
pub mod server;
pub mod stream;

pub use hub::Hub;
pub use stream::Stream;
