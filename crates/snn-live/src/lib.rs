//! Live view shared by every model: binary records, a broadcast hub, the
//! web server, stream helpers for trainer taps and the binary harness.

pub mod app;
pub mod frame;
pub mod hub;
pub mod pop;
pub mod server;
pub mod stream;

pub use hub::Hub;
pub use pop::PopTap;
pub use stream::Stream;
