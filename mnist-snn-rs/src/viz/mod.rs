//! Live view: binary frames, broadcast hub, trainer tap and the web server.

pub mod frame;
pub mod hub;
pub mod server;
pub mod tap;

pub use hub::Hub;
pub use tap::Tap;
