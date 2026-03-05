//! Writes MNIST as N-MNIST style recordings (simulated saccades, `.bin` files),
//! to try the event pipeline without the real recordings.
//! usage: cargo run --release -p snn-core --example export_nmnist -- <mnist dir> <out dir> [train] [test]

use snn_core::events::Saccades;
use snn_core::{dataset, nmnist};
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let (mnist, out) = (
        args.next().expect("mnist dir"),
        args.next().expect("out dir"),
    );
    let mut count = |default: usize| args.next().map_or(default, |s| s.parse().expect("count"));
    let (train, test) = (count(60_000), count(10_000));
    let saccades = Saccades::default();
    for (set, is_train, n) in [("Train", true, train), ("Test", false, test)] {
        let mut data = dataset::mnist(Path::new(&mnist), is_train).expect("mnist");
        data.truncate(n);
        nmnist::export(&data, &saccades, Path::new(&out), set).expect("export");
        println!("{set}: {} recordings", data.len());
    }
}
