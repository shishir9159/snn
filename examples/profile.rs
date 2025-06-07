//! Per-phase timing of the training pipeline on MNIST.
//! usage: cargo run --release --example profile -- <mnist dir> [images]

use snn::{dataset, encode, Layer, Params, Rng, SpikeTrain};
use std::path::Path;
use std::time::{Duration, Instant};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("mnist dir");
    let n: usize = args
        .next()
        .map_or(20_000, |s| s.parse().expect("image count"));
    let mut data = dataset::mnist(Path::new(&dir), true).expect("mnist");
    data.truncate(n);

    let p = Params::default();
    let mut rng = Rng::new(1);
    let mut layer = Layer::new(784, 40, &p, &mut rng);
    let (mut pot, mut train) = (Vec::new(), SpikeTrain::new());
    let mut t = [Duration::ZERO; 4];
    for i in 0..data.len() {
        let t0 = Instant::now();
        encode::receptive_field(data.image(i), 28, 28, &mut pot);
        let t1 = Instant::now();
        train.encode(&pot, p.time_steps, &p.encode, &mut rng);
        let t2 = Instant::now();
        // inference on a copy, so learning below sees the same weights
        let mut frozen = layer.clone();
        let t3 = Instant::now();
        frozen.present(&train, &p, false, &mut ());
        let t4 = Instant::now();
        layer.present(&train, &p, true, &mut ());
        let t5 = Instant::now();
        t[0] += t1 - t0;
        t[1] += t2 - t1;
        t[2] += t4 - t3;
        t[3] += t5 - t4;
    }
    let names = ["receptive field", "encode", "present", "present + learning"];
    for (name, d) in names.iter().zip(t) {
        let us = d.as_secs_f64() * 1e6 / data.len() as f64;
        println!("{name:20} {us:8.2} us/img");
    }
}
