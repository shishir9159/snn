//! Per-phase timing of SpNCN on MNIST.
//! usage: cargo run --release -p mnist-spncn-rs --example spncn-profile -- <mnist dir> [images]

use mnist_spncn::{Net, Params};
use snn_core::encode::{Coding, EncodeParams};
use snn_core::{dataset, Rng, SpikeTrain};
use std::path::Path;
use std::time::{Duration, Instant};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("mnist dir");
    let n: usize = args
        .next()
        .map_or(1000, |s| s.parse().expect("image count"));
    let mut data = dataset::mnist(Path::new(&dir), true).expect("mnist");
    data.truncate(n);

    let p = Params::default();
    let ep = EncodeParams {
        coding: Coding::Poisson,
        pot_min: 0.0,
        pot_max: 1.0,
        freq_min: 0.0,
        freq_max: p.max_rate,
        period_scale: 1000.0 / p.dt,
    };
    let mut rng = Rng::new(1);
    let mut net = Net::new(784, p.clone(), &mut rng);
    let (mut pot, mut train) = (Vec::new(), SpikeTrain::new());
    let mut t = [Duration::ZERO; 3];
    for i in 0..data.len() {
        let t0 = Instant::now();
        pot.clear();
        pot.extend(data.image(i).iter().map(|&v| v as f32 / 255.0));
        train.encode(&pot, p.steps(), &ep, &mut rng);
        let t1 = Instant::now();
        // inference on a copy, so learning below sees the same weights
        let mut frozen = net.clone();
        let t2 = Instant::now();
        frozen.present(&train, false);
        let t3 = Instant::now();
        net.present(&train, true);
        let t4 = Instant::now();
        t[0] += t1 - t0;
        t[1] += t3 - t2;
        t[2] += t4 - t3;
    }
    let names = ["encode", "present", "present + learning"];
    for (name, d) in names.iter().zip(t) {
        let us = d.as_secs_f64() * 1e6 / data.len() as f64;
        println!("{name:20} {us:9.1} us/img");
    }
}
