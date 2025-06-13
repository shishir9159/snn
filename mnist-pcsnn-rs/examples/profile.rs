//! Per-phase timing of PC-SNN on MNIST.
//! usage: cargo run --release -p mnist-pcsnn-rs --example pcsnn-profile -- <mnist dir> [images]

use mnist_pcsnn::{Net, Params};
use snn_core::{dataset, Rng};
use std::path::Path;
use std::time::{Duration, Instant};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("mnist dir");
    let n: usize = args
        .next()
        .map_or(10_000, |s| s.parse().expect("image count"));
    let mut data = dataset::mnist(Path::new(&dir), true).expect("mnist");
    data.truncate(n);

    let mut rng = Rng::new(1);
    let mut net = Net::new(784, 10, Params::default(), &mut rng);
    let mut t = [Duration::ZERO; 4];
    for i in 0..data.len() {
        let t0 = Instant::now();
        net.encode(data.image(i));
        net.drop_hidden(Some(&mut rng));
        let t1 = Instant::now();
        net.forward();
        let t2 = Instant::now();
        net.infer(data.label(i));
        let t3 = Instant::now();
        net.learn();
        net.revive(&mut rng);
        let t4 = Instant::now();
        t[0] += t1 - t0;
        t[1] += t2 - t1;
        t[2] += t3 - t2;
        t[3] += t4 - t3;
    }
    let names = ["encode", "forward", "inference", "learning"];
    for (name, d) in names.iter().zip(t) {
        let us = d.as_secs_f64() * 1e6 / data.len() as f64;
        println!("{name:12} {us:8.1} us/img");
    }
}
