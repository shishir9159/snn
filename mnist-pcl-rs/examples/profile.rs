//! Per-phase timing of Predictive Coding Light on MNIST.
//! usage: cargo run --release -p mnist-pcl-rs --example profile -- <mnist dir> [images]

use mnist_pcl::{Net, Params};
use snn_core::events::{Events, Saccades};
use snn_core::{dataset, Rng};
use std::path::Path;
use std::time::{Duration, Instant};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("mnist dir");
    let n: usize = args.next().map_or(500, |s| s.parse().expect("image count"));
    let mut data = dataset::mnist(Path::new(&dir), true).expect("mnist");
    data.truncate(n);

    let s = Saccades::default();
    let mut ev = Events::new();
    let mut net = Net::new(34, Params::default(), &mut Rng::new(1));
    let mut t = [Duration::ZERO; 3];
    let mut events = 0;
    for i in 0..data.len() {
        let t0 = Instant::now();
        ev.generate(data.image(i), 28, 28, &s);
        events += ev.total();
        // inference on a copy, so learning below sees the same weights
        let mut frozen = net.clone();
        let t1 = Instant::now();
        frozen.reset();
        for step in 1..=ev.steps() {
            frozen.step(step as u32, ev.at(step), false);
        }
        let t2 = Instant::now();
        net.reset();
        for step in 1..=ev.steps() {
            net.step(step as u32, ev.at(step), true);
        }
        let t3 = Instant::now();
        t[0] += t1 - t0;
        t[1] += t2 - t1;
        t[2] += t3 - t2;
    }
    println!("{:.0} events per image", events as f64 / data.len() as f64);
    let names = ["events", "network", "network + learning"];
    for (name, d) in names.iter().zip(t) {
        let us = d.as_secs_f64() * 1e6 / data.len() as f64;
        println!("{name:20} {us:9.1} us/img");
    }
}
