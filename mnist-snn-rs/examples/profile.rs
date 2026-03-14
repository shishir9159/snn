//! Per-phase timing of the training pipeline on MNIST.
//! usage: cargo run --release -p mnist-snn-rs --example profile -- <mnist dir> [images] [outputs]

use mnist_snn::layer::Reward;
use mnist_snn::params::Rule;
use mnist_snn::{Layer, Params};
use snn_core::{dataset, encode, Rng, SpikeTrain};
use std::path::Path;
use std::time::{Duration, Instant};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("mnist dir");
    let n: usize = args
        .next()
        .map_or(20_000, |s| s.parse().expect("image count"));
    let outputs: usize = args.next().map_or(40, |s| s.parse().expect("output count"));
    let mut data = dataset::mnist(Path::new(&dir), true).expect("mnist");
    data.truncate(n);

    let p = Params::default();
    let mut rng = Rng::new(1);
    let mut layer = Layer::new(784, outputs, &p, &mut rng);
    let delayed_p = Params {
        rule: Rule::Delayed,
        ..p.clone()
    };
    let mut delayed = layer.clone();
    let voltage_p = Params {
        rule: Rule::Voltage,
        ..p.clone()
    };
    let mut voltage = layer.clone();
    let (mut pot, mut train) = (Vec::new(), SpikeTrain::new());
    let mut t = [Duration::ZERO; 6];
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
        let r = Reward {
            label: data.label(i),
            classes: 10,
            reward: 0.5,
            punish: 0.5,
        };
        delayed.present_rewarded(&train, &delayed_p, r, &mut ());
        let t6 = Instant::now();
        voltage.present(&train, &voltage_p, true, &mut ());
        let t7 = Instant::now();
        t[0] += t1 - t0;
        t[1] += t2 - t1;
        t[2] += t4 - t3;
        t[3] += t5 - t4;
        t[4] += t6 - t5;
        t[5] += t7 - t6;
    }
    let names = [
        "receptive field",
        "encode",
        "present",
        "present + learning",
        "delayed reward",
        "voltage stdp",
    ];
    for (name, d) in names.iter().zip(t) {
        let us = d.as_secs_f64() * 1e6 / data.len() as f64;
        println!("{name:20} {us:8.2} us/img");
    }
}
