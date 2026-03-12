//! Per-phase timing of the deep convolutional network on MNIST, one image
//! count per training phase.
//! usage: cargo run --release -p mnist-dcsnn-rs --example dcsnn-profile -- <mnist dir> [images]

use mnist_dcsnn::{Net, Params, Phase};
use snn_core::{dataset, Rng};
use std::path::Path;
use std::time::{Duration, Instant};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("mnist dir");
    let n: usize = args
        .next()
        .map_or(2000, |s| s.parse().expect("image count"));
    let mut data = dataset::mnist(Path::new(&dir), true).expect("mnist");
    data.truncate(n);

    let mut rng = Rng::new(1);
    let mut net = Net::new(28, 28, 10, Params::default(), &mut rng).expect("shapes");
    let phases = [
        (Phase::Layer1, "l1"),
        (Phase::Layer2, "l2"),
        (Phase::Decision, "decision"),
    ];
    for (phase, prefix) in phases {
        let mut t = [Duration::ZERO; 3];
        // spikes of S1, L1, L1 pooled, L2, L2 pooled; synops into L1, L2, L3
        let (mut spikes, mut synops) = ([0usize; 5], [0u64; 3]);
        for i in 0..data.len() {
            let t0 = Instant::now();
            net.encode(data.image(i));
            let t1 = Instant::now();
            let answer = net.forward(phase, true);
            let t2 = Instant::now();
            net.learn(phase, data.label(i), answer);
            let t3 = Instant::now();
            t[0] += t1 - t0;
            t[1] += t2 - t1;
            t[2] += t3 - t2;
            let waves = [
                &net.input,
                &net.fire1.out,
                &net.pool1,
                &net.fire2.out,
                &net.pool2,
            ];
            for (s, w) in spikes.iter_mut().zip(waves).take(2 * net.depth + 1) {
                *s += w.spikes();
            }
            for (s, f) in synops
                .iter_mut()
                .zip([&net.fire1, &net.fire2, &net.fire3])
                .take(net.depth)
            {
                *s += f.synops;
            }
        }
        let n = data.len() as f64;
        let spikes = spikes.map(|s| (s as f64 / n).round());
        let synops = synops.map(|s| (s as f64 / n / 1e3).round());
        println!("{prefix:20} spikes {spikes:?}  synops {synops:?}K per img");
        let names = ["encode", "forward", "learning"];
        for (name, d) in names.iter().zip(t) {
            let us = d.as_secs_f64() * 1e6 / data.len() as f64;
            println!("{:20} {us:8.1} us/img", format!("{prefix} {name}"));
        }
    }
}
