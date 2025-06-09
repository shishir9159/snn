//! Streams made-up population activity to the dashboard's generic view, for
//! working on views without training a model.
//! usage: cargo run -p snn-live --example pop_demo -- [port]

use snn_core::{dataset, Control, ImageInfo, Pacer, Rng};
use snn_live::app::{self, Args};
use snn_live::frame::{self, Buf, Meta, Param, Pop, Role};
use snn_live::pop::End;
use snn_live::PopTap;

const T: usize = 200;
const HIDDEN: usize = 100;

fn main() {
    let port = std::env::args().nth(1).unwrap_or("7878".into());
    let a = Args::parse(["--port".into(), port], |_, _| Ok(false)).expect("port");
    let data = dataset::synthetic(1000, 1);
    let meta = Meta {
        model: "pop",
        dataset: "demo",
        width: 28,
        height: 28,
        classes: 10,
        n_train: data.len(),
        n_test: 0,
        epochs: 1,
        steps: T,
        dt_ms: 1.0,
    };
    let pops = [
        Pop::grid("input", 28, 28, Role::Input),
        Pop::new("hidden", HIDDEN, Role::Latent),
        Pop::new("label", 10, Role::Label),
    ];
    let params = [Param::num("Steps", T as f32, "ms")];
    let mut packet = app::meta_packet(&meta, "Model · demo", &params, &pops);
    let mut b = Buf::new();
    frame::series(
        &mut b,
        &["‖e‖", "drive"],
        &["V", "J", "z"],
        &["error", "spikes"],
    );
    packet.extend(b.take());

    app::serve_live(&a, packet, move |hub, ctl| {
        run(&data, PopTap::new(hub, ctl.clone()), &ctl)
    })
    .expect("live view");
}

fn run(data: &dataset::Dataset, mut tap: PopTap, ctl: &Control) {
    let mut rng = Rng::new(7);
    let mut pacer = Pacer::new();
    let field: Vec<f32> = (0..16 * 784).map(|_| rng.range(-1.0, 1.0)).collect();
    let (mut hits, mut seen) = (0, 0);
    for i in 0..data.len() {
        let info = ImageInfo {
            epoch: 0,
            index: i,
            total: data.len(),
            label: data.label(i),
            learn: true,
        };
        let (pop, unit) = tap.probe_request().unwrap_or((1, 0));
        tap.begin_image(&info, data.image(i), (pop, unit, 1.0));
        let (mut v, mut counts, mut spikes) = (0.0f32, [0u32; 10], [0u32; 3]);
        for t in 1..=T {
            let input: Vec<u16> = (0..784)
                .filter(|&p| rng.f32() < data.image(i)[p] as f32 / 255.0 * 0.06)
                .map(|p| p as u16)
                .collect();
            let hidden: Vec<u16> = (0..HIDDEN)
                .filter(|_| rng.f32() < 0.02)
                .map(|j| j as u16)
                .collect();
            let label: Vec<u16> = (0..10)
                .filter(|&c| rng.f32() < if c == info.label as usize { 0.05 } else { 0.01 })
                .map(|c| c as u16)
                .collect();
            label.iter().for_each(|&c| counts[c as usize] += 1);
            for (s, n) in spikes
                .iter_mut()
                .zip([input.len(), hidden.len(), label.len()])
            {
                *s += n as u32;
            }
            v = if v >= 1.0 { 0.0 } else { v + rng.f32() * 0.05 };
            let err = (-(t as f32) / 60.0).exp();
            tap.step(
                t,
                &[&input, &hidden, &label],
                &[err, input.len() as f32],
                &[v, v * 0.5, err],
            );
            pacer.tick(ctl);
        }
        let predicted = (0..10).max_by_key(|&c| counts[c]).map(|c| c as u8);
        let correct = predicted == Some(info.label);
        hits += correct as usize;
        seen += 1;
        let scores: Vec<f32> = counts.iter().map(|&c| c as f32).collect();
        let e = End {
            predicted,
            correct,
            running_accuracy: hits as f32 / seen as f32,
            scores: &scores,
            spikes: &spikes,
            scalars: &[rng.f32(), spikes.iter().sum::<u32>() as f32],
        };
        let units: Vec<u16> = (0..16).collect();
        let labels: Vec<Option<u8>> = (0..16).map(|k| Some(k as u8 % 10)).collect();
        let values: Vec<f32> = (0..16).map(|k| k as f32 / 8.0 - 1.0).collect();
        let fields = |b: &mut Buf| {
            let f = frame::Fields {
                title: "Fig. 02 — demo fields",
                pop: 1,
                w: 28,
                h: 28,
                channels: 1,
                units: &units,
                labels: &labels,
                values: &values,
                data: &field,
            };
            frame::fields(b, &f);
        };
        tap.end_image(&info, &e, |_| {}, fields);
        if ctl.stopped() {
            break;
        }
    }
}
