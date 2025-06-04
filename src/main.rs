use snn::{encode, Neuron, Params, Rng, SpikeTrain};

fn main() {
    let p = Params::default();

    // a vertical bar through a blank 28x28 image
    let mut img = [0u8; 28 * 28];
    for y in 4..24 {
        for x in 13..16 {
            img[y * 28 + x] = 255;
        }
    }

    let mut pot = Vec::new();
    encode::receptive_field(&img, 28, 28, &mut pot);
    let mut train = SpikeTrain::new();
    train.encode(&pot, p.time_steps, &p.encode, &mut Rng::new(0));
    println!(
        "input spikes: {}, peak simultaneous: {}",
        train.total(),
        train.max_simultaneous()
    );

    // one neuron with uniform weights on every input
    let mut n = Neuron::new(&p.lif);
    let threshold = train.max_simultaneous() as f32 / p.threshold_div;
    let mut fired = 0;
    for t in 1..=p.time_steps {
        let input = 0.2 * train.active(t).len() as f32;
        n.integrate(t as i32, input, &p.lif);
        fired += n.fire(t as i32, threshold, &p.lif) as u32;
    }
    println!("threshold {threshold:.1}, output spikes: {fired}");
}
