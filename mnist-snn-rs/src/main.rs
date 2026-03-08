use mnist_snn::fixed::{self, Fixed};
use mnist_snn::tap::Tap;
use mnist_snn::train::Trainer;
use mnist_snn::{Layer, Params};
use snn_core::encode::receptive_field;
use snn_core::weights::{self, Files};
use snn_core::{Control, Dataset, Overrides, Rng, SpikeTrain};
use snn_live::app::{self, Args, Printer};
use snn_live::frame::{Meta, Param, Pop, Role};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "  --outputs N        output neurons (default 40)
  --weights-png FILE save weight maps as an image
  --int-bits N       after training, also run the test set through the
                     integer-only path with N-bit weights (default: the
                     trained weight_bits, else 8)
  --vectors DIR      write integer test vectors for the first test images";

/// Test images `--vectors` writes, each with a trace of every potential.
const VECTORS: usize = 100;

#[derive(Debug, Clone)]
struct Opts {
    outputs: usize,
    files: Files,
    weights_png: Option<PathBuf>,
    int_bits: Option<u32>,
    vectors: Option<PathBuf>,
    params: Params,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<(Args, Opts), String> {
    let mut o = Opts {
        outputs: 40,
        files: Files::default(),
        weights_png: None,
        int_bits: None,
        vectors: None,
        params: Params::default(),
    };
    let mut set = Overrides::default();
    let a = Args::parse(args, |flag, val| {
        match flag {
            "--outputs" => {
                let n: usize = val
                    .parse()
                    .map_err(|_| format!("{flag}: bad value {val}"))?;
                o.outputs = n.max(1);
            }
            "--weights-png" => o.weights_png = Some(val.into()),
            "--int-bits" => {
                let bits = val
                    .parse()
                    .ok()
                    .filter(|b| (1..=fixed::MAX_BITS).contains(b));
                let bad = || format!("{flag}: {val} is not 1 to {}", fixed::MAX_BITS);
                o.int_bits = Some(bits.ok_or_else(bad)?);
            }
            "--vectors" => o.vectors = Some(val.into()),
            _ => return Ok(o.files.flag(flag, val) || set.flag(flag, val)?),
        }
        Ok(true)
    })?;
    set.apply(|k, v| o.params.set(k, v))?;
    Ok((a, o))
}

fn build_trainer(
    a: &Args,
    o: &Opts,
    params: &Params,
    train: &Dataset,
    classes: usize,
) -> Result<Trainer, String> {
    let n_in = train.width * train.height;
    let Some((rows, w)) = o.files.start(n_in)? else {
        return Ok(Trainer::new(
            params.clone(),
            n_in,
            o.outputs,
            classes,
            a.seed,
        ));
    };
    let layer = Layer::with_weights(n_in, rows, w, params);
    let mut t = Trainer::with_layer(params.clone(), layer, classes, Rng::new(a.seed));
    t.assign_labels(train);
    Ok(t)
}

fn save_outputs(o: &Opts, t: &Trainer, side: usize) -> Result<(), String> {
    let l = &t.layer;
    if let Some(path) = o.files.finish(l.n_out, l.n_in, l.weights())? {
        println!("weights saved to {}", path.display());
    }
    if let Some(path) = &o.weights_png {
        let (lo, hi) = (t.params.stdp.w_min, t.params.stdp.w_max);
        weights::save_png(path, l.weights(), side, lo, hi).map_err(|e| e.to_string())?;
        println!("weight maps saved to {}", path.display());
    }
    Ok(())
}

/// Runs the test set through the float layer and the integer path, reports how
/// they compare and writes test vectors if asked.
fn integer(o: &Opts, t: &mut Trainer, test: &Dataset) -> Result<(), String> {
    if o.int_bits.is_none() && o.vectors.is_none() {
        return Ok(());
    }
    if test.is_empty() {
        return Err("the integer path runs on the test set, which is empty".into());
    }
    let p = &t.params;
    let bits = o
        .int_bits
        .unwrap_or(if p.weight_bits > 0 { p.weight_bits } else { 8 })
        .min(fixed::MAX_BITS);
    let mut f = Fixed::new(&t.layer, p, bits);
    let io = |dir: &PathBuf, e: std::io::Error| format!("{}: {e}", dir.display());
    if let Some(dir) = &o.vectors {
        f.write_config(dir, p.time_steps).map_err(|e| io(dir, e))?;
    }
    let (mut pot, mut train, mut rng) = (Vec::new(), SpikeTrain::new(), Rng::new(0));
    let (mut hits, mut float_hits, mut same, mut peak) = (0, 0, 0, 0);
    for i in 0..test.len() {
        receptive_field(test.image(i), test.width, test.height, &mut pot);
        train.encode(&pot, p.time_steps, &p.encode, &mut rng);
        let float = t.layer.present(&train, p, false, &mut ()).winner;
        let run = match &o.vectors {
            Some(dir) if i < VECTORS => f
                .write_image(dir, i, test.label(i), &train)
                .map_err(|e| io(dir, e))?,
            _ => f.present(&train, |_, _, _| {}),
        };
        let label = Some(test.label(i));
        hits += (run.winner.and_then(|w| t.labels.of(w)) == label) as usize;
        float_hits += (float.and_then(|w| t.labels.of(w)) == label) as usize;
        same += (run.winner == float) as usize;
        peak = peak.max(run.peak);
    }
    let pct = |n: usize| 100.0 * n as f32 / test.len() as f32;
    println!(
        "integer path, {bits}-bit weights: test {:.1}% (float {:.1}%), same winner on {:.1}%, potentials need {} bits",
        pct(hits),
        pct(float_hits),
        pct(same),
        Fixed::register_bits(peak)
    );
    if let Some(dir) = &o.vectors {
        let n = test.len().min(VECTORS);
        println!("test vectors for {n} images written to {}", dir.display());
    }
    Ok(())
}

fn weight_range(p: &Params) -> String {
    let range = format!("{}–{}", p.stdp.w_min, p.stdp.w_max);
    match p.weight_bits {
        0 => range,
        bits => format!("{range} · {bits} bit"),
    }
}

/// Rows of the dashboard's model panel.
fn model_params(p: &Params) -> Vec<Param<'static>> {
    let div = if p.threshold_div == 1.0 {
        String::new()
    } else {
        format!("÷{} ", p.threshold_div)
    };
    vec![
        Param::text("Threshold", div + "adaptive"),
        Param::num("Leak", p.lif.leak, "/ms"),
        Param::num("Refractory", p.lif.t_ref as f32, "ms"),
        Param::num("Floor", p.lif.floor, ""),
        Param::text("Weights", weight_range(p)),
        Param::num("Learning rate", p.stdp.sigma, ""),
        Param::text("Rule", p.rule.to_string()),
    ]
}

fn run(a: Args, o: Opts) -> Result<(), String> {
    let (train, test) = app::load_data(&a)?;
    let params = o.params.clone();
    let classes = train.classes.max(test.classes);
    let mut trainer = build_trainer(&a, &o, &params, &train, classes)?;
    if params.weight_bits > 0 {
        let n = (trainer.layer.n_in * trainer.layer.n_out) as u64;
        let bytes = n * params.weight_bits.min(16) as u64 / 8;
        println!(
            "weights: {} bits each, {bytes} bytes instead of {}",
            params.weight_bits,
            n * 4
        );
    }
    let epochs = a.epochs();
    let printer = Printer::new(train.len()).dense(trainer.layer.dense_macs());

    if a.headless {
        let ctl = Control::new(0);
        let test_set = (!test.is_empty()).then_some(&test);
        if epochs == 0 {
            if let Some(t) = test_set {
                let pass = trainer.evaluate(t, 0, &mut (), &ctl);
                println!("test {:.1}%", pass.accuracy() * 100.0);
            }
        }
        trainer.fit(&train, test_set, epochs, &mut { printer }, &ctl);
        save_outputs(&o, &trainer, train.width)?;
        return integer(&o, &mut trainer, &test);
    }

    let meta = Meta {
        model: "wta",
        dataset: &train.name,
        width: train.width,
        height: train.height,
        classes,
        n_train: train.len(),
        n_test: test.len(),
        epochs,
        steps: params.time_steps,
        dt_ms: 1.0,
        dense_macs: trainer.layer.dense_macs(),
    };
    let pops = [
        Pop::grid("input", train.width, train.height, Role::Input),
        Pop::new("output", trainer.layer.n_out, Role::Output),
    ];
    let packet = app::meta_packet(&meta, "Neuron model · LIF", &model_params(&params), &pops);
    let side = train.width;
    app::serve_live(&a, packet, move |hub, ctl| {
        let tap = Tap::new(hub, params.stdp.w_min, params.stdp.w_max);
        let test_set = (!test.is_empty()).then_some(&test);
        trainer.fit(&train, test_set, epochs, &mut (tap, printer), &ctl);
        let done = save_outputs(&o, &trainer, side).and_then(|_| integer(&o, &mut trainer, &test));
        if let Err(e) = done {
            eprintln!("error: {e}");
        }
    })
}

fn main() -> ExitCode {
    let (args, opts) = match parse_args(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("{e}");
            }
            eprintln!(
                "usage: mnist-snn-rs [options]\n{}\n{USAGE}\n{}\n{}",
                app::USAGE,
                Files::USAGE,
                Overrides::usage(Params::KEYS)
            );
            return ExitCode::from(2);
        }
    };
    match run(args, opts) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<(Args, Opts), String> {
        parse_args(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn model_flags() {
        let (a, o) = parse(&[
            "--outputs",
            "12",
            "--load",
            "a.txt",
            "--save",
            "b.txt",
            "--weights-png",
            "c.png",
            "--train",
            "5",
        ])
        .unwrap();
        assert_eq!((a.train, o.outputs), (Some(5), 12));
        assert_eq!(o.files.load, Some("a.txt".into()));
        assert_eq!(o.files.save, Some("b.txt".into()));
        assert_eq!(o.weights_png, Some("c.png".into()));
        assert!(parse(&["--outputs", "many"])
            .unwrap_err()
            .contains("bad value"));
        assert!(parse(&["--bogus", "1"]).unwrap_err().contains("unknown"));
        assert_eq!(parse(&["--outputs", "0"]).unwrap().1.outputs, 1);
    }

    #[test]
    fn set_overrides_constants_whatever_the_order() {
        let (_, o) = parse(&["--set", "leak=0.4", "--outputs", "8", "--set", "w_max=3"]).unwrap();
        assert_eq!(
            (o.params.lif.leak, o.params.stdp.w_max, o.outputs),
            (0.4, 3.0, 8)
        );
        let (_, o) = parse(&["--set", "leak=0.4", "--set", "leak=0.5"]).unwrap();
        assert_eq!(o.params.lif.leak, 0.5, "the last one wins");
        assert!(parse(&["--set", "leek=1"])
            .unwrap_err()
            .contains("unknown parameter leek"));
        assert!(parse(&["--set", "leak=fast"])
            .unwrap_err()
            .contains("bad value"));
        assert!(parse(&["--set", "leak"])
            .unwrap_err()
            .contains("name=value"));
    }

    #[test]
    fn weight_bits_can_be_set_and_are_shown() {
        let (_, o) = parse(&[
            "--set",
            "weight_bits=4",
            "--set",
            "stochastic_rounding=false",
        ])
        .unwrap();
        assert_eq!(
            (o.params.weight_bits, o.params.stochastic_rounding),
            (4, false)
        );
        assert_eq!(weight_range(&o.params), "0–2 · 4 bit");
        assert_eq!(weight_range(&Params::default()), "0–2");
        assert!(parse(&["--set", "weight_bits=-1"]).is_err());
    }

    #[test]
    fn overridden_constants_change_the_run() {
        let train = |extra: &[&str]| {
            let mut args = vec![
                "--headless",
                "--train",
                "300",
                "--test",
                "60",
                "--outputs",
                "10",
            ];
            args.extend(extra);
            let (a, o) = parse(&args).unwrap();
            let (train, test) = app::load_data(&a).unwrap();
            let mut t = build_trainer(&a, &o, &o.params, &train, 10).unwrap();
            t.fit(&train, Some(&test), 1, &mut (), &Control::new(0))
                .remove(0)
                .train_accuracy
        };
        assert_eq!(train(&[]), train(&[]), "same constants, same run");
        assert_ne!(train(&[]), train(&["--set", "sigma=0.02"]));
    }

    #[test]
    fn integer_flags_and_report() {
        let (_, o) = parse(&["--int-bits", "6", "--vectors", "v"]).unwrap();
        assert_eq!((o.int_bits, o.vectors), (Some(6), Some("v".into())));
        for bad in ["0", "13", "x"] {
            assert!(parse(&["--int-bits", bad]).unwrap_err().contains("1 to 12"));
        }
        let dir = std::env::temp_dir().join(format!("snn-main-vectors-{}", std::process::id()));
        let d = dir.to_str().unwrap();
        let args = [
            "--headless",
            "--train",
            "60",
            "--test",
            "3",
            "--outputs",
            "5",
        ];
        let (a, o) = parse(&[&args[..], &["--vectors", d]].concat()).unwrap();
        let ran = run(a, o);
        let files = std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0);
        std::fs::remove_dir_all(&dir).ok();
        ran.unwrap();
        assert_eq!(
            files,
            3 + 3 * 3,
            "config, weights, gains and three files per image"
        );
    }

    #[test]
    fn saved_weights_load_back_and_must_fit_the_images() {
        let path = std::env::temp_dir().join(format!("snn-main-{}.txt", std::process::id()));
        let file = path.to_str().unwrap();
        let (a, o) = parse(&["--headless", "--train", "80", "--test", "20"]).unwrap();
        run(a, o).unwrap();
        let (a, o) = parse(&[
            "--headless",
            "--train",
            "80",
            "--outputs",
            "6",
            "--save",
            file,
        ])
        .unwrap();
        run(a, o).unwrap();
        let (rows, cols, _) = weights::load(&path).unwrap();
        assert_eq!((rows, cols), (6, 784));

        let evaluate = [
            "--headless",
            "--epochs",
            "0",
            "--train",
            "80",
            "--test",
            "20",
            "--load",
            file,
        ];
        let (a, o) = parse(&evaluate).unwrap();
        run(a, o).unwrap();

        weights::save(&path, 2, 10, &[0.5; 20]).unwrap();
        let (a, o) = parse(&evaluate).unwrap();
        let err = run(a, o).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(err.contains("expected 784"), "{err}");
    }
}
