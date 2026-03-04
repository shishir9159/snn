use mnist_snn::tap::Tap;
use mnist_snn::train::Trainer;
use mnist_snn::{Layer, Params};
use snn_core::weights::{self, Files};
use snn_core::{Control, Dataset, Overrides, Rng};
use snn_live::app::{self, Args, Printer};
use snn_live::frame::{Meta, Param, Pop, Role};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "  --outputs N        output neurons (default 40)
  --weights-png FILE save weight maps as an image";

#[derive(Debug, Clone)]
struct Opts {
    outputs: usize,
    files: Files,
    weights_png: Option<PathBuf>,
    params: Params,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<(Args, Opts), String> {
    let mut o = Opts {
        outputs: 40,
        files: Files::default(),
        weights_png: None,
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
        Param::text("Weights", format!("{}–{}", p.stdp.w_min, p.stdp.w_max)),
        Param::num("Learning rate", p.stdp.sigma, ""),
    ]
}

fn run(a: Args, o: Opts) -> Result<(), String> {
    let (train, test) = app::load_data(&a)?;
    let params = o.params.clone();
    let classes = train.classes.max(test.classes);
    let mut trainer = build_trainer(&a, &o, &params, &train, classes)?;
    let epochs = a.epochs();
    let printer = Printer::new(train.len());

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
        return save_outputs(&o, &trainer, train.width);
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
        if let Err(e) = save_outputs(&o, &trainer, side) {
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
