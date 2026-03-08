use mnist_pcsnn::tap::Tap;
use mnist_pcsnn::train::Trainer;
use mnist_pcsnn::Params;
use snn_core::{weights, Control, Dataset, Overrides};
use snn_live::app::{self, Args, Printer};
use snn_live::frame::{self, Buf, Meta, Param, Pop, Role};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "  --hidden N         hidden neurons (default 200)
  --fields-png FILE  save hidden receptive fields as an image";

#[derive(Debug)]
struct Opts {
    params: Params,
    fields_png: Option<PathBuf>,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<(Args, Opts), String> {
    let mut o = Opts {
        params: Params::default(),
        fields_png: None,
    };
    let mut set = Overrides::default();
    let a = Args::parse(args, |flag, val| {
        match flag {
            "--hidden" => {
                let n: usize = val
                    .parse()
                    .map_err(|_| format!("{flag}: bad value {val}"))?;
                o.params.hidden = n.max(1);
            }
            "--fields-png" => o.fields_png = Some(val.into()),
            _ => return set.flag(flag, val),
        }
        Ok(true)
    })?;
    set.apply(|k, v| o.params.set(k, v))?;
    Ok((a, o))
}

/// Rows of the dashboard's model panel.
fn model_params(p: &Params) -> Vec<Param<'static>> {
    vec![
        Param::text("Coding", "first spike".into()),
        Param::num("Hidden", p.hidden as f32, ""),
        Param::num("Threshold", p.threshold, ""),
        Param::num("Window", p.t_max, "steps"),
        Param::num("γ margin", p.gamma, "steps"),
        Param::num("Σ", p.sigma, ""),
        Param::text("Inference", format!("{} × {}", p.iterations, p.infer_rate)),
        Param::text("η", format!("{} / {}", p.eta_hidden, p.eta_output)),
        Param::num("Dropout", p.dropout, ""),
    ]
}

fn save_fields(o: &Opts, t: &Trainer, side: usize) -> Result<(), String> {
    let Some(path) = &o.fields_png else {
        return Ok(());
    };
    let (mut all, mut f) = (Vec::new(), Vec::new());
    for j in 0..t.net.p.hidden {
        t.net.field(j, &mut f);
        all.extend_from_slice(&f);
    }
    let hi = all.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    weights::save_png(path, &all, side, -hi, hi).map_err(|e| e.to_string())?;
    println!("fields saved to {}", path.display());
    Ok(())
}

fn robustness(a: &Args, t: &mut Trainer, test: &Dataset) {
    let ctl = Control::new(0);
    app::robustness(a, |d| {
        t.perturb = d;
        Some(t.evaluate(test, 0, &mut (), &ctl).accuracy())
    });
    t.perturb = None;
}

fn run(a: Args, o: Opts) -> Result<(), String> {
    let (train, test) = app::load_data(&a)?;
    let classes = train.classes.max(test.classes);
    let p = o.params.clone();
    let mut trainer = Trainer::new(p.clone(), train.width * train.height, classes, a.seed);
    let epochs = a.epochs();
    let printer = Printer::new(train.len()).dense(trainer.net.dense_macs());
    let side = train.width;

    if a.headless {
        let ctl = Control::new(0);
        let test_set = (!test.is_empty()).then_some(&test);
        trainer.fit(&train, test_set, epochs, &mut { printer }, &ctl);
        robustness(&a, &mut trainer, &test);
        return save_fields(&o, &trainer, side);
    }

    let meta = Meta {
        model: "pcsnn",
        dataset: &train.name,
        width: train.width,
        height: train.height,
        classes,
        n_train: train.len(),
        n_test: test.len(),
        epochs,
        steps: p.steps(),
        dt_ms: 1.0,
        dense_macs: trainer.net.dense_macs(),
    };
    let pops = [
        Pop::grid("input", train.width, train.height, Role::Input),
        Pop::new("hidden", p.hidden, Role::Latent),
        Pop::new("output", classes, Role::Output),
    ];
    let mut packet = app::meta_packet(&meta, "Model · PC-SNN", &model_params(&p), &pops);
    let (step, probe, image) = Tap::series();
    let mut b = Buf::new();
    frame::series(&mut b, &step, &probe, &image);
    packet.extend(b.take());

    let (w, h) = (train.width, train.height);
    let args = a.clone();
    app::serve_live(&a, packet, move |hub, ctl| {
        let tap = Tap::new(hub, ctl.clone(), w, h);
        let test_set = (!test.is_empty()).then_some(&test);
        trainer.fit(&train, test_set, epochs, &mut (tap, printer), &ctl);
        robustness(&args, &mut trainer, &test);
        if let Err(e) = save_fields(&o, &trainer, side) {
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
                "usage: mnist-pcsnn-rs [options]\n{}\n{USAGE}\n{}",
                app::USAGE,
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
    fn model_flags_and_set() {
        let (_, o) = parse(&["--hidden", "64", "--fields-png", "f.png"]).unwrap();
        assert_eq!(o.params.hidden, 64);
        assert_eq!(o.fields_png, Some("f.png".into()));
        let (_, o) = parse(&[
            "--set",
            "dropout=0.3",
            "--hidden",
            "10",
            "--set",
            "hidden=20",
        ])
        .unwrap();
        assert_eq!((o.params.dropout, o.params.hidden), (0.3, 20));
        assert!(parse(&["--hidden", "x"]).is_err());
        assert!(parse(&["--set", "dropout=lots"])
            .unwrap_err()
            .contains("bad value"));
    }
}
