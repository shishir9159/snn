use mnist_spncn::tap::Tap;
use mnist_spncn::train::Trainer;
use mnist_spncn::Params;
use snn_core::{weights, Control, Dataset, Overrides};
use snn_live::app::{self, Args, Printer};
use snn_live::frame::{self, Buf, Meta, Param, Pop, Role};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "  --layers N,N       latent layer sizes, bottom up (default 400,100)
  --dt MS            integration step in ms (default 1; paper 0.25)
  --fields-png FILE  save layer-1 generative fields as an image";

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
        let bad = || format!("{flag}: bad value {val}");
        match flag {
            "--layers" => {
                let sizes: Result<Vec<usize>, _> = val.split(',').map(str::parse).collect();
                o.params.layers = sizes.map_err(|_| bad())?;
                if o.params.layers.is_empty() || o.params.layers.contains(&0) {
                    return Err(bad());
                }
            }
            "--dt" => o.params.dt = val.parse().map_err(|_| bad())?,
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
    let layers: Vec<String> = p.layers.iter().map(|n| n.to_string()).collect();
    vec![
        Param::text("Latents", layers.join(" → ")),
        Param::num("τ membrane", p.tau_m, "ms"),
        Param::num("τ current", p.tau_j, "ms"),
        Param::num("τ trace", p.tau_tr, "ms"),
        Param::text("Threshold", format!("adapts · {} Hz", p.target_rate)),
        Param::num("ST-LRA α", p.alpha, ""),
        Param::num("Weight norm", p.norm, ""),
        Param::text("Input", format!("Poisson ≤ {} Hz", p.max_rate)),
    ]
}

fn save_fields(o: &Opts, t: &Trainer, side: usize) -> Result<(), String> {
    if let Some(path) = &o.fields_png {
        let w = &t.net.w[0];
        let hi = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        weights::save_png(path, w, side, -hi, hi).map_err(|e| e.to_string())?;
        println!("fields saved to {}", path.display());
    }
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
    let printer = Printer::new(train.len()).dense(trainer.dense_macs());
    let side = train.width;

    if a.headless {
        let ctl = Control::new(0);
        let test_set = (!test.is_empty()).then_some(&test);
        trainer.fit(&train, test_set, epochs, &mut { printer }, &ctl);
        robustness(&a, &mut trainer, &test);
        return save_fields(&o, &trainer, side);
    }

    let meta = Meta {
        model: "spncn",
        dataset: &train.name,
        width: train.width,
        height: train.height,
        classes,
        n_train: train.len(),
        n_test: test.len(),
        epochs,
        steps: p.steps(),
        dt_ms: p.dt,
        dense_macs: trainer.dense_macs(),
    };
    let names: Vec<String> = (1..=p.layers.len()).map(|l| format!("L{l}")).collect();
    let mut pops = vec![Pop::grid("input", train.width, train.height, Role::Input)];
    pops.extend(
        names
            .iter()
            .zip(&p.layers)
            .map(|(n, &s)| Pop::new(n, s, Role::Latent)),
    );
    let mut packet = app::meta_packet(&meta, "Model · SpNCN", &model_params(&p), &pops);
    let (step, probe, image) = Tap::series(p.layers.len());
    let step: Vec<&str> = step.iter().map(String::as_str).collect();
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
                "usage: mnist-spncn-rs [options]\n{}\n{USAGE}\n{}",
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
        let (_, o) = parse(&["--layers", "30,10", "--dt", "2", "--fields-png", "f.png"]).unwrap();
        assert_eq!((o.params.layers, o.params.dt), (vec![30, 10], 2.0));
        assert_eq!(o.fields_png, Some("f.png".into()));
        let (_, o) = parse(&["--set", "alpha=0.01", "--dt", "2", "--set", "dt=0.5"]).unwrap();
        assert_eq!(
            (o.params.alpha, o.params.dt),
            (0.01, 0.5),
            "--set applies last"
        );
        assert!(parse(&["--layers", "30,0"]).is_err());
        assert!(parse(&["--set", "layers=3"])
            .unwrap_err()
            .contains("unknown parameter"));
    }
}
