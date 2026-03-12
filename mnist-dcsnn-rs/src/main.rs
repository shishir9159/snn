use mnist_dcsnn::train::Trainer;
use mnist_dcsnn::Params;
use snn_core::{weights, Control, Dataset, Exits, Overrides};
use snn_live::app::{self, Args, Printer};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "  --epochs N         epochs of the decision layer, after the feature layers'
  --fields-png FILE  save the L1 features, seen through the filters, as an image";

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
            "--fields-png" => o.fields_png = Some(val.into()),
            _ => return set.flag(flag, val),
        }
        Ok(true)
    })?;
    set.apply(|k, v| o.params.set(k, v))?;
    Ok((a, o))
}

fn save_fields(o: &Opts, t: &Trainer) -> Result<(), String> {
    let Some(path) = &o.fields_png else {
        return Ok(());
    };
    let (mut all, mut f) = (Vec::new(), Vec::new());
    for m in 0..t.net.l1.output.maps {
        t.net.l1_feature(m, &mut f);
        all.extend_from_slice(&f);
    }
    let side = (f.len() as f32).sqrt() as usize;
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
    app::early_exit(a, || {
        t.exits = Some(Exits::default());
        t.evaluate(test, 0, &mut (), &ctl);
        t.exits.take().unwrap_or_default()
    });
}

fn run(a: Args, o: Opts) -> Result<(), String> {
    let (train, test) = app::load_data(&a)?;
    let classes = train.classes.max(test.classes);
    let mut trainer = Trainer::new(o.params.clone(), train.width, train.height, classes, a.seed)?;
    let epochs = a.epochs();
    let printer = Printer::new(train.len()).dense(trainer.net.dense_macs());
    let ctl = Control::new(0);
    let test_set = (!test.is_empty()).then_some(&test);
    trainer.fit(&train, test_set, epochs, &mut { printer }, &ctl);
    robustness(&a, &mut trainer, &test);
    save_fields(&o, &trainer)
}

fn main() -> ExitCode {
    let (args, opts) = match parse_args(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("{e}");
            }
            eprintln!(
                "usage: mnist-dcsnn-rs [options]\n{}\n{USAGE}\n{}",
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
        let (_, o) = parse(&["--fields-png", "f.png", "--set", "l2_maps=100"]).unwrap();
        assert_eq!(o.fields_png, Some("f.png".into()));
        assert_eq!(o.params.l2.maps, 100);
        assert!(parse(&["--set", "l1_threshold=high"])
            .unwrap_err()
            .contains("bad value"));
        assert!(parse(&["--maps", "3"]).unwrap_err().contains("unknown"));
    }
}
