use mnist_dcsnn::tap::{self, Tap};
use mnist_dcsnn::train::Trainer;
use mnist_dcsnn::Params;
use snn_core::{weights, Control, Dataset, Exits, Overrides};
use snn_live::app::{self, Args, Printer};
use snn_live::frame::{self, Buf, Meta, Param, Pop, Role};
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

/// Rows of the dashboard's model panel.
fn model_params(t: &Trainer, epochs: usize) -> Vec<Param<'static>> {
    let (p, n) = (&t.net.p, &t.net);
    let conv = |l: &mnist_dcsnn::params::Layer| {
        format!("{} × {k}×{k} · θ {}", l.maps, l.threshold, k = l.kernel)
    };
    vec![
        Param::text("Coding", format!("rank order · {} steps", p.steps)),
        Param::text("S1", "DoG 3 · 7 · 13 ON/OFF".into()),
        Param::text("L1", conv(&p.l1)),
        Param::text("L2", conv(&p.l2)),
        Param::text(
            "L3",
            format!(
                "{} × {k}×{k} · {} / class",
                n.l3.output.maps,
                p.per_class,
                k = p.l3_kernel
            ),
        ),
        Param::text("Pooling", format!("{} · {}", p.l1.pool, p.l2.pool)),
        Param::text(
            "Winners",
            format!(
                "{} r{} · {} r{}",
                p.l1.winners, p.l1.radius, p.l2.winners, p.l2.radius
            ),
        ),
        Param::text(
            "STDP",
            format!("{} / {} ×2 per {}", p.ltp, p.ltd, p.rate_every),
        ),
        Param::text(
            "R-STDP",
            format!(
                "{} / {} · {} / {}",
                p.reward_ltp, p.reward_ltd, p.punish_ltp, p.punish_ltd
            ),
        ),
        Param::text(
            "Epochs",
            format!("{} + {} + {epochs}", p.l1.epochs, p.l2.epochs),
        ),
    ]
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

    if a.headless {
        let ctl = Control::new(0);
        let test_set = (!test.is_empty()).then_some(&test);
        trainer.fit(&train, test_set, epochs, &mut { printer }, &ctl);
        robustness(&a, &mut trainer, &test);
        return save_fields(&o, &trainer);
    }

    let net = &trainer.net;
    let meta = Meta {
        model: "dcsnn",
        dataset: &train.name,
        width: train.width,
        height: train.height,
        classes,
        n_train: train.len(),
        n_test: test.len(),
        epochs: trainer.epochs(epochs),
        steps: net.p.steps,
        dt_ms: 1.0,
        dense_macs: net.dense_macs(),
    };
    let pop = |name, s: mnist_dcsnn::wave::Shape, role| Pop::grid(name, s.w * s.maps, s.h, role);
    let pops = [
        pop("S1", net.input.shape, Role::Input),
        pop("L1", net.l1.output, Role::Latent),
        pop("L2", net.l2.output, Role::Latent),
        pop("L3", net.l3.output, Role::Output),
    ];
    let mut packet = app::meta_packet(
        &meta,
        "Model · DCSNN",
        &model_params(&trainer, epochs),
        &pops,
    );
    let (step, probe, image) = Tap::series();
    let mut b = Buf::new();
    frame::series(&mut b, &step, &probe, &image);
    tap::arch(&mut b, &trainer.net);
    packet.extend(b.take());

    let args = a.clone();
    app::serve_live(&a, packet, move |hub, ctl| {
        let tap = Tap::new(hub, ctl.clone());
        let test_set = (!test.is_empty()).then_some(&test);
        trainer.fit(&train, test_set, epochs, &mut (tap, printer), &ctl);
        robustness(&args, &mut trainer, &test);
        if let Err(e) = save_fields(&o, &trainer) {
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
