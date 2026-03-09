use mnist_pcl::input::Input;
use mnist_pcl::tap::Tap;
use mnist_pcl::train::Trainer;
use mnist_pcl::Params;
use snn_core::events::Saccades;
use snn_core::nmnist::Recordings;
use snn_core::perturb::Kind;
use snn_core::{Control, Exits, Overrides};
use snn_live::app::{self, Args, Printer};
use snn_live::frame::{self, Buf, Meta, Param, Pop, Role};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "  --nmnist DIR       N-MNIST recordings (DIR/Train, DIR/Test) instead of
                     saccades over MNIST or PNGs; --train and --test limit them
  --no-inhibition    switch plastic lateral and top-down inhibition off";

#[derive(Debug)]
struct Opts {
    params: Params,
    nmnist: Option<PathBuf>,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<(Args, Opts), String> {
    let mut o = Opts {
        params: Params::default(),
        nmnist: None,
    };
    let mut args: Vec<String> = args.into_iter().collect();
    // the one flag without a value
    if let Some(i) = args.iter().position(|a| a == "--no-inhibition") {
        args.remove(i);
        o.params.norm_lateral = 1e-3;
        o.params.norm_top_down = 1e-3;
    }
    let mut set = Overrides::default();
    let a = Args::parse(args, |flag, val| match flag {
        "--nmnist" => {
            o.nmnist = Some(val.into());
            Ok(true)
        }
        _ => set.flag(flag, val),
    })?;
    if o.nmnist.is_some() && (a.mnist.is_some() || a.png.is_some()) {
        return Err("--nmnist replaces --mnist and --png; give one source".into());
    }
    set.apply(|k, v| o.params.set(k, v))?;
    Ok((a, o))
}

/// Rows of the dashboard's model panel; `source` says where the events come from.
fn model_params(p: &Params, side: usize, source: Vec<Param<'static>>) -> Vec<Param<'static>> {
    let n = (side - p.rf) / p.stride + 1;
    let m = n + 1 - p.complex_rf;
    let mut rows = vec![
        Param::text("Retina", format!("{side}² · ON/OFF")),
        Param::text("Simple", format!("{n}² × {}", p.features)),
        Param::text("Complex", format!("{m}² × {}", p.complex_features)),
    ];
    rows.extend(source);
    rows.extend([
        Param::num("τ simple", p.tau_m, "ms"),
        Param::num("Threshold", p.v_thr, ""),
        Param::num("Lateral inh.", p.norm_lateral, ""),
        Param::num("Top-down inh.", p.norm_top_down, ""),
    ]);
    rows
}

fn run(a: Args, o: Opts) -> Result<(), String> {
    let Some(dir) = o.nmnist else {
        let (train, test) = app::load_data(&a)?;
        return session(a, o.params, train, test);
    };
    let load = |train, limit| {
        Recordings::load(&dir, train, limit).map_err(|e| format!("{}: {e}", dir.display()))
    };
    let (train, test) = (load(true, a.train)?, load(false, a.test)?);
    if train.is_empty() {
        return Err(format!("{}: no training recordings", dir.display()));
    }
    println!(
        "{}: {} train, {} test, up to {} ms, {} classes",
        train.name,
        train.len(),
        test.len(),
        train.steps(),
        train.classes
    );
    session(a, o.params, train, test)
}

// recordings have no pixels to add noise to
fn robustness<D: Input>(a: &Args, t: &mut Trainer, test: &D) {
    let ctl = Control::new(0);
    let images = test.len() > 0 && test.pixels(0).is_some();
    app::robustness(a, |d| {
        if d.as_ref().is_some_and(|d| d.kind == Kind::Noise) && !images {
            return None;
        }
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

fn session<D: Input + Send + 'static>(a: Args, p: Params, train: D, test: D) -> Result<(), String> {
    let classes = train.classes().max(test.classes());
    let saccades = Saccades::default();
    let side = train.retina(&saccades);
    let mut trainer = Trainer::with_retina(p.clone(), side, classes, a.seed);
    let epochs = a.epochs();
    let printer = Printer::new(train.len()).dense(trainer.dense_macs());

    if a.headless {
        let ctl = Control::new(0);
        let test_set = (!test.is_empty()).then_some(&test);
        trainer.fit(&train, test_set, epochs, &mut { printer }, &ctl);
        robustness(&a, &mut trainer, &test);
        return Ok(());
    }

    let (width, height) = train.picture();
    let meta = Meta {
        model: "pcl",
        dataset: train.name(),
        width,
        height,
        classes,
        n_train: train.len(),
        n_test: test.len(),
        epochs,
        steps: train.steps(&saccades),
        dt_ms: 1.0,
        dense_macs: trainer.dense_macs(),
    };
    let (net, k) = (&trainer.net, p.features);
    let pops = [
        Pop::grid("on", side, side, Role::Input),
        Pop::grid("off", side, side, Role::Input),
        Pop::grid("simple", net.n * k / 4, 4 * net.n, Role::Latent),
        Pop::grid(
            "complex",
            net.m * 4,
            net.m * p.complex_features / 4,
            Role::Latent,
        ),
    ];
    let rows = model_params(&p, side, train.source(&saccades));
    let mut packet = app::meta_packet(&meta, "Model · PCL", &rows, &pops);
    let (step, probe, image) = Tap::series();
    let mut b = Buf::new();
    frame::series(&mut b, &step, &probe, &image);
    packet.extend(b.take());

    let args = a.clone();
    app::serve_live(&a, packet, move |hub, ctl| {
        let tap = Tap::new(hub, ctl.clone());
        let test_set = (!test.is_empty()).then_some(&test);
        trainer.fit(&train, test_set, epochs, &mut (tap, printer), &ctl);
        robustness(&args, &mut trainer, &test);
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
                "usage: mnist-pcl-rs [options]\n{}\n{USAGE}\n{}",
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
    use snn_core::{dataset, nmnist};

    fn parse(args: &[&str]) -> Result<(Args, Opts), String> {
        parse_args(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn no_inhibition_and_set() {
        let defaults = Params::default();
        let (a, o) = parse(&["--headless", "--set", "v_thr=12"]).unwrap();
        let p = o.params;
        assert!(a.headless);
        assert_eq!((p.v_thr, p.norm_lateral), (12.0, defaults.norm_lateral));
        let (_, o) = parse(&["--no-inhibition"]).unwrap();
        assert_eq!(
            (o.params.norm_lateral, o.params.norm_top_down),
            (1e-3, 1e-3)
        );
        let (_, o) = parse(&["--set", "norm_lateral=5", "--no-inhibition"]).unwrap();
        assert_eq!(
            (o.params.norm_lateral, o.params.norm_top_down),
            (5.0, 1e-3),
            "--set wins"
        );
        assert!(parse(&["--set", "stride=0.5"]).is_err());
        assert!(parse(&["--bogus", "1"])
            .unwrap_err()
            .contains("unknown option"));
    }

    #[test]
    fn nmnist_is_a_source_of_its_own() {
        let (a, o) = parse(&["--nmnist", "rec", "--train", "50"]).unwrap();
        assert_eq!((o.nmnist, a.train), (Some("rec".into()), Some(50)));
        let err = parse(&["--nmnist", "rec", "--mnist", "data"]).unwrap_err();
        assert!(err.contains("give one source"), "{err}");
    }

    #[test]
    fn trains_from_recordings_on_disk() {
        let root = std::env::temp_dir().join(format!("snn-pcl-main-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        let s = Saccades::default();
        nmnist::export(&dataset::synthetic(60, 1), &s, &root, "Train").unwrap();
        nmnist::export(&dataset::synthetic(30, 2), &s, &root, "Test").unwrap();
        let dir = root.to_str().unwrap();
        let (a, o) = parse(&[
            "--headless",
            "--nmnist",
            dir,
            "--train",
            "40",
            "--test",
            "20",
        ])
        .unwrap();
        let result = run(a, o);
        let (a, o) = parse(&["--headless", "--nmnist", "/no/such/dir"]).unwrap();
        let missing = run(a, o).unwrap_err();
        std::fs::remove_dir_all(&root).ok();
        result.unwrap();
        assert!(missing.contains("/no/such/dir"), "{missing}");
    }
}
