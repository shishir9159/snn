use mnist_pcl::tap::Tap;
use mnist_pcl::train::Trainer;
use mnist_pcl::Params;
use snn_core::{Control, Overrides};
use snn_live::app::{self, Args, Printer};
use snn_live::frame::{self, Buf, Meta, Param, Pop, Role};
use std::process::ExitCode;

const USAGE: &str = "  --no-inhibition    switch plastic lateral and top-down inhibition off";

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<(Args, Params), String> {
    let mut p = Params::default();
    let mut args: Vec<String> = args.into_iter().collect();
    // the one flag without a value
    if let Some(i) = args.iter().position(|a| a == "--no-inhibition") {
        args.remove(i);
        p.norm_lateral = 1e-3;
        p.norm_top_down = 1e-3;
    }
    let mut set = Overrides::default();
    let a = Args::parse(args, |flag, val| set.flag(flag, val))?;
    set.apply(|k, v| p.set(k, v))?;
    Ok((a, p))
}

/// Rows of the dashboard's model panel.
fn model_params(p: &Params, side: usize, leg: usize, amp: f32) -> Vec<Param<'static>> {
    let n = (side - p.rf) / p.stride + 1;
    let m = n + 1 - p.complex_rf;
    vec![
        Param::text("Retina", format!("{side}² · ON/OFF")),
        Param::text("Simple", format!("{n}² × {}", p.features)),
        Param::text("Complex", format!("{m}² × {}", p.complex_features)),
        Param::num("Saccade px", amp, ""),
        Param::num("Saccade leg", leg as f32, "ms"),
        Param::num("τ simple", p.tau_m, "ms"),
        Param::num("Threshold", p.v_thr, ""),
        Param::num("Lateral inh.", p.norm_lateral, ""),
        Param::num("Top-down inh.", p.norm_top_down, ""),
    ]
}

fn run(a: Args, p: Params) -> Result<(), String> {
    let (train, test) = app::load_data(&a)?;
    let classes = train.classes.max(test.classes);
    let mut trainer = Trainer::new(p.clone(), train.width, train.height, classes, a.seed);
    let epochs = a.epochs();
    let printer = Printer::new(train.len());

    if a.headless {
        let ctl = Control::new(0);
        let test_set = (!test.is_empty()).then_some(&test);
        trainer.fit(&train, test_set, epochs, &mut { printer }, &ctl);
        return Ok(());
    }

    let (side, s) = (trainer.net.side, trainer.saccades.clone());
    let meta = Meta {
        model: "pcl",
        dataset: &train.name,
        width: train.width,
        height: train.height,
        classes,
        n_train: train.len(),
        n_test: test.len(),
        epochs,
        steps: s.steps(),
        dt_ms: 1.0,
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
    let rows = model_params(&p, side, s.leg, s.amplitude);
    let mut packet = app::meta_packet(&meta, "Model · PCL", &rows, &pops);
    let (step, probe, image) = Tap::series();
    let mut b = Buf::new();
    frame::series(&mut b, &step, &probe, &image);
    packet.extend(b.take());

    app::serve_live(&a, packet, move |hub, ctl| {
        let tap = Tap::new(hub, ctl.clone());
        let test_set = (!test.is_empty()).then_some(&test);
        trainer.fit(&train, test_set, epochs, &mut (tap, printer), &ctl);
    })
}

fn main() -> ExitCode {
    let (args, params) = match parse_args(std::env::args().skip(1)) {
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
    match run(args, params) {
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

    fn parse(args: &[&str]) -> Result<(Args, Params), String> {
        parse_args(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn no_inhibition_and_set() {
        let defaults = Params::default();
        let (a, p) = parse(&["--headless", "--set", "v_thr=12"]).unwrap();
        assert!(a.headless);
        assert_eq!((p.v_thr, p.norm_lateral), (12.0, defaults.norm_lateral));
        let (_, p) = parse(&["--no-inhibition"]).unwrap();
        assert_eq!((p.norm_lateral, p.norm_top_down), (1e-3, 1e-3));
        let (_, p) = parse(&["--set", "norm_lateral=5", "--no-inhibition"]).unwrap();
        assert_eq!((p.norm_lateral, p.norm_top_down), (5.0, 1e-3), "--set wins");
        assert!(parse(&["--set", "stride=0.5"]).is_err());
        assert!(parse(&["--bogus", "1"])
            .unwrap_err()
            .contains("unknown option"));
    }
}
