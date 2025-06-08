use mnist_snn::tap::Tap;
use mnist_snn::train::Trainer;
use mnist_snn::{Layer, Params};
use snn_core::{weights, Control, Dataset, Rng};
use snn_live::app::{self, Args, Printer};
use snn_live::frame::{Meta, Param, Pop, Role};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "  --outputs N        output neurons (default 40)
  --load FILE        start from saved weights
  --save FILE        save weights after training
  --weights-png FILE save weight maps as an image";

#[derive(Debug, Clone)]
struct Opts {
    outputs: usize,
    load: Option<PathBuf>,
    save: Option<PathBuf>,
    weights_png: Option<PathBuf>,
}

fn parse_args() -> Result<(Args, Opts), String> {
    let mut o = Opts {
        outputs: 40,
        load: None,
        save: None,
        weights_png: None,
    };
    let a = Args::parse(std::env::args().skip(1), |flag, val| {
        match flag {
            "--outputs" => {
                let n: usize = val
                    .parse()
                    .map_err(|_| format!("{flag}: bad value {val}"))?;
                o.outputs = n.max(1);
            }
            "--load" => o.load = Some(val.into()),
            "--save" => o.save = Some(val.into()),
            "--weights-png" => o.weights_png = Some(val.into()),
            _ => return Ok(false),
        }
        Ok(true)
    })?;
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
    let Some(path) = &o.load else {
        return Ok(Trainer::new(
            params.clone(),
            n_in,
            o.outputs,
            classes,
            a.seed,
        ));
    };
    let (rows, cols, w) = weights::load(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if cols != n_in {
        return Err(format!(
            "{}: {cols} inputs, images have {n_in}",
            path.display()
        ));
    }
    let layer = Layer::with_weights(n_in, rows, w, params);
    let mut t = Trainer::with_layer(params.clone(), layer, classes, Rng::new(a.seed));
    t.assign_labels(train);
    Ok(t)
}

fn save_outputs(o: &Opts, t: &Trainer, side: usize) -> Result<(), String> {
    let l = &t.layer;
    if let Some(path) = &o.save {
        weights::save(path, l.n_out, l.n_in, l.weights()).map_err(|e| e.to_string())?;
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
    let params = Params::default();
    let classes = train.classes.max(test.classes);
    let mut trainer = build_trainer(&a, &o, &params, &train, classes)?;
    let epochs = a.epochs();
    let printer = Printer::new(train.len());

    if a.headless {
        let ctl = Control::new(0);
        let test_set = (!test.is_empty()).then_some(&test);
        if epochs == 0 {
            if let Some(t) = test_set {
                println!("test {:.1}%", trainer.evaluate(t, 0, &mut (), &ctl) * 100.0);
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
    let (args, opts) = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("{e}");
            }
            eprintln!("usage: mnist-snn-rs [options]\n{}\n{USAGE}", app::USAGE);
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
