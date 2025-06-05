use snn::dataset::{self, Dataset};
use snn::train::{Control, EpochResult, ImageResult, Labels, Monitor, Trainer};
use snn::{weights, Layer, Params, Rng};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "\
usage: snn [options]
  --mnist DIR        MNIST idx files (plain or .gz) in DIR
  --png DIR          images in DIR/<class>/*.png
                     (default: synthetic digits)
  --train N          training images
  --test N           test images
  --epochs N         passes over the training set (default 1)
  --outputs N        output neurons (default 40)
  --seed N           random seed (default 1)
  --load FILE        start from saved weights
  --save FILE        save weights after training
  --weights-png FILE save weight maps as an image";

#[derive(Debug)]
struct Args {
    mnist: Option<PathBuf>,
    png: Option<PathBuf>,
    train: Option<usize>,
    test: Option<usize>,
    epochs: usize,
    outputs: usize,
    seed: u64,
    load: Option<PathBuf>,
    save: Option<PathBuf>,
    weights_png: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        mnist: None,
        png: None,
        train: None,
        test: None,
        epochs: 1,
        outputs: 40,
        seed: 1,
        load: None,
        save: None,
        weights_png: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "-h" || flag == "--help" {
            return Err(String::new());
        }
        let val = it.next().ok_or(format!("{flag} needs a value"))?;
        let num = |v: &str| {
            v.parse::<usize>()
                .map_err(|_| format!("{flag}: not a number: {v}"))
        };
        match flag.as_str() {
            "--mnist" => a.mnist = Some(val.into()),
            "--png" => a.png = Some(val.into()),
            "--train" => a.train = Some(num(&val)?),
            "--test" => a.test = Some(num(&val)?),
            "--epochs" => a.epochs = num(&val)?,
            "--outputs" => a.outputs = num(&val)?.max(1),
            "--seed" => a.seed = num(&val)? as u64,
            "--load" => a.load = Some(val.into()),
            "--save" => a.save = Some(val.into()),
            "--weights-png" => a.weights_png = Some(val.into()),
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    Ok(a)
}

fn load_data(a: &Args) -> Result<(Dataset, Dataset), String> {
    let (mut train, mut test) = if let Some(dir) = &a.mnist {
        let train = dataset::mnist(dir, true).map_err(|e| e.to_string())?;
        let test = dataset::mnist(dir, false).map_err(|e| e.to_string())?;
        (train, test)
    } else if let Some(dir) = &a.png {
        let mut all = dataset::png_dir(dir, 28).map_err(|e| e.to_string())?;
        all.shuffle(&mut Rng::new(a.seed));
        let n_test = a.test.unwrap_or(all.len() / 5);
        let test = all.split_off(n_test);
        (all, test)
    } else {
        let train = dataset::synthetic(a.train.unwrap_or(6000), a.seed);
        let test = dataset::synthetic(a.test.unwrap_or(1000), a.seed ^ 0x5eed);
        (train, test)
    };
    train.truncate(a.train.unwrap_or(usize::MAX));
    test.truncate(a.test.unwrap_or(usize::MAX));
    if train.is_empty() && a.epochs > 0 {
        return Err("no training images".into());
    }
    Ok((train, test))
}

/// Prints progress lines while training headless.
struct Progress {
    every: usize,
    t0: Instant,
}

impl Monitor for Progress {
    fn image_end(&mut self, r: &ImageResult, _layer: &Layer, _labels: &Labels) {
        let i = r.info;
        if i.learn && (i.index + 1) % self.every == 0 {
            let rate = self.every as f32 / self.t0.elapsed().as_secs_f32();
            println!(
                "epoch {}  {:>6}/{}  running acc {:5.1}%  {:6.0} img/s",
                i.epoch + 1,
                i.index + 1,
                i.total,
                r.running_accuracy * 100.0,
                rate
            );
            self.t0 = Instant::now();
        }
    }

    fn epoch_end(&mut self, e: &EpochResult) {
        let test = e
            .test_accuracy
            .map_or("-".into(), |a| format!("{:.1}%", a * 100.0));
        println!(
            "epoch {} done: train {:.1}%  test {}  ({:.0} img/s)",
            e.epoch + 1,
            e.train_accuracy * 100.0,
            test,
            e.images_per_sec
        );
    }
}

fn run(a: Args) -> Result<(), String> {
    let (train, test) = load_data(&a)?;
    println!(
        "{}: {} train, {} test, {}x{}, {} classes",
        train.name,
        train.len(),
        test.len(),
        train.width,
        train.height,
        train.classes
    );
    let params = Params::default();
    let n_in = train.width * train.height;
    let classes = train.classes.max(test.classes);
    let mut trainer = match &a.load {
        Some(path) => {
            let (rows, cols, w) =
                weights::load(path).map_err(|e| format!("{}: {e}", path.display()))?;
            if cols != n_in {
                return Err(format!(
                    "{}: {cols} inputs, images have {n_in}",
                    path.display()
                ));
            }
            let layer = Layer::with_weights(n_in, rows, w, &params);
            let mut t = Trainer::with_layer(params.clone(), layer, classes, Rng::new(a.seed));
            t.assign_labels(&train);
            t
        }
        None => Trainer::new(params.clone(), n_in, a.outputs, classes, a.seed),
    };

    let ctl = Control::new(0);
    let mut progress = Progress {
        every: (train.len() / 10).max(1),
        t0: Instant::now(),
    };
    let test_set = (!test.is_empty()).then_some(&test);
    if a.epochs == 0 {
        if let Some(t) = test_set {
            let acc = trainer.evaluate(t, 0, &mut (), &ctl);
            println!("test {:.1}%", acc * 100.0);
        }
    }
    trainer.fit(&train, test_set, a.epochs, &mut progress, &ctl);

    let l = &trainer.layer;
    if let Some(path) = &a.save {
        weights::save(path, l.n_out, l.n_in, &l.weights).map_err(|e| e.to_string())?;
        println!("weights saved to {}", path.display());
    }
    if let Some(path) = &a.weights_png {
        let (lo, hi) = (params.stdp.w_min, params.stdp.w_max);
        weights::save_png(path, &l.weights, train.width, lo, hi).map_err(|e| e.to_string())?;
        println!("weight maps saved to {}", path.display());
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("{e}");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
