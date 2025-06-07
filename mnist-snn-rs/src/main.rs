use mnist_snn::train::{ImageResult, Labels, Monitor, Trainer};
use mnist_snn::viz::{frame, server, Hub, Tap};
use mnist_snn::{Layer, Params};
use snn_core::dataset::{self, Dataset};
use snn_core::{weights, Control, EpochResult, Rng};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "\
usage: mnist-snn-rs [options]
  --mnist DIR        MNIST idx files (plain or .gz) in DIR
  --png DIR          images in DIR/<class>/*.png
                     (default: synthetic digits)
  --train N          training images
  --test N           test images
  --epochs N         passes over the training set (default 1 headless, 10 live)
  --outputs N        output neurons (default 40)
  --seed N           random seed (default 1)
  --load FILE        start from saved weights
  --save FILE        save weights after training
  --weights-png FILE save weight maps as an image
  --headless         train without the live view
  --port N           live view port (default 7878)
  --speed X          multiple of real time, 1 ms per step; or max (default 1)";

/// One simulated step per millisecond.
const REAL_TIME: f32 = 1000.0;

#[derive(Debug, Clone)]
struct Args {
    mnist: Option<PathBuf>,
    png: Option<PathBuf>,
    train: Option<usize>,
    test: Option<usize>,
    epochs: Option<usize>,
    outputs: usize,
    seed: u64,
    load: Option<PathBuf>,
    save: Option<PathBuf>,
    weights_png: Option<PathBuf>,
    headless: bool,
    port: u16,
    /// steps per second, 0 for unthrottled
    speed: u32,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        mnist: None,
        png: None,
        train: None,
        test: None,
        epochs: None,
        outputs: 40,
        seed: 1,
        load: None,
        save: None,
        weights_png: None,
        headless: false,
        port: 7878,
        speed: REAL_TIME as u32,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "-h" | "--help" => return Err(String::new()),
            "--headless" => {
                a.headless = true;
                continue;
            }
            _ => {}
        }
        let val = it.next().ok_or(format!("{flag} needs a value"))?;
        let bad = || format!("{flag}: bad value {val}");
        let num = |v: &str| v.parse::<usize>().map_err(|_| bad());
        match flag.as_str() {
            "--mnist" => a.mnist = Some(val.into()),
            "--png" => a.png = Some(val.into()),
            "--train" => a.train = Some(num(&val)?),
            "--test" => a.test = Some(num(&val)?),
            "--epochs" => a.epochs = Some(num(&val)?),
            "--outputs" => a.outputs = num(&val)?.max(1),
            "--seed" => a.seed = num(&val)? as u64,
            "--load" => a.load = Some(val.into()),
            "--save" => a.save = Some(val.into()),
            "--weights-png" => a.weights_png = Some(val.into()),
            "--port" => a.port = val.parse().map_err(|_| bad())?,
            "--speed" if val == "max" => a.speed = 0,
            "--speed" => {
                let x: f32 = val.parse().map_err(|_| bad())?;
                a.speed = (x * REAL_TIME).max(1.0) as u32;
            }
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
    if train.is_empty() {
        return Err("no training images".into());
    }
    Ok((train, test))
}

/// Prints progress lines while training.
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
        self.t0 = Instant::now();
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

fn build_trainer(
    a: &Args,
    params: &Params,
    train: &Dataset,
    classes: usize,
) -> Result<Trainer, String> {
    let n_in = train.width * train.height;
    let Some(path) = &a.load else {
        return Ok(Trainer::new(
            params.clone(),
            n_in,
            a.outputs,
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

fn save_outputs(a: &Args, t: &Trainer, side: usize) -> Result<(), String> {
    let l = &t.layer;
    if let Some(path) = &a.save {
        weights::save(path, l.n_out, l.n_in, l.weights()).map_err(|e| e.to_string())?;
        println!("weights saved to {}", path.display());
    }
    if let Some(path) = &a.weights_png {
        let (lo, hi) = (t.params.stdp.w_min, t.params.stdp.w_max);
        weights::save_png(path, l.weights(), side, lo, hi).map_err(|e| e.to_string())?;
        println!("weight maps saved to {}", path.display());
    }
    Ok(())
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
    let classes = train.classes.max(test.classes);
    let mut trainer = build_trainer(&a, &params, &train, classes)?;
    let epochs = a.epochs.unwrap_or(if a.headless { 1 } else { 10 });
    let progress = Progress {
        every: (train.len() / 10).max(1),
        t0: Instant::now(),
    };

    if a.headless {
        let ctl = Control::new(0);
        let test_set = (!test.is_empty()).then_some(&test);
        if epochs == 0 {
            if let Some(t) = test_set {
                println!("test {:.1}%", trainer.evaluate(t, 0, &mut (), &ctl) * 100.0);
            }
        }
        trainer.fit(&train, test_set, epochs, &mut { progress }, &ctl);
        return save_outputs(&a, &trainer, train.width);
    }

    let hub = Arc::new(Hub::new());
    let ctl = Arc::new(Control::new(a.speed));
    let mut b = frame::Buf::new();
    let meta = frame::Meta {
        dataset: &train.name,
        width: train.width,
        height: train.height,
        n_out: trainer.layer.n_out,
        classes,
        n_train: train.len(),
        n_test: test.len(),
        epochs,
        params: &params,
    };
    frame::meta(&mut b, &meta);
    hub.set_meta(b.take());

    let listener =
        TcpListener::bind(("127.0.0.1", a.port)).map_err(|e| format!("port {}: {e}", a.port))?;
    println!("live view: http://127.0.0.1:{}", a.port);

    let (h, c) = (hub.clone(), ctl.clone());
    let worker = std::thread::spawn(move || {
        let tap = Tap::new(h.clone(), params.stdp.w_min, params.stdp.w_max);
        let test_set = (!test.is_empty()).then_some(&test);
        trainer.fit(&train, test_set, epochs, &mut (tap, progress), &c);
        c.finish();
        server::publish_state(&h, &c);
        if let Err(e) = save_outputs(&a, &trainer, train.width) {
            eprintln!("error: {e}");
        }
    });
    server::serve(listener, hub, ctl);
    worker.join().map_err(|_| "trainer panicked".to_string())
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
