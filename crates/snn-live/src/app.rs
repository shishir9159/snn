//! What every model binary shares: command-line flags, dataset loading,
//! progress lines and the live-view harness.

use crate::{frame, server, Hub};
use snn_core::dataset::{self, Dataset};
use snn_core::{Control, EpochResult, ImageInfo, Rng};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

pub const USAGE: &str = "  --mnist DIR        MNIST idx files (plain or .gz) in DIR
  --png DIR          images in DIR/<class>/*.png
                     (default: synthetic digits)
  --train N          training images
  --test N           test images
  --epochs N         passes over the training set (default 1 headless, 10 live)
  --seed N           random seed (default 1)
  --headless         train without the live view
  --port N           live view port (default 7878)
  --speed X          multiple of real time, 1 ms per step; or max (default 1)";

/// One simulated step per millisecond.
const REAL_TIME: f32 = 1000.0;

#[derive(Debug, Clone)]
pub struct Args {
    pub mnist: Option<PathBuf>,
    pub png: Option<PathBuf>,
    pub train: Option<usize>,
    pub test: Option<usize>,
    pub epochs: Option<usize>,
    pub seed: u64,
    pub headless: bool,
    pub port: u16,
    /// steps per second, 0 for unthrottled
    pub speed: u32,
}

impl Args {
    /// Parses the shared flags; any other flag and its value go to `extra`,
    /// which returns whether it knew the flag. An empty error asks for usage.
    pub fn parse<F>(args: impl IntoIterator<Item = String>, mut extra: F) -> Result<Args, String>
    where
        F: FnMut(&str, &str) -> Result<bool, String>,
    {
        let mut a = Args {
            mnist: None,
            png: None,
            train: None,
            test: None,
            epochs: None,
            seed: 1,
            headless: false,
            port: 7878,
            speed: REAL_TIME as u32,
        };
        let mut it = args.into_iter();
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
                "--seed" => a.seed = num(&val)? as u64,
                "--port" => a.port = val.parse().map_err(|_| bad())?,
                "--speed" if val == "max" => a.speed = 0,
                "--speed" => {
                    let x: f32 = val.parse().map_err(|_| bad())?;
                    a.speed = (x * REAL_TIME).max(1.0) as u32;
                }
                _ if extra(&flag, &val)? => {}
                _ => return Err(format!("unknown option {flag}")),
            }
        }
        Ok(a)
    }

    pub fn epochs(&self) -> usize {
        self.epochs.unwrap_or(if self.headless { 1 } else { 10 })
    }
}

/// Training and test sets the flags ask for.
pub fn load_data(a: &Args) -> Result<(Dataset, Dataset), String> {
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
    println!(
        "{}: {} train, {} test, {}x{}, {} classes",
        train.name,
        train.len(),
        test.len(),
        train.width,
        train.height,
        train.classes
    );
    Ok((train, test))
}

/// Prints a progress line every tenth of an epoch and a summary per epoch.
pub struct Printer {
    every: usize,
    t0: Instant,
}

impl Printer {
    pub fn new(train_len: usize) -> Self {
        Self {
            every: (train_len / 10).max(1),
            t0: Instant::now(),
        }
    }

    pub fn image(&mut self, i: &ImageInfo, running_accuracy: f32) {
        if i.learn && (i.index + 1) % self.every == 0 {
            let rate = self.every as f32 / self.t0.elapsed().as_secs_f32();
            println!(
                "epoch {}  {:>6}/{}  running acc {:5.1}%  {:6.0} img/s",
                i.epoch + 1,
                i.index + 1,
                i.total,
                running_accuracy * 100.0,
                rate
            );
            self.t0 = Instant::now();
        }
    }

    pub fn epoch(&mut self, e: &EpochResult) {
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

/// Serves the dashboard and runs `work` on its own thread; keeps serving after
/// it returns. `meta` is what every viewer receives first.
pub fn serve_live<F>(a: &Args, meta: Vec<u8>, work: F) -> Result<(), String>
where
    F: FnOnce(Arc<Hub>, Arc<Control>) + Send + 'static,
{
    let (hub, ctl) = start_live(a.port, a.speed, meta)?;
    let (h, c) = (hub.clone(), ctl.clone());
    std::thread::spawn(move || work(h, c))
        .join()
        .map_err(|_| "trainer panicked".to_string())?;
    finish_live(&hub, &ctl);
    loop {
        std::thread::park();
    }
}

/// Binds `port` and serves the dashboard on a background thread.
pub fn start_live(
    port: u16,
    speed: u32,
    meta: Vec<u8>,
) -> Result<(Arc<Hub>, Arc<Control>), String> {
    let hub = Arc::new(Hub::new());
    let ctl = Arc::new(Control::new(speed));
    hub.set_meta(meta);
    let listener =
        TcpListener::bind(("127.0.0.1", port)).map_err(|e| format!("port {port}: {e}"))?;
    println!("live view: http://127.0.0.1:{port}");
    let (h, c) = (hub.clone(), ctl.clone());
    std::thread::spawn(move || server::serve(listener, h, c));
    Ok((hub, ctl))
}

/// Tells viewers training is over.
pub fn finish_live(hub: &Hub, ctl: &Control) {
    ctl.finish();
    server::publish_state(hub, ctl);
}

/// Builds the packet every viewer starts with.
pub fn meta_packet(
    m: &frame::Meta,
    title: &str,
    ps: &[frame::Param],
    pops: &[frame::Pop],
) -> Vec<u8> {
    let mut b = frame::Buf::new();
    frame::meta(&mut b, m);
    frame::params(&mut b, title, ps);
    frame::pops(&mut b, pops);
    b.take()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Result<Args, String> {
        Args::parse(s.split_whitespace().map(String::from), |f, v| {
            Ok(f == "--outputs" && v == "12")
        })
    }

    #[test]
    fn shared_and_model_flags() {
        let a = parse("--headless --train 5 --outputs 12 --speed max").unwrap();
        assert!(a.headless);
        assert_eq!(a.train, Some(5));
        assert_eq!(a.speed, 0);
        assert_eq!(a.epochs(), 1);
        assert_eq!(parse("--speed 2").unwrap().speed, 2000);
        assert!(parse("--bogus 1").unwrap_err().contains("unknown"));
        assert_eq!(parse("--help").unwrap_err(), "");
    }
}
