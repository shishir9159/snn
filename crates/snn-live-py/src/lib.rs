//! Python module `snn_live`: the live view for models written in Python. It
//! wraps the same server, hub and population tap the Rust models use, so a
//! Python trainer hands over numpy arrays and never encodes records itself.

// keyword arguments on the Python side
#![allow(clippy::too_many_arguments)]

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use snn_core::{Control, EpochResult, ImageInfo, Pacer};
use snn_live::frame::{self, Buf, Param, Pop, Role};
use snn_live::pop::End;
use snn_live::{app, Hub, PopTap};
use std::net::Ipv4Addr;
use std::sync::Arc;
use std::time::Duration;

/// One simulated step per millisecond.
const REAL_TIME: f32 = 1000.0;

type Info = (usize, usize, usize, u8, bool);

fn image_info((epoch, index, total, label, learn): Info) -> ImageInfo {
    ImageInfo {
        epoch,
        index,
        total,
        label,
        learn,
    }
}

fn role(name: &str) -> PyResult<Role> {
    Ok(match name {
        "input" => Role::Input,
        "latent" => Role::Latent,
        "prediction" => Role::Prediction,
        "error" => Role::Error,
        "label" => Role::Label,
        "output" => Role::Output,
        _ => return Err(PyValueError::new_err(format!("unknown role {name}"))),
    })
}

fn item<'py, T: FromPyObject<'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("meta needs {key}")))?
        .extract()
}

/// META, PARAMS, POPS and SERIES: what every viewer starts with.
fn intro(
    meta: &Bound<'_, PyDict>,
    title: &str,
    params: &[Bound<'_, PyTuple>],
    pops: &[Bound<'_, PyTuple>],
    [step, probe, image]: [Vec<String>; 3],
) -> PyResult<Vec<u8>> {
    let (model, dataset): (String, String) = (item(meta, "model")?, item(meta, "dataset")?);
    let m = frame::Meta {
        model: &model,
        dataset: &dataset,
        width: item(meta, "width")?,
        height: item(meta, "height")?,
        classes: item(meta, "classes")?,
        n_train: item(meta, "n_train")?,
        n_test: item(meta, "n_test")?,
        epochs: item(meta, "epochs")?,
        steps: item(meta, "steps")?,
        dt_ms: meta.get_item("dt_ms")?.map_or(Ok(1.0), |v| v.extract())?,
    };

    // (name, number or text, unit)
    let mut rows = Vec::new();
    for p in params {
        let name: String = p.get_item(0)?.extract()?;
        let unit: String = match p.len() {
            3 => p.get_item(2)?.extract()?,
            _ => String::new(),
        };
        rows.push((name, p.get_item(1)?, unit));
    }
    let mut ps = Vec::new();
    for (name, v, unit) in &rows {
        ps.push(match v.extract::<f32>() {
            Ok(x) => Param::num(name, x, unit),
            Err(_) => Param::text(name, v.str()?.to_string()),
        });
    }

    // (name, size, role) or (name, w, h, role)
    let mut names = Vec::new();
    for p in pops {
        names.push(p.get_item(0)?.extract::<String>()?);
    }
    let mut ns = Vec::new();
    for (p, name) in pops.iter().zip(&names) {
        let r = role(&p.get_item(p.len() - 1)?.extract::<String>()?)?;
        ns.push(match p.len() {
            3 => Pop::new(name, p.get_item(1)?.extract()?, r),
            4 => Pop::grid(
                name,
                p.get_item(1)?.extract()?,
                p.get_item(2)?.extract()?,
                r,
            ),
            _ => {
                return Err(PyValueError::new_err(
                    "pop is (name, size, role) or (name, w, h, role)",
                ))
            }
        });
    }

    let mut packet = app::meta_packet(&m, title, &ps, &ns);
    fn strs(v: &[String]) -> Vec<&str> {
        v.iter().map(String::as_str).collect()
    }
    let mut b = Buf::new();
    frame::series(&mut b, &strs(&step), &strs(&probe), &strs(&image));
    packet.extend(b.take());
    Ok(packet)
}

/// Live dashboard for one training run, served at http://127.0.0.1:`port`.
///
/// Per presentation: `begin_image`, then `step` for every simulated step,
/// `map` for extra images, `fields` when `fields_due`, then `end_image`.
/// Arrays are numpy (or any buffer): pixels uint8, spike indices uint16,
/// everything else float32. Only streamed presentations are encoded; check
/// `streaming` before copying spikes off a GPU.
#[pyclass(module = "snn_live")]
struct Live {
    hub: Arc<Hub>,
    ctl: Arc<Control>,
    tap: PopTap,
    pacer: Pacer,
    /// latest FIELDS records; reused until the model sends new ones
    fields: Vec<u8>,
    fresh: bool,
}

#[pymethods]
impl Live {
    /// `meta`: model, dataset, width, height, classes, n_train, n_test, epochs,
    /// steps, dt_ms. `speed` is a multiple of real time (1 ms per step), None
    /// for unthrottled.
    #[new]
    #[pyo3(signature = (meta, pops, *, title = "Model", params = vec![], series = None, port = 7878, speed = Some(1.0)))]
    fn new(
        meta: &Bound<'_, PyDict>,
        pops: Vec<Bound<'_, PyTuple>>,
        title: &str,
        params: Vec<Bound<'_, PyTuple>>,
        series: Option<(Vec<String>, Vec<String>, Vec<String>)>,
        port: u16,
        speed: Option<f32>,
    ) -> PyResult<Self> {
        let (step, probe, image) = series.unwrap_or_default();
        let packet = intro(meta, title, &params, &pops, [step, probe, image])?;
        let speed = speed.map_or(0, |x| (x * REAL_TIME).max(1.0) as u32);
        let (hub, ctl) = app::start_live(Ipv4Addr::LOCALHOST.into(), port, speed, packet)
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            tap: PopTap::new(hub.clone(), ctl.clone()),
            hub,
            ctl,
            pacer: Pacer::new(),
            fields: Vec::new(),
            fresh: false,
        })
    }

    #[getter]
    fn viewers(&self) -> usize {
        self.hub.viewers()
    }

    #[getter]
    fn streaming(&self) -> bool {
        self.tap.streaming()
    }

    #[getter]
    fn fields_due(&self) -> bool {
        self.tap.fields_due()
    }

    #[getter]
    fn stopped(&self) -> bool {
        self.ctl.stopped()
    }

    /// The (population, unit) a viewer clicked, or None.
    fn probe_request(&self) -> Option<(usize, usize)> {
        self.tap.probe_request()
    }

    /// `info` is (epoch, index, total, label, learn); `probe` is
    /// (population, unit, threshold). Returns whether this presentation streams.
    fn begin_image(
        &mut self,
        py: Python<'_>,
        info: Info,
        pixels: PyBuffer<u8>,
        probe: (usize, usize, f32),
    ) -> PyResult<bool> {
        Ok(self
            .tap
            .begin_image(&image_info(info), &pixels.to_vec(py)?, probe))
    }

    /// One simulated step, held to the dashboard's speed. `spikes` holds the
    /// spiking units of each population; it may be None when not streaming.
    #[pyo3(signature = (t, spikes = None, scalars = None, probe = None))]
    fn step(
        &mut self,
        py: Python<'_>,
        t: usize,
        spikes: Option<Vec<PyBuffer<u16>>>,
        scalars: Option<PyBuffer<f32>>,
        probe: Option<PyBuffer<f32>>,
    ) -> PyResult<()> {
        if self.ctl.speed() != 0 || self.ctl.is_paused() {
            let (pacer, ctl) = (&mut self.pacer, &self.ctl);
            py.allow_threads(|| pacer.tick(ctl));
        }
        if !self.tap.streaming() {
            self.tap.step(t, &[], &[], &[]);
            return Ok(());
        }
        let spikes = spikes.ok_or_else(|| PyValueError::new_err("streaming step needs spikes"))?;
        let pops = spikes
            .iter()
            .map(|b| b.to_vec(py))
            .collect::<PyResult<Vec<_>>>()?;
        let pops: Vec<&[u16]> = pops.iter().map(Vec::as_slice).collect();
        let opt = |b: &Option<PyBuffer<f32>>| b.as_ref().map_or(Ok(Vec::new()), |b| b.to_vec(py));
        self.tap.step(t, &pops, &opt(&scalars)?, &opt(&probe)?);
        Ok(())
    }

    /// An image the view draws, e.g. a prediction; `data` is 2-D (h, w).
    /// `scale` defaults to the largest magnitude.
    #[pyo3(signature = (id, data, signed = false, scale = None))]
    fn map(
        &mut self,
        py: Python<'_>,
        id: u8,
        data: PyBuffer<f32>,
        signed: bool,
        scale: Option<f32>,
    ) -> PyResult<()> {
        if !self.tap.streaming() {
            return Ok(());
        }
        let &[h, w] = data.shape() else {
            return Err(PyValueError::new_err("map data must be 2-D"));
        };
        let v = data.to_vec(py)?;
        let scale = scale.unwrap_or_else(|| v.iter().fold(0.0f32, |m, x| m.max(x.abs())));
        self.tap
            .extra(|b| frame::map(b, id, w, h, signed, scale, &v));
        Ok(())
    }

    /// Weight images of `units` of population `pop`; `data` is (n, h, w) or
    /// (n, channels, h, w). Labels of -1 mean none.
    #[pyo3(signature = (title, pop, data, units, labels = None, values = None))]
    fn fields(
        &mut self,
        py: Python<'_>,
        title: &str,
        pop: usize,
        data: PyBuffer<f32>,
        units: PyBuffer<u16>,
        labels: Option<Vec<i32>>,
        values: Option<PyBuffer<f32>>,
    ) -> PyResult<()> {
        let (n, channels, h, w) = match *data.shape() {
            [n, h, w] => (n, 1, h, w),
            [n, c, h, w] => (n, c, h, w),
            _ => return Err(PyValueError::new_err("fields data must be 3-D or 4-D")),
        };
        let units = units.to_vec(py)?;
        let labels: Vec<Option<u8>> = match labels {
            Some(l) => l.iter().map(|&x| u8::try_from(x).ok()).collect(),
            None => vec![None; n],
        };
        let values = match values {
            Some(v) => v.to_vec(py)?,
            None => vec![0.0; n],
        };
        if units.len() != n || labels.len() != n || values.len() != n {
            return Err(PyValueError::new_err(
                "units, labels and values need one entry per field",
            ));
        }
        if !self.fresh {
            self.fields.clear();
            self.fresh = true;
        }
        let f = frame::Fields {
            title,
            pop,
            w,
            h,
            channels,
            units: &units,
            labels: &labels,
            values: &values,
            data: &data.to_vec(py)?,
        };
        let mut b = Buf::new();
        frame::fields(&mut b, &f);
        self.fields.extend(b.take());
        Ok(())
    }

    /// Ends a presentation. `predicted` is None for no decision; `spikes` counts
    /// spikes per population; `scalars` are per-image metrics, smoothed into
    /// history.
    #[pyo3(signature = (info, predicted, correct, running_accuracy, scores, spikes, scalars = None))]
    fn end_image(
        &mut self,
        py: Python<'_>,
        info: Info,
        predicted: Option<u8>,
        correct: bool,
        running_accuracy: f32,
        scores: PyBuffer<f32>,
        spikes: Vec<u32>,
        scalars: Option<PyBuffer<f32>>,
    ) -> PyResult<()> {
        let scalars = scalars.as_ref().map_or(Ok(Vec::new()), |b| b.to_vec(py))?;
        let e = End {
            predicted,
            correct,
            running_accuracy,
            scores: &scores.to_vec(py)?,
            spikes: &spikes,
            scalars: &scalars,
        };
        let fields = &self.fields;
        self.tap
            .end_image(&image_info(info), &e, |_| {}, |b| b.raw(fields));
        self.fresh = false;
        Ok(())
    }

    #[pyo3(signature = (epoch, train_accuracy, test_accuracy = None, images_per_sec = 0.0))]
    fn epoch_end(
        &mut self,
        epoch: usize,
        train_accuracy: f32,
        test_accuracy: Option<f32>,
        images_per_sec: f32,
    ) {
        self.tap.epoch_end(&EpochResult {
            epoch,
            train_accuracy,
            test_accuracy,
            images_per_sec,
        });
    }

    /// Marks training done and keeps serving until Ctrl-C.
    fn hold(&self, py: Python<'_>) -> PyResult<()> {
        app::finish_live(&self.hub, &self.ctl);
        loop {
            py.allow_threads(|| std::thread::sleep(Duration::from_millis(100)));
            py.check_signals()?;
        }
    }
}

#[pymodule]
#[pyo3(name = "snn_live")]
fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Live>()
}
