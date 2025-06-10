//! Tap for population models: streams spikes of every population, one probed
//! neuron, per-image outcomes and weight fields with the generic records the
//! dashboard draws for any such model.

use crate::frame::{self, Buf};
use crate::stream::Series;
use crate::{Hub, Stream};
use snn_core::{Control, EpochResult, ImageInfo};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Weight fields go out at most this often.
const FIELDS_GAP: Duration = Duration::from_millis(250);
/// Smoothing of per-image scalars, in images.
const SMOOTH: f32 = 200.0;

pub struct PopTap {
    pub s: Stream,
    ctl: Arc<Control>,
    last_fields: Instant,
    /// latest FIELDS records, replayed to joining viewers
    fields_rec: Vec<u8>,
    smoothed: Vec<f32>,
}

/// What a model reports when a presentation ends.
pub struct End<'a> {
    pub predicted: Option<u8>,
    pub correct: bool,
    pub running_accuracy: f32,
    pub scores: &'a [f32],
    pub spikes: &'a [u32],
    pub scalars: &'a [f32],
}

impl PopTap {
    pub fn new(hub: Arc<Hub>, ctl: Arc<Control>) -> Self {
        Self {
            s: Stream::new(hub),
            ctl,
            last_fields: Instant::now() - Duration::from_secs(1),
            fields_rec: Vec::new(),
            smoothed: Vec::new(),
        }
    }

    /// The (population, unit) a viewer asked to probe, if any.
    pub fn probe_request(&self) -> Option<(usize, usize)> {
        self.ctl.probe()
    }

    /// Starts a presentation; when it is streamed, announces the probed neuron.
    pub fn begin_image(
        &mut self,
        info: &ImageInfo,
        pixels: &[u8],
        probe: (usize, usize, f32),
    ) -> bool {
        let streaming = self.s.begin_image(info, pixels);
        if streaming {
            frame::probe(&mut self.s.buf, probe.0, probe.1, probe.2);
        }
        streaming
    }

    pub fn streaming(&self) -> bool {
        self.s.streaming()
    }

    /// One simulated step; only streamed presentations are encoded.
    pub fn step(&mut self, t: usize, spikes: &[&[u16]], scalars: &[f32], probe: &[f32]) {
        self.s.sim_steps += 1;
        if !self.s.streaming() {
            return;
        }
        let b = self.s.step(frame::SPIKES, t);
        frame::spikes_step(b, spikes, scalars, probe);
        if self.s.flush_due() {
            self.s.flush();
        }
    }

    /// Adds records to a streamed presentation between steps, e.g. a map every
    /// few steps.
    pub fn extra<F: FnOnce(&mut Buf)>(&mut self, f: F) {
        if self.s.streaming() {
            self.s.close_steps();
            f(&mut self.s.buf);
        }
    }

    /// Ends a presentation. `streamed` adds records (e.g. maps) to a streamed one;
    /// `fields` writes FIELDS records when they are due.
    pub fn end_image<S, F>(&mut self, info: &ImageInfo, e: &End, streamed: S, fields: F)
    where
        S: FnOnce(&mut Buf),
        F: FnOnce(&mut Buf),
    {
        self.s.end_image(info, e.running_accuracy);
        if info.learn {
            self.smooth(e.scalars);
        }
        let viewers = self.s.viewers();
        let resume_due = self.s.resume_due();
        let send_fields = viewers && self.last_fields.elapsed() >= FIELDS_GAP;
        if send_fields || resume_due {
            let mut b = Buf::new();
            fields(&mut b);
            self.fields_rec = b.take();
        }
        if resume_due {
            self.s.refresh_resume(&self.fields_rec);
        }
        if !viewers {
            return;
        }
        let streaming = self.s.streaming();
        if streaming {
            self.s.close_steps();
            streamed(&mut self.s.buf);
            let o = frame::Outcome {
                seq: self.s.seq(),
                predicted: e.predicted,
                correct: e.correct,
                learn: info.learn,
                running_accuracy: e.running_accuracy,
                scores: e.scores,
                spikes: e.spikes,
                scalars: e.scalars,
                smoothed: &self.smoothed,
            };
            frame::outcome(&mut self.s.buf, &o);
        }
        if send_fields {
            self.s.close_steps();
            self.s.buf.raw(&self.fields_rec);
            self.last_fields = Instant::now();
        }
        if streaming || send_fields || self.s.flush_due() {
            self.s.flush();
        }
    }

    // running means of the per-image scalars; history series 1.. follow them
    fn smooth(&mut self, scalars: &[f32]) {
        if self.smoothed.len() != scalars.len() {
            self.smoothed = scalars.to_vec();
            self.s.history.resize(1 + scalars.len(), Series::default());
        }
        let k = 1.0 / SMOOTH.min(self.s.images as f32).max(1.0);
        for (i, (m, &v)) in self.smoothed.iter_mut().zip(scalars).enumerate() {
            *m += k * (v - *m);
            self.s.history[1 + i].record(self.s.images, *m);
        }
    }

    pub fn epoch_end(&mut self, e: &EpochResult) {
        self.s.epoch_end(e, &self.fields_rec);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::records;

    fn info(index: usize) -> ImageInfo {
        ImageInfo {
            epoch: 0,
            index,
            total: 3,
            label: 1,
            learn: true,
        }
    }

    fn end(scalar: f32) -> End<'static> {
        End {
            predicted: Some(1),
            correct: true,
            running_accuracy: 1.0,
            scores: &[0.0, 1.0],
            spikes: &[4, 2],
            scalars: if scalar > 1.0 { &[2.0] } else { &[0.0] },
        }
    }

    #[test]
    fn streams_steps_outcome_and_fields() {
        let hub = Arc::new(Hub::new());
        let mut v = hub.join();
        let mut tap = PopTap::new(hub.clone(), Arc::new(Control::new(0)));
        assert!(tap.begin_image(&info(0), &[0; 4], (1, 0, 0.4)));
        tap.step(1, &[&[2], &[]], &[0.5], &[0.1, 0.2, 0.3]);
        tap.step(2, &[&[], &[0]], &[0.25], &[0.1, 0.2, 0.3]);
        let fields = |b: &mut Buf| frame::map(b, 9, 1, 1, false, 1.0, &[1.0]);
        let streamed = |b: &mut Buf| frame::map(b, 0, 2, 2, false, 1.0, &[0.0; 4]);
        tap.end_image(&info(0), &end(2.0), streamed, fields);
        let kinds: Vec<u8> = v
            .poll(Duration::ZERO)
            .iter()
            .flat_map(|p| records(p).iter().map(|r| r.0).collect::<Vec<_>>())
            .collect();
        // the first step is flushed on its own, so its record is split
        let want = [
            frame::IMAGE,
            frame::PROBE,
            frame::SPIKES,
            frame::SPIKES,
            frame::MAP,
            frame::OUTCOME,
            frame::MAP,
            frame::PROGRESS,
        ];
        assert_eq!(kinds, want);
        assert_eq!(tap.s.sim_steps, 2);
    }

    #[test]
    fn scalars_are_smoothed_into_history() {
        let hub = Arc::new(Hub::new());
        let mut tap = PopTap::new(hub, Arc::new(Control::new(0)));
        for i in 0..3 {
            tap.begin_image(&info(i), &[0; 4], (0, 0, 1.0));
            let e = end(if i == 0 { 2.0 } else { 0.0 });
            tap.end_image(&info(i), &e, |_| {}, |_| {});
        }
        assert_eq!(tap.s.history.len(), 2);
        let pts = &tap.s.history[1].points;
        assert_eq!(pts.len(), 3);
        // first image sets the mean, the next two pull it halfway, then a third
        assert_eq!(pts[0].1, 2.0);
        assert!((pts[2].1 - 2.0 / 3.0).abs() < 1e-6);
    }
}
