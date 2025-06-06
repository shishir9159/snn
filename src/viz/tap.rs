//! Trainer-side encoder: turns training events into packets on the hub.

use super::frame::{self, Buf};
use super::hub::Hub;
use crate::encode::SpikeTrain;
use crate::layer::{Layer, Step};
use crate::train::{EpochResult, ImageInfo, ImageResult, Labels, Monitor};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Packets go out at most this often.
const FLUSH: Duration = Duration::from_millis(16);
/// Fastest rate at which whole presentations are streamed step by step.
const STREAM_GAP: Duration = Duration::from_millis(33);
const WEIGHTS_GAP: Duration = Duration::from_millis(100);
/// How often the snapshot for joining viewers is refreshed.
const RESUME_GAP: Duration = Duration::from_millis(250);
/// History points kept before halving the resolution.
const HISTORY: usize = 2048;

pub struct Tap {
    hub: Arc<Hub>,
    buf: Buf,
    w_range: (f32, f32),
    streaming: bool,
    /// offset of the open STEPS record's count, and the count
    steps_open: Option<(usize, u16)>,
    last_flush: Instant,
    last_stream: Instant,
    last_weights: Instant,
    last_resume: Instant,
    /// latest encoded WEIGHTS record
    weights_rec: Vec<u8>,
    seq: u32,
    sim_steps: u64,
    images: u64,
    progress: Option<(ImageInfo, f32)>,
    rate: (Instant, u64, f32),
    history: Vec<(u32, f32)>,
    history_every: u64,
    potentials: Vec<f32>,
    /// finished epochs with the images seen by then, replayed to late viewers
    epochs: Vec<(EpochResult, u64)>,
}

impl Tap {
    pub fn new(hub: Arc<Hub>, w_min: f32, w_max: f32) -> Self {
        let long_ago = Instant::now() - Duration::from_secs(1);
        Self {
            hub,
            buf: Buf::new(),
            w_range: (w_min, w_max),
            streaming: false,
            steps_open: None,
            last_flush: long_ago,
            last_stream: long_ago,
            last_weights: long_ago,
            last_resume: long_ago,
            weights_rec: Vec::new(),
            seq: 0,
            sim_steps: 0,
            images: 0,
            progress: None,
            rate: (Instant::now(), 0, 0.0),
            history: Vec::new(),
            history_every: 1,
            potentials: Vec::new(),
            epochs: Vec::new(),
        }
    }

    fn close_steps(&mut self) {
        if let Some((at, n)) = self.steps_open.take() {
            self.buf.patch_u16(at, n);
            self.buf.end();
        }
    }

    fn flush(&mut self) {
        self.close_steps();
        let now = Instant::now();
        let dt = now.duration_since(self.rate.0).as_secs_f32();
        if dt >= 0.5 {
            self.rate = (now, self.images, (self.images - self.rate.1) as f32 / dt);
        }
        if let Some(p) = self.progress_record() {
            frame::progress(&mut self.buf, &p);
        }
        let packet = self.buf.take();
        if !packet.is_empty() && self.hub.viewers() > 0 {
            self.hub.publish(packet);
        }
        self.last_flush = now;
    }

    fn progress_record(&self) -> Option<frame::Progress> {
        self.progress.map(|(info, acc)| frame::Progress {
            epoch: info.epoch,
            index: info.index,
            total: info.total,
            learn: info.learn,
            running_accuracy: acc,
            sim_steps: self.sim_steps,
            images: self.images,
            images_per_sec: self.rate.2,
        })
    }

    fn encode_weights(&mut self, layer: &Layer, labels: &Labels) {
        let assigned: Vec<Option<u8>> = (0..layer.n_out).map(|j| labels.of(j)).collect();
        let mut b = Buf::new();
        let w = frame::Weights {
            rows: layer.n_out,
            cols: layer.n_in,
            w: &layer.weights,
            lo: self.w_range.0,
            hi: self.w_range.1,
            labels: &assigned,
            theta: &layer.theta,
        };
        frame::weights(&mut b, &w);
        self.weights_rec = b.take();
    }

    /// Caches what a viewer joining now needs: weights, history, epochs and progress.
    fn refresh_resume(&mut self) {
        let mut b = Buf::new();
        b.raw(&self.weights_rec);
        frame::history(&mut b, &self.history);
        for (e, images) in &self.epochs {
            frame::epoch(&mut b, e, *images);
        }
        if let Some(p) = self.progress_record() {
            frame::progress(&mut b, &p);
        }
        self.hub.set_resume(b.take());
        self.last_resume = Instant::now();
    }

    fn record_history(&mut self, acc: f32) {
        if self.images % self.history_every == 0 {
            self.history.push((self.images as u32, acc));
            if self.history.len() >= HISTORY {
                let kept: Vec<_> = self.history.iter().copied().step_by(2).collect();
                self.history = kept;
                self.history_every *= 2;
            }
        }
    }
}

impl Monitor for Tap {
    fn image_start(&mut self, info: &ImageInfo, pixels: &[u8], _train: &SpikeTrain) {
        self.seq += 1;
        let now = Instant::now();
        self.streaming = self.hub.viewers() > 0 && now >= self.last_stream + STREAM_GAP;
        if self.streaming {
            self.last_stream = now;
            self.close_steps();
            frame::image(&mut self.buf, self.seq, info, pixels);
        }
    }

    fn step(&mut self, s: &Step) {
        self.sim_steps += 1;
        if !self.streaming {
            return;
        }
        if s.t == 1 {
            self.close_steps();
            frame::thresholds(&mut self.buf, s.thresholds);
        }
        let (at, n) = match self.steps_open {
            Some(open) => open,
            None => (frame::steps_begin(&mut self.buf, s.t), 0),
        };
        self.potentials.clear();
        self.potentials.extend(s.neurons.iter().map(|n| n.p));
        frame::step(&mut self.buf, s.active, s.fired, &self.potentials, s.winner);
        self.steps_open = Some((at, n + 1));
        if self.last_flush.elapsed() >= FLUSH {
            self.flush();
        }
    }

    fn image_end(&mut self, r: &ImageResult, layer: &Layer, labels: &Labels) {
        if r.info.learn {
            self.images += 1;
            self.record_history(r.running_accuracy);
        }
        self.progress = Some((r.info, r.running_accuracy));
        let viewers = self.hub.viewers() > 0;
        let resume_due = self.last_resume.elapsed() >= RESUME_GAP;
        let send_weights = viewers && self.last_weights.elapsed() >= WEIGHTS_GAP;
        if send_weights || resume_due {
            self.encode_weights(layer, labels);
        }
        if resume_due {
            self.refresh_resume();
        }
        if !viewers {
            return;
        }
        if self.streaming {
            let assigned: Vec<Option<u8>> = (0..layer.n_out).map(|j| labels.of(j)).collect();
            self.close_steps();
            let res = frame::Result {
                seq: self.seq,
                winner: r.outcome.winner,
                predicted: r.predicted,
                correct: r.correct,
                learn: r.info.learn,
                running_accuracy: r.running_accuracy,
                spikes: &r.outcome.spikes,
                theta: &layer.theta,
                labels: &assigned,
            };
            frame::result(&mut self.buf, &res);
        }
        if send_weights {
            self.close_steps();
            self.buf.raw(&self.weights_rec);
            self.last_weights = Instant::now();
        }
        if self.streaming || send_weights || self.last_flush.elapsed() >= FLUSH {
            self.flush();
        }
    }

    fn epoch_end(&mut self, e: &EpochResult) {
        self.close_steps();
        frame::epoch(&mut self.buf, e, self.images);
        self.epochs.push((e.clone(), self.images));
        self.refresh_resume();
        self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::synthetic;
    use crate::train::{Control, Trainer};
    use crate::Params;

    fn kinds(packet: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut at = 0;
        while at < packet.len() {
            out.push(packet[at]);
            at += 5 + u32::from_le_bytes(packet[at + 1..at + 5].try_into().unwrap()) as usize;
        }
        out
    }

    #[test]
    fn silent_without_viewers() {
        let hub = Arc::new(Hub::new());
        let mut tap = Tap::new(hub.clone(), 0.0, 2.0);
        let data = synthetic(5, 1);
        let mut t = Trainer::new(Params::default(), 784, 4, 10, 1);
        t.fit(&data, None, 1, &mut tap, &Control::new(0));
        let mut v = hub.join();
        assert!(v.poll(Duration::ZERO).is_empty());
        assert_eq!(tap.sim_steps, 5 * 200);
        assert_eq!(tap.history.len(), 5);
    }

    #[test]
    fn streams_a_full_presentation() {
        let hub = Arc::new(Hub::new());
        let mut v = hub.join();
        let mut tap = Tap::new(hub.clone(), 0.0, 2.0);
        let data = synthetic(3, 1);
        let mut t = Trainer::new(Params::default(), 784, 4, 10, 1);
        t.fit(&data, None, 1, &mut tap, &Control::new(0));

        let all: Vec<u8> = v
            .poll(Duration::ZERO)
            .iter()
            .flat_map(|p| kinds(p))
            .collect();
        let count = |kind| all.iter().filter(|&&k| k == kind).count();
        assert!(count(frame::IMAGE) >= 1);
        assert_eq!(count(frame::THRESH), count(frame::IMAGE));
        assert_eq!(count(frame::RESULT), count(frame::IMAGE));
        assert!(all.contains(&frame::WEIGHTS));
        assert!(all.contains(&frame::PROGRESS));
        assert_eq!(all.last(), Some(&frame::PROGRESS));
        assert!(all.contains(&frame::EPOCH));
    }

    #[test]
    fn late_viewer_gets_a_snapshot() {
        let hub = Arc::new(Hub::new());
        let mut tap = Tap::new(hub.clone(), 0.0, 2.0);
        let data = synthetic(5, 1);
        let mut t = Trainer::new(Params::default(), 784, 4, 10, 1);
        t.fit(&data, None, 1, &mut tap, &Control::new(0));
        let v = hub.join();
        let intro: Vec<u8> = v.intro().iter().flat_map(|p| kinds(p)).collect();
        let want = [
            frame::WEIGHTS,
            frame::HISTORY,
            frame::EPOCH,
            frame::PROGRESS,
        ];
        assert_eq!(intro, want);
    }

    #[test]
    fn history_halves_when_full() {
        let mut tap = Tap::new(Arc::new(Hub::new()), 0.0, 2.0);
        for i in 0..HISTORY as u64 * 3 {
            tap.images = i;
            tap.record_history(0.5);
        }
        assert!(tap.history.len() < HISTORY);
        assert!(tap.history_every >= 2);
        assert!(tap.history.windows(2).all(|w| w[0].0 < w[1].0));
    }
}
