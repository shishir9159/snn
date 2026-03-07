//! The model-independent half of a trainer tap: decides which presentations
//! are streamed, batches records into packets, tracks progress and history,
//! and keeps the snapshot a joining viewer starts from.

use crate::frame::{self, Buf};
use crate::hub::Hub;
use snn_core::{Activity, Confusion, EpochResult, ImageInfo, PerImage};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Packets go out at most this often.
const FLUSH: Duration = Duration::from_millis(16);
/// Fastest rate at which whole presentations are streamed step by step.
const STREAM_GAP: Duration = Duration::from_millis(33);
/// How often the snapshot for joining viewers is refreshed.
const RESUME_GAP: Duration = Duration::from_millis(250);
/// History points kept before halving the resolution.
pub const HISTORY: usize = 2048;
/// Images the running means of per-image figures span.
pub const SMOOTH: f32 = 200.0;

/// One history series, thinned by half whenever it fills up.
#[derive(Debug, Clone, Default)]
pub struct Series {
    pub points: Vec<(u32, f32)>,
    every: u64,
}

impl Series {
    pub fn record(&mut self, images: u64, v: f32) {
        let every = self.every.max(1);
        if images.is_multiple_of(every) {
            self.points.push((images as u32, v));
            if self.points.len() >= HISTORY {
                self.points = self.points.iter().copied().step_by(2).collect();
                self.every = every * 2;
            }
        }
    }
}

pub struct Stream {
    hub: Arc<Hub>,
    pub buf: Buf,
    streaming: bool,
    /// offset of the open step record's count, and the count
    steps_open: Option<(usize, u16)>,
    last_flush: Instant,
    last_stream: Instant,
    last_resume: Instant,
    seq: u32,
    pub sim_steps: u64,
    pub images: u64,
    progress: Option<(ImageInfo, f32)>,
    /// running mean of the training images' activity
    activity: PerImage,
    rate: (Instant, u64, f32),
    /// series 0 is running accuracy; models may add more
    pub history: Vec<Series>,
    /// finished epochs with the images seen by then, replayed to late viewers
    epochs: Vec<(EpochResult, u64)>,
    /// the latest test pass, replayed to late viewers
    confusion: Option<Confusion>,
}

impl Stream {
    pub fn new(hub: Arc<Hub>) -> Self {
        let long_ago = Instant::now() - Duration::from_secs(1);
        Self {
            hub,
            buf: Buf::new(),
            streaming: false,
            steps_open: None,
            last_flush: long_ago,
            last_stream: long_ago,
            last_resume: long_ago,
            seq: 0,
            sim_steps: 0,
            images: 0,
            progress: None,
            activity: PerImage::default(),
            rate: (Instant::now(), 0, 0.0),
            history: vec![Series::default()],
            epochs: Vec::new(),
            confusion: None,
        }
    }

    pub fn viewers(&self) -> bool {
        self.hub.viewers() > 0
    }

    /// Whether the current presentation is streamed step by step.
    pub fn streaming(&self) -> bool {
        self.streaming
    }

    pub fn seq(&self) -> u32 {
        self.seq
    }

    /// Starts a presentation; streams it if a viewer is watching and the last
    /// streamed one was long enough ago.
    pub fn begin_image(&mut self, info: &ImageInfo, pixels: &[u8]) -> bool {
        self.seq += 1;
        let now = Instant::now();
        self.streaming = self.viewers() && now >= self.last_stream + STREAM_GAP;
        if self.streaming {
            self.last_stream = now;
            self.close_steps();
            frame::image(&mut self.buf, self.seq, info, pixels);
        }
        self.streaming
    }

    /// Buffer for one more step in a record of `kind`, opened at step `t` if
    /// none is open.
    pub fn step(&mut self, kind: u8, t: usize) -> &mut Buf {
        let (at, n) = match self.steps_open {
            Some(open) => open,
            None => (frame::steps_begin(&mut self.buf, kind, t), 0),
        };
        self.steps_open = Some((at, n + 1));
        &mut self.buf
    }

    pub fn close_steps(&mut self) {
        if let Some((at, n)) = self.steps_open.take() {
            self.buf.patch_u16(at, n);
            self.buf.end();
        }
    }

    pub fn flush_due(&self) -> bool {
        self.last_flush.elapsed() >= FLUSH
    }

    pub fn resume_due(&self) -> bool {
        self.last_resume.elapsed() >= RESUME_GAP
    }

    /// Publishes what is buffered, with a fresh progress record.
    pub fn flush(&mut self) {
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
        if !packet.is_empty() && self.viewers() {
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
            activity: self.activity,
        })
    }

    /// Counts a finished presentation; training images extend the accuracy
    /// history and the running mean of activity.
    pub fn end_image(&mut self, info: &ImageInfo, running_accuracy: f32, activity: Activity) {
        if info.learn {
            self.images += 1;
            self.history[0].record(self.images, running_accuracy);
            let k = 1.0 / SMOOTH.min(self.images as f32);
            let now = activity.per_image(1);
            let mean = &mut self.activity;
            mean.spikes += k * (now.spikes - mean.spikes);
            mean.synops += k * (now.synops - mean.synops);
            mean.macs += k * (now.macs - mean.macs);
        }
        self.progress = Some((*info, running_accuracy));
    }

    /// Caches what a viewer joining now needs: the model's own records, then
    /// history, epochs and progress.
    pub fn refresh_resume(&mut self, model: &[u8]) {
        let mut b = Buf::new();
        b.raw(model);
        for (i, s) in self.history.iter().enumerate() {
            frame::history(&mut b, i as u8, &s.points);
        }
        for (e, images) in &self.epochs {
            frame::epoch(&mut b, e, *images);
        }
        if let Some(c) = &self.confusion {
            frame::confusion(&mut b, c);
        }
        if let Some(p) = self.progress_record() {
            frame::progress(&mut b, &p);
        }
        self.hub.set_resume(b.take());
        self.last_resume = Instant::now();
    }

    pub fn epoch_end(&mut self, e: &EpochResult, model: &[u8]) {
        self.close_steps();
        frame::epoch(&mut self.buf, e, self.images);
        if let Some(c) = &e.confusion {
            frame::confusion(&mut self.buf, c);
            self.confusion = Some(c.clone());
        }
        let kept = EpochResult {
            confusion: None,
            ..e.clone()
        };
        self.epochs.push((kept, self.images));
        self.refresh_resume(model);
        self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(learn: bool) -> ImageInfo {
        ImageInfo {
            epoch: 0,
            index: 0,
            total: 1,
            label: 3,
            learn,
        }
    }

    #[test]
    fn history_halves_when_full() {
        let mut s = Series::default();
        for i in 0..HISTORY as u64 * 3 {
            s.record(i, 0.5);
        }
        assert!(s.points.len() < HISTORY);
        assert!(s.every >= 2);
        assert!(s.points.windows(2).all(|w| w[0].0 < w[1].0));
    }

    #[test]
    fn steps_share_one_record_until_closed() {
        let hub = Arc::new(Hub::new());
        let mut v = hub.join();
        let mut s = Stream::new(hub.clone());
        assert!(s.begin_image(&info(true), &[0; 4]));
        for t in 1..=3 {
            s.step(40, t).u8(t as u8);
        }
        s.end_image(&info(true), 0.5, Activity::new(4, 40));
        s.flush();
        let packets = v.poll(Duration::ZERO);
        let r = frame::records(&packets[0]);
        let kinds: Vec<u8> = r.iter().map(|x| x.0).collect();
        assert_eq!(kinds, [frame::IMAGE, 40, frame::PROGRESS]);
        assert_eq!(r[1].1, &[1, 0, 3, 0, 1, 2, 3]);
        assert_eq!(s.images, 1);
    }

    #[test]
    fn activity_is_a_running_mean_of_training_images() {
        let mut s = Stream::new(Arc::new(Hub::new()));
        s.end_image(&info(true), 0.0, Activity::new(10, 100));
        assert_eq!(
            s.activity,
            PerImage {
                spikes: 10.0,
                synops: 100.0,
                macs: 0.0
            }
        );
        s.end_image(&info(true), 0.0, Activity::new(20, 300).with_macs(50));
        assert_eq!(
            s.activity,
            PerImage {
                spikes: 15.0,
                synops: 200.0,
                macs: 25.0
            }
        );
        s.end_image(&info(false), 0.0, Activity::new(1000, 9000));
        assert_eq!(s.activity.spikes, 15.0, "test images are not averaged in");
        let p = s.progress_record().unwrap();
        assert_eq!(p.activity, s.activity);
    }

    #[test]
    fn late_viewers_get_the_latest_confusion_once() {
        let hub = Arc::new(Hub::new());
        let mut s = Stream::new(hub.clone());
        let epoch = |n: usize, hits: u8| {
            let mut c = Confusion::new(2);
            (0..hits).for_each(|_| c.record(1, Some(1)));
            EpochResult {
                epoch: n,
                train_accuracy: 0.5,
                test_accuracy: Some(c.accuracy()),
                images_per_sec: 1.0,
                activity: PerImage::default(),
                confusion: Some(c),
            }
        };
        s.epoch_end(&epoch(0, 3), &[]);
        s.epoch_end(&epoch(1, 5), &[]);
        let intro = hub.join().intro();
        let kinds: Vec<Vec<u8>> = intro
            .iter()
            .map(|p| frame::records(p).iter().map(|r| r.0).collect())
            .collect();
        let confusions = kinds
            .concat()
            .iter()
            .filter(|&&k| k == frame::CONFUSION)
            .count();
        assert_eq!(confusions, 1);
        let resume = &intro[intro.len() - 1];
        let (_, p) = frame::records(resume)
            .into_iter()
            .find(|r| r.0 == frame::CONFUSION)
            .unwrap();
        assert_eq!(
            u32::from_le_bytes(p[1 + 4 * 4..1 + 5 * 4].try_into().unwrap()),
            5
        );
    }

    #[test]
    fn nothing_streams_without_viewers() {
        let hub = Arc::new(Hub::new());
        let mut s = Stream::new(hub.clone());
        assert!(!s.begin_image(&info(false), &[0; 4]));
        s.end_image(&info(false), 0.0, Activity::default());
        s.flush();
        assert_eq!(s.images, 0);
        assert!(hub.join().poll(Duration::ZERO).is_empty());
    }
}
