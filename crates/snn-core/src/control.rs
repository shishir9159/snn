//! Run control shared with other threads (pause, speed, stop) and step pacing.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// Pause, speed and stop requests shared with other threads.
#[derive(Debug)]
pub struct Control {
    paused: AtomicBool,
    lock: Mutex<()>,
    wake: Condvar,
    /// simulation steps per second, 0 for unthrottled
    speed: AtomicU32,
    stop: AtomicBool,
    done: AtomicBool,
}

impl Control {
    pub fn new(speed: u32) -> Self {
        Self {
            paused: AtomicBool::new(false),
            lock: Mutex::new(()),
            wake: Condvar::new(),
            speed: AtomicU32::new(speed),
            stop: AtomicBool::new(false),
            done: AtomicBool::new(false),
        }
    }

    pub fn set_paused(&self, paused: bool) {
        let _g = self.lock.lock().unwrap();
        self.paused.store(paused, Ordering::Relaxed);
        self.wake.notify_all();
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    pub fn set_speed(&self, steps_per_sec: u32) {
        self.speed.store(steps_per_sec, Ordering::Relaxed);
    }

    pub fn speed(&self) -> u32 {
        self.speed.load(Ordering::Relaxed)
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
        self.set_paused(false);
    }

    pub fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Marks training as finished.
    pub fn finish(&self) {
        self.done.store(true, Ordering::Relaxed);
    }

    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Relaxed)
    }

    /// Blocks while paused; returns how long it waited.
    pub fn wait_while_paused(&self) -> Duration {
        if !self.is_paused() {
            return Duration::ZERO;
        }
        let t0 = Instant::now();
        let mut g = self.lock.lock().unwrap();
        while self.is_paused() && !self.stopped() {
            g = self.wake.wait(g).unwrap();
        }
        t0.elapsed()
    }
}

/// Holds a simulation to the control's step rate.
pub struct Pacer {
    origin: Instant,
    steps: u64,
    rate: u32,
}

impl Pacer {
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
            steps: 0,
            rate: 0,
        }
    }

    pub fn tick(&mut self, ctl: &Control) {
        if ctl.wait_while_paused() > Duration::ZERO || ctl.speed() != self.rate {
            *self = Self {
                rate: ctl.speed(),
                ..Self::new()
            };
        }
        if self.rate == 0 {
            return;
        }
        self.steps += 1;
        let due = self.origin + Duration::from_secs_f64(self.steps as f64 / self.rate as f64);
        let now = Instant::now();
        // sleep in chunks of at least 2 ms; coarse OS timers would overshoot finer waits
        if due > now + Duration::from_millis(2) {
            std::thread::sleep(due - now);
        } else if now > due + Duration::from_millis(250) {
            // fell far behind (e.g. a slow frame); don't sprint to catch up
            *self = Self {
                rate: self.rate,
                ..Self::new()
            };
        }
    }
}

impl Default for Pacer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pacing_holds_the_step_rate() {
        let ctl = Control::new(2000);
        let mut p = Pacer::new();
        let t0 = Instant::now();
        for _ in 0..200 {
            p.tick(&ctl);
        }
        let dt = t0.elapsed().as_secs_f32();
        assert!((0.08..0.3).contains(&dt), "200 steps at 2000/s took {dt}s");
    }

    #[test]
    fn pause_blocks_until_resumed() {
        let ctl = std::sync::Arc::new(Control::new(0));
        ctl.set_paused(true);
        let c = ctl.clone();
        let h = std::thread::spawn(move || c.wait_while_paused());
        std::thread::sleep(Duration::from_millis(30));
        ctl.set_paused(false);
        assert!(h.join().unwrap() >= Duration::from_millis(20));
    }
}
