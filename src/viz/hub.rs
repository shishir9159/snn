//! Broadcast of record packets from the trainer to any number of viewers.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;
use tungstenite::Bytes;

/// Packets kept for viewers that fall behind; older ones are skipped.
const RING: usize = 512;

#[derive(Default)]
struct Shared {
    /// sequence number of the newest packet
    seq: u64,
    ring: VecDeque<Bytes>,
    meta: Bytes,
    state: Bytes,
}

#[derive(Default)]
pub struct Hub {
    shared: Mutex<Shared>,
    wake: Condvar,
    viewers: AtomicUsize,
    snapshot_wanted: AtomicBool,
}

impl Hub {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn viewers(&self) -> usize {
        self.viewers.load(Ordering::Relaxed)
    }

    pub fn publish(&self, packet: Vec<u8>) {
        let mut s = self.shared.lock().unwrap();
        s.seq += 1;
        s.ring.push_back(packet.into());
        if s.ring.len() > RING {
            s.ring.pop_front();
        }
        self.wake.notify_all();
    }

    /// Sets the description every new viewer receives first.
    pub fn set_meta(&self, packet: Vec<u8>) {
        self.shared.lock().unwrap().meta = packet.into();
    }

    /// Replaces the control state and broadcasts it.
    pub fn set_state(&self, packet: Vec<u8>) {
        self.shared.lock().unwrap().state = packet.clone().into();
        self.publish(packet);
    }

    /// True once after a viewer joined: the trainer should resend weights and history.
    pub fn take_snapshot_request(&self) -> bool {
        self.snapshot_wanted.swap(false, Ordering::Relaxed)
    }

    pub fn join(&self) -> Viewer<'_> {
        self.viewers.fetch_add(1, Ordering::Relaxed);
        self.snapshot_wanted.store(true, Ordering::Relaxed);
        let s = self.shared.lock().unwrap();
        Viewer {
            hub: self,
            next: s.seq + 1,
        }
    }
}

/// One viewer's position in the packet stream.
pub struct Viewer<'a> {
    hub: &'a Hub,
    next: u64,
}

impl Viewer<'_> {
    /// Packets that bring a new viewer up to date.
    pub fn intro(&self) -> Vec<Bytes> {
        let s = self.hub.shared.lock().unwrap();
        [&s.meta, &s.state]
            .into_iter()
            .filter(|p| !p.is_empty())
            .cloned()
            .collect()
    }

    /// Packets published since the last call, waiting up to `timeout` for one.
    pub fn poll(&mut self, timeout: Duration) -> Vec<Bytes> {
        let mut s = self.hub.shared.lock().unwrap();
        if s.seq < self.next {
            s = self.hub.wake.wait_timeout(s, timeout).unwrap().0;
        }
        let oldest = s.seq + 1 - s.ring.len() as u64;
        let from = self.next.max(oldest);
        self.next = s.seq + 1;
        s.ring.range((from - oldest) as usize..).cloned().collect()
    }
}

impl Drop for Viewer<'_> {
    fn drop(&mut self) {
        self.hub.viewers.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn viewer_sees_packets_after_joining() {
        let hub = Hub::new();
        hub.publish(vec![1]);
        let mut v = hub.join();
        assert_eq!(hub.viewers(), 1);
        assert!(hub.take_snapshot_request());
        assert!(!hub.take_snapshot_request());
        hub.publish(vec![2]);
        hub.publish(vec![3]);
        let got: Vec<Vec<u8>> = v.poll(Duration::ZERO).iter().map(|b| b.to_vec()).collect();
        assert_eq!(got, vec![vec![2], vec![3]]);
        assert!(v.poll(Duration::from_millis(1)).is_empty());
        drop(v);
        assert_eq!(hub.viewers(), 0);
    }

    #[test]
    fn slow_viewer_skips_to_the_ring() {
        let hub = Hub::new();
        let mut v = hub.join();
        for i in 0..RING + 10 {
            hub.publish(vec![(i % 256) as u8]);
        }
        let got = v.poll(Duration::ZERO);
        assert_eq!(got.len(), RING);
        assert_eq!(got[0][0], 10);
    }

    #[test]
    fn intro_has_meta_then_state() {
        let hub = Hub::new();
        hub.set_meta(vec![9]);
        hub.set_state(vec![7]);
        let v = hub.join();
        let intro: Vec<Vec<u8>> = v.intro().iter().map(|b| b.to_vec()).collect();
        assert_eq!(intro, vec![vec![9], vec![7]]);
    }

    #[test]
    fn poll_wakes_on_publish() {
        let hub = Arc::new(Hub::new());
        let h = hub.clone();
        let t = std::thread::spawn(move || {
            let mut v = h.join();
            v.poll(Duration::from_secs(5)).len()
        });
        while hub.viewers() == 0 {
            std::thread::yield_now();
        }
        hub.publish(vec![1]);
        assert_eq!(t.join().unwrap(), 1);
    }
}
