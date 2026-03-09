//! N-MNIST (Orchard et al. 2015): MNIST digits recorded from a screen by an
//! event camera that makes three small saccades. Each sample is a `.bin` file of
//! 5-byte events: x, y, then one polarity bit (1 = ON, as the dataset's readme
//! has it) and a 23-bit timestamp in microseconds. The set unpacks to
//! `Train/<digit>/*.bin` and `Test/<digit>/*.bin`; `just nmnist-small` fetches
//! 1,100 real recordings in that layout, which the tests check the reader on.

use crate::dataset::Dataset;
use crate::events::{Events, Saccades};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// The camera is 34×34.
pub const SIDE: usize = 34;

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// One sample's events bucketed by millisecond, in the layout of [`Events`].
#[derive(Debug, Clone)]
struct Sample {
    ids: Vec<u16>,
    start: Vec<u32>,
}

/// Decodes a `.bin` file into event ids (`polarity * SIDE² + y * SIDE + x`,
/// polarity 0 for ON as in [`Events`]) bucketed by step: step `t` holds the
/// events from millisecond `t - 1` up to `t`.
fn bucket(bytes: &[u8]) -> io::Result<Sample> {
    if !bytes.len().is_multiple_of(5) {
        return Err(invalid(format!(
            "{} bytes is not whole 5-byte events",
            bytes.len()
        )));
    }
    let mut events: Vec<(u32, u16)> = Vec::with_capacity(bytes.len() / 5);
    for e in bytes.chunks_exact(5) {
        let (x, y) = (e[0] as usize, e[1] as usize);
        if x >= SIDE || y >= SIDE {
            return Err(invalid(format!(
                "event at ({x}, {y}) is off the {SIDE}x{SIDE} sensor"
            )));
        }
        let on = e[2] & 0x80 != 0;
        let micros = (u32::from(e[2] & 0x7f) << 16) | (u32::from(e[3]) << 8) | u32::from(e[4]);
        let id = (!on as usize) * SIDE * SIDE + y * SIDE + x;
        events.push((micros / 1000 + 1, id as u16));
    }
    // recordings are in time order, but a stable sort makes sure
    events.sort_by_key(|&(step, _)| step);
    let steps = events.last().map_or(0, |&(step, _)| step as usize);
    let mut start = vec![0u32; steps + 2];
    for &(step, _) in &events {
        start[step as usize + 1] += 1;
    }
    for t in 1..start.len() {
        start[t] += start[t - 1];
    }
    Ok(Sample {
        ids: events.into_iter().map(|(_, id)| id).collect(),
        start,
    })
}

/// Writes events in the `.bin` format, each in the middle of its millisecond.
pub fn encode(events: &Events) -> Vec<u8> {
    assert_eq!(
        events.side, SIDE,
        "the format is for the {SIDE}x{SIDE} sensor"
    );
    let mut bytes = Vec::with_capacity(events.total() * 5);
    for t in 1..=events.steps() {
        let micros = (t as u32 - 1) * 1000 + 500;
        for &id in events.at(t) {
            let (on, pix) = ((id as usize) < SIDE * SIDE, id as usize % (SIDE * SIDE));
            let high = ((micros >> 16) & 0x7f) as u8 | if on { 0x80 } else { 0 };
            bytes.extend([(pix % SIDE) as u8, (pix / SIDE) as u8, high]);
            bytes.extend([(micros >> 8) as u8, micros as u8]);
        }
    }
    bytes
}

/// Turns `data` into recordings by simulated saccades and writes them as
/// `root/<set>/<digit>/<n>.bin`, the layout [`Recordings::load`] reads.
pub fn export(data: &Dataset, saccades: &Saccades, root: &Path, set: &str) -> io::Result<()> {
    let mut events = Events::new();
    let mut per_digit = vec![0usize; 256];
    for i in 0..data.len() {
        events.generate(data.image(i), data.width, data.height, saccades);
        let digit = data.label(i);
        let dir = root.join(set).join(digit.to_string());
        fs::create_dir_all(&dir)?;
        per_digit[digit as usize] += 1;
        let name = format!("{:05}.bin", per_digit[digit as usize]);
        fs::write(dir.join(name), encode(&events))?;
    }
    Ok(())
}

/// A set of labelled recordings, ready to replay.
#[derive(Debug, Clone)]
pub struct Recordings {
    pub name: String,
    pub classes: usize,
    steps: usize,
    labels: Vec<u8>,
    samples: Vec<Sample>,
}

impl Recordings {
    /// Loads `dir/Train` or `dir/Test`. With a `limit`, the files are taken
    /// round-robin over the digits, so a small set is still balanced.
    pub fn load(dir: &Path, train: bool, limit: Option<usize>) -> io::Result<Self> {
        let set = if train { "Train" } else { "Test" };
        let root = [set.to_string(), set.to_lowercase()]
            .iter()
            .map(|n| dir.join(n))
            .find(|p| p.is_dir())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("{set} folder not found in {}", dir.display()),
                )
            })?;
        let mut digits: Vec<(u8, PathBuf)> = fs::read_dir(&root)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .filter_map(|p| Some((p.file_name()?.to_str()?.parse().ok()?, p)))
            .collect();
        digits.sort();
        if digits.is_empty() {
            return Err(invalid(format!(
                "{}: no class folders named 0, 1, ...",
                root.display()
            )));
        }
        let mut files: Vec<Vec<PathBuf>> = Vec::new();
        for (_, dir) in &digits {
            let mut list: Vec<PathBuf> = fs::read_dir(dir)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "bin"))
                .collect();
            list.sort();
            files.push(list);
        }

        let mut rec = Recordings {
            name: format!("n-mnist/{}", set.to_lowercase()),
            classes: digits.iter().map(|d| d.0 as usize + 1).max().unwrap_or(0),
            steps: 0,
            labels: Vec::new(),
            samples: Vec::new(),
        };
        let limit = limit.unwrap_or(usize::MAX);
        'all: for round in 0.. {
            let mut any = false;
            for ((label, _), list) in digits.iter().zip(&files) {
                let Some(path) = list.get(round) else {
                    continue;
                };
                if rec.len() >= limit {
                    break 'all;
                }
                any = true;
                let bytes = fs::read(path)?;
                let sample =
                    bucket(&bytes).map_err(|e| invalid(format!("{}: {e}", path.display())))?;
                rec.steps = rec.steps.max(sample.start.len() - 2);
                rec.labels.push(*label);
                rec.samples.push(sample);
            }
            if !any {
                break;
            }
        }
        Ok(rec)
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn label(&self, i: usize) -> u8 {
        self.labels[i]
    }

    /// Steps (ms) of the longest recording.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Keeps at most the first `n` recordings.
    pub fn truncate(&mut self, n: usize) {
        self.labels.truncate(n);
        self.samples.truncate(n);
    }

    /// Puts recording `i` into `events`.
    pub fn fill(&self, i: usize, events: &mut Events) {
        let s = &self.samples[i];
        events.replace(SIDE, &s.start, &s.ids);
    }

    /// Events per pixel of recording `i`, both polarities, scaled to bytes.
    pub fn preview(&self, i: usize, out: &mut Vec<u8>) {
        out.clear();
        out.resize(SIDE * SIDE, 0);
        let mut count = vec![0u16; SIDE * SIDE];
        for &id in &self.samples[i].ids {
            count[id as usize % (SIDE * SIDE)] += 1;
        }
        let top = count.iter().copied().max().unwrap_or(0).max(1) as f32;
        for (o, &c) in out.iter_mut().zip(&count) {
            *o = (c as f32 / top * 255.0).round() as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::Saccades;

    /// One event in the file format, written by hand to check [`encode`] against.
    fn pack(x: u8, y: u8, on: bool, micros: u32) -> [u8; 5] {
        let high = ((micros >> 16) & 0x7f) as u8 | if on { 0x80 } else { 0 };
        [x, y, high, (micros >> 8) as u8, micros as u8]
    }

    fn file(events: &[(u8, u8, bool, u32)]) -> Vec<u8> {
        events
            .iter()
            .flat_map(|&(x, y, on, t)| pack(x, y, on, t))
            .collect()
    }

    #[test]
    fn decodes_position_polarity_and_time() {
        let s = bucket(&file(&[(5, 7, true, 1_234_567), (33, 0, false, 999)])).unwrap();
        // 999 us is in the first millisecond, 1_234_567 us in the 1235th
        assert_eq!(s.start.len(), 1235 + 2);
        assert_eq!(&s.ids, &[(34 * 34 + 33) as u16, (7 * 34 + 5) as u16]);
        assert_eq!(s.start[1], 0);
        assert_eq!(s.start[2], 1, "the first event is in step 1");
        assert_eq!(*s.start.last().unwrap(), 2);
    }

    #[test]
    fn rejects_damaged_files() {
        assert!(bucket(&[1, 2, 3])
            .unwrap_err()
            .to_string()
            .contains("5-byte"));
        assert!(bucket(&file(&[(34, 0, true, 0)]))
            .unwrap_err()
            .to_string()
            .contains("off the"));
        assert_eq!(bucket(&[]).unwrap().ids.len(), 0);
    }

    #[test]
    fn replays_what_the_saccade_generator_made() {
        let mut img = vec![0u8; 28 * 28];
        for y in 6..22 {
            for x in 12..16 {
                img[y * 28 + x] = 255;
            }
        }
        let mut made = Events::new();
        made.generate(&img, 28, 28, &Saccades::default());
        let bytes = encode(&made);
        assert_eq!(bytes.len(), 5 * made.total());
        assert_eq!(bytes[..5], pack_first(&made));
        let s = bucket(&bytes).unwrap();
        let mut replay = Events::new();
        replay.replace(SIDE, &s.start, &s.ids);
        assert!(made.total() > 100);
        assert_eq!(replay.total(), made.total());
        for t in 1..=made.steps().min(replay.steps()) {
            assert_eq!(replay.at(t), made.at(t), "step {t}");
        }
    }

    /// The first event as `pack` writes it.
    fn pack_first(e: &Events) -> [u8; 5] {
        let t = (1..=e.steps()).find(|&t| !e.at(t).is_empty()).unwrap();
        let id = e.at(t)[0] as usize;
        let (on, pix) = (id < SIDE * SIDE, id % (SIDE * SIDE));
        let micros = (t as u32 - 1) * 1000 + 500;
        pack((pix % SIDE) as u8, (pix / SIDE) as u8, on, micros)
    }

    fn tree(name: &str, per_digit: &[usize]) -> PathBuf {
        let root = std::env::temp_dir().join(format!("snn-nmnist-{name}-{}", std::process::id()));
        fs::remove_dir_all(&root).ok();
        for (d, &n) in per_digit.iter().enumerate() {
            let dir = root.join("Train").join(d.to_string());
            fs::create_dir_all(&dir).unwrap();
            for k in 0..n {
                // the sample's event count says which file it is
                let events: Vec<_> = (0..=k + 10 * d)
                    .map(|i| (i as u8 % 34, 3, true, i as u32 * 1000))
                    .collect();
                fs::write(dir.join(format!("{:05}.bin", k + 1)), file(&events)).unwrap();
            }
        }
        root
    }

    #[test]
    fn loads_class_folders_round_robin() {
        let root = tree("rr", &[3, 3, 1]);
        let all = Recordings::load(&root, true, None).unwrap();
        assert_eq!(
            (all.len(), all.classes, all.name.as_str()),
            (7, 3, "n-mnist/train")
        );
        let few = Recordings::load(&root, true, Some(4)).unwrap();
        assert_eq!(
            (0..4).map(|i| few.label(i)).collect::<Vec<_>>(),
            [0, 1, 2, 0]
        );
        let mut ev = Events::new();
        few.fill(1, &mut ev);
        assert_eq!(ev.total(), 11, "digit 1's first file has 11 events");
        assert_eq!(ev.side, SIDE);
        assert_eq!(few.steps(), 21);
        let mut pic = Vec::new();
        few.preview(0, &mut pic);
        assert_eq!((pic.len(), pic[3 * SIDE]), (SIDE * SIDE, 255));
        assert!(Recordings::load(&root, false, None)
            .unwrap_err()
            .to_string()
            .contains("Test folder"));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn exported_images_load_back_as_the_same_events() {
        let data = crate::dataset::synthetic(12, 3);
        let root = std::env::temp_dir().join(format!("snn-nmnist-export-{}", std::process::id()));
        fs::remove_dir_all(&root).ok();
        let s = Saccades::default();
        export(&data, &s, &root, "Train").unwrap();
        let rec = Recordings::load(&root, true, None).unwrap();
        assert_eq!((rec.len(), rec.steps() <= s.steps()), (12, true));
        let (mut want, mut got) = (Events::new(), Events::new());
        // the files come back digit by digit; find each image by its label and order
        let mut seen = [0usize; 10];
        let mut by_digit: Vec<Vec<usize>> = vec![Vec::new(); 10];
        for i in 0..data.len() {
            by_digit[data.label(i) as usize].push(i);
        }
        for r in 0..rec.len() {
            let d = rec.label(r) as usize;
            let i = by_digit[d][seen[d]];
            seen[d] += 1;
            want.generate(data.image(i), data.width, data.height, &s);
            rec.fill(r, &mut got);
            assert_eq!(got.total(), want.total());
            for t in 1..=got.steps() {
                assert_eq!(got.at(t), want.at(t), "digit {d} step {t}");
            }
        }
        fs::remove_dir_all(&root).ok();
    }

    fn correlation(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len() as f32;
        let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
        let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
        for (x, y) in a.iter().zip(b) {
            ab += (x - ma) * (y - mb);
            aa += (x - ma) * (x - ma);
            bb += (y - mb) * (y - mb);
        }
        ab / (aa * bb).sqrt().max(1e-9)
    }

    /// Events per pixel for each 100 ms leg and polarity.
    fn leg_maps(e: &Events) -> Vec<Vec<f32>> {
        let mut maps = vec![vec![0.0; SIDE * SIDE]; 6];
        for t in 1..=e.steps() {
            let leg = ((t - 1) / 100).min(2);
            for &id in e.at(t) {
                let (pol, pix) = (id as usize / (SIDE * SIDE), id as usize % (SIDE * SIDE));
                maps[leg * 2 + pol][pix] += 1.0;
            }
        }
        maps
    }

    /// Real recordings, when `just nmnist-small` and `just mnist` fetched them:
    /// they decode on the 34x34 sensor, their events sit on their MNIST digit,
    /// and the saccade simulator makes events like theirs from the same digit.
    #[test]
    fn real_recordings_match_their_digits_and_the_simulator() {
        let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
        let train = data.join("nmnist-small/Train");
        if !train.is_dir() || !data.join("mnist/train-images-idx3-ubyte.gz").is_file() {
            eprintln!("skipped: needs `just nmnist-small` and `just mnist`");
            return;
        }
        let mnist = crate::dataset::mnist(&data.join("mnist"), true).unwrap();
        let s = Saccades::default();
        let (mut placed, mut alike, mut sim) = (Vec::new(), Vec::new(), Events::new());
        for digit in 0..10u8 {
            let mut files: Vec<PathBuf> = fs::read_dir(train.join(digit.to_string()))
                .unwrap()
                .map(|e| e.unwrap().path())
                .collect();
            files.sort();
            for path in files.iter().take(10) {
                // a sample is named after its MNIST index plus one
                let id: usize = path.file_stem().unwrap().to_str().unwrap().parse().unwrap();
                assert_eq!(mnist.label(id - 1), digit, "{}", path.display());
                let sample = bucket(&fs::read(path).unwrap()).unwrap();
                let mut real = Events::new();
                real.replace(SIDE, &sample.start, &sample.ids);
                assert!(
                    (290..=330).contains(&real.steps()),
                    "three saccades of 100 ms"
                );
                let mut count = vec![0.0; SIDE * SIDE];
                let mut image = vec![0.0; SIDE * SIDE];
                sample
                    .ids
                    .iter()
                    .for_each(|&i| count[i as usize % (SIDE * SIDE)] += 1.0);
                for (k, &p) in mnist.image(id - 1).iter().enumerate() {
                    image[(k / 28 + 3) * SIDE + k % 28 + 3] = p as f32;
                }
                placed.push(correlation(&count, &image));
                sim.generate(mnist.image(id - 1), 28, 28, &s);
                let (r, g) = (leg_maps(&real), leg_maps(&sim));
                alike.push(
                    r.iter()
                        .zip(&g)
                        .map(|(a, b)| correlation(a, b))
                        .sum::<f32>()
                        / 6.0,
                );
            }
        }
        placed.sort_by(f32::total_cmp);
        let mean = alike.iter().sum::<f32>() / alike.len() as f32;
        assert!(
            placed[placed.len() / 2] > 0.6,
            "median {}",
            placed[placed.len() / 2]
        );
        assert!(mean > 0.45, "simulated against recorded events: {mean}");
    }

    #[test]
    fn truncate_keeps_the_first_recordings() {
        let root = tree("cut", &[2, 2]);
        let mut r = Recordings::load(&root, true, None).unwrap();
        r.truncate(3);
        assert_eq!((r.len(), r.label(2)), (3, 0));
        fs::remove_dir_all(&root).ok();
    }
}
