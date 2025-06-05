//! Labelled grayscale image sets: MNIST IDX files, PNG folders and synthetic digits.
//! Images are row-major u8, bright strokes on a dark background.

use crate::rng::Rng;
use std::f32::consts::TAU;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Dataset {
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub classes: usize,
    pixels: Vec<u8>,
    labels: Vec<u8>,
}

impl Dataset {
    pub fn new(name: &str, width: usize, height: usize, pixels: Vec<u8>, labels: Vec<u8>) -> Self {
        assert_eq!(pixels.len(), width * height * labels.len());
        let classes = labels.iter().map(|&l| l as usize + 1).max().unwrap_or(0);
        Self {
            name: name.into(),
            width,
            height,
            classes,
            pixels,
            labels,
        }
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn image(&self, i: usize) -> &[u8] {
        let n = self.width * self.height;
        &self.pixels[i * n..(i + 1) * n]
    }

    pub fn label(&self, i: usize) -> u8 {
        self.labels[i]
    }

    /// Keeps at most the first `n` images.
    pub fn truncate(&mut self, n: usize) {
        if n < self.len() {
            self.labels.truncate(n);
            self.pixels.truncate(n * self.width * self.height);
        }
    }

    /// Moves the last `n` images into a second set.
    pub fn split_off(&mut self, n: usize) -> Dataset {
        let at = self.len() - n.min(self.len());
        let px = at * self.width * self.height;
        Dataset {
            name: self.name.clone(),
            width: self.width,
            height: self.height,
            classes: self.classes,
            pixels: self.pixels.split_off(px),
            labels: self.labels.split_off(at),
        }
    }
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Reads a file, inflating it when it is gzip compressed.
pub fn read_maybe_gz(path: &Path) -> io::Result<Vec<u8>> {
    let raw = fs::read(path)?;
    if raw.starts_with(&[0x1f, 0x8b]) {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(&raw[..]).read_to_end(&mut out)?;
        Ok(out)
    } else {
        Ok(raw)
    }
}

fn be_u32(b: &[u8], at: usize) -> io::Result<usize> {
    b.get(at..at + 4)
        .map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]) as usize)
        .ok_or_else(|| invalid("truncated idx header"))
}

/// Parses an IDX3 image file into (rows, cols, pixels).
pub fn parse_idx_images(b: &[u8]) -> io::Result<(usize, usize, Vec<u8>)> {
    if be_u32(b, 0)? != 0x0803 {
        return Err(invalid("not an idx3 image file"));
    }
    let (n, rows, cols) = (be_u32(b, 4)?, be_u32(b, 8)?, be_u32(b, 12)?);
    let body = b
        .get(16..16 + n * rows * cols)
        .ok_or_else(|| invalid("truncated idx image data"))?;
    Ok((rows, cols, body.to_vec()))
}

/// Parses an IDX1 label file.
pub fn parse_idx_labels(b: &[u8]) -> io::Result<Vec<u8>> {
    if be_u32(b, 0)? != 0x0801 {
        return Err(invalid("not an idx1 label file"));
    }
    let n = be_u32(b, 4)?;
    let body = b
        .get(8..8 + n)
        .ok_or_else(|| invalid("truncated idx label data"))?;
    Ok(body.to_vec())
}

fn find(dir: &Path, stems: &[String]) -> io::Result<PathBuf> {
    for stem in stems {
        for ext in ["", ".gz"] {
            let p = dir.join(format!("{stem}{ext}"));
            if p.is_file() {
                return Ok(p);
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("{} not found in {}", stems[0], dir.display()),
    ))
}

/// Loads the MNIST training (`train`) or test set from `dir`, plain or gzipped.
pub fn mnist(dir: &Path, train: bool) -> io::Result<Dataset> {
    let set = if train { "train" } else { "t10k" };
    let names = |kind: &str, idx: &str| {
        vec![
            format!("{set}-{kind}-{idx}-ubyte"),
            format!("{set}-{kind}.{idx}-ubyte"),
        ]
    };
    let (rows, cols, pixels) =
        parse_idx_images(&read_maybe_gz(&find(dir, &names("images", "idx3"))?)?)?;
    let labels = parse_idx_labels(&read_maybe_gz(&find(dir, &names("labels", "idx1"))?)?)?;
    if labels.len() * rows * cols != pixels.len() {
        return Err(invalid("mnist image and label counts differ"));
    }
    Ok(Dataset::new(
        &format!("mnist/{set}"),
        cols,
        rows,
        pixels,
        labels,
    ))
}

/// Loads `dir/<class>/*.png`, classes sorted by folder name. Images are converted to
/// grayscale, inverted when the background is light, and resized to `side`².
pub fn png_dir(dir: &Path, side: usize) -> io::Result<Dataset> {
    let mut classes: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    classes.sort();
    if classes.is_empty() || classes.len() > 256 {
        return Err(invalid(format!(
            "{}: expected 1-256 class folders",
            dir.display()
        )));
    }

    let (mut pixels, mut labels) = (Vec::new(), Vec::new());
    for (c, class) in classes.iter().enumerate() {
        let mut files: Vec<PathBuf> = fs::read_dir(class)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("png")))
            .collect();
        files.sort();
        for f in files {
            let img = image::open(&f).map_err(|e| invalid(format!("{}: {e}", f.display())))?;
            let img = image::imageops::resize(
                &img.to_luma8(),
                side as u32,
                side as u32,
                image::imageops::FilterType::Triangle,
            );
            let mut px = img.into_raw();
            let mean = px.iter().map(|&v| v as usize).sum::<usize>() / px.len().max(1);
            if mean > 127 {
                px.iter_mut().for_each(|v| *v = 255 - *v);
            }
            pixels.extend_from_slice(&px);
            labels.push(c as u8);
        }
    }
    let mut ds = Dataset::new(
        &format!("png/{}", dir.display()),
        side,
        side,
        pixels,
        labels,
    );
    ds.classes = classes.len();
    Ok(ds)
}

type Stroke = Vec<(f32, f32)>;

fn arc(cx: f32, cy: f32, rx: f32, ry: f32, from_deg: f32, to_deg: f32) -> Stroke {
    let n = (((to_deg - from_deg).abs() / 22.5).ceil() as usize).max(2);
    (0..=n)
        .map(|k| {
            let a = (from_deg + (to_deg - from_deg) * k as f32 / n as f32) * TAU / 360.0;
            (cx + rx * a.cos(), cy + ry * a.sin())
        })
        .collect()
}

/// Strokes of digit `d` in a unit box, y pointing down.
fn glyph(d: u8) -> Vec<Stroke> {
    match d {
        0 => vec![arc(0.5, 0.5, 0.26, 0.36, 0.0, 360.0)],
        1 => vec![vec![(0.42, 0.27), (0.55, 0.14), (0.55, 0.86)]],
        2 => {
            let mut s = arc(0.5, 0.33, 0.22, 0.19, 200.0, 380.0);
            s.extend([(0.27, 0.86), (0.76, 0.86)]);
            vec![s]
        }
        3 => vec![
            arc(0.5, 0.31, 0.2, 0.17, 200.0, 450.0),
            arc(0.5, 0.67, 0.23, 0.19, 270.0, 520.0),
        ],
        4 => vec![vec![(0.62, 0.86), (0.62, 0.14), (0.24, 0.64), (0.78, 0.64)]],
        5 => vec![
            vec![(0.72, 0.15), (0.36, 0.15), (0.34, 0.47)],
            arc(0.5, 0.64, 0.23, 0.21, 230.0, 520.0),
        ],
        6 => vec![
            vec![(0.68, 0.16), (0.5, 0.28), (0.33, 0.6)],
            arc(0.5, 0.66, 0.2, 0.2, 0.0, 360.0),
        ],
        7 => vec![vec![(0.25, 0.15), (0.75, 0.15), (0.45, 0.86)]],
        8 => vec![
            arc(0.5, 0.31, 0.18, 0.17, 0.0, 360.0),
            arc(0.5, 0.67, 0.22, 0.2, 0.0, 360.0),
        ],
        _ => vec![
            arc(0.5, 0.34, 0.2, 0.2, 0.0, 360.0),
            vec![(0.7, 0.34), (0.62, 0.86)],
        ],
    }
}

fn seg_dist(px: f32, py: f32, (ax, ay): (f32, f32), (bx, by): (f32, f32)) -> f32 {
    let (dx, dy) = (bx - ax, by - ay);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 {
        (((px - ax) * dx + (py - ay) * dy) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    ((px - ax - t * dx).powi(2) + (py - ay - t * dy).powi(2)).sqrt()
}

/// Renders digit `d` into a `side`² image with random scale, slant, rotation,
/// position and stroke width.
pub fn render_digit(d: u8, side: usize, rng: &mut Rng, out: &mut [u8]) {
    let s = side as f32;
    let size = s * 0.72 * rng.range(0.85, 1.1);
    let (sx, sy) = (size * rng.range(0.85, 1.15), size);
    let shear = rng.range(-0.25, 0.25);
    let rot = rng.range(-0.2, 0.2);
    let (c, sn) = (rot.cos(), rot.sin());
    let (cx, cy) = (
        s / 2.0 + rng.range(-2.0, 2.0),
        s / 2.0 + rng.range(-2.0, 2.0),
    );
    let half_width = rng.range(0.9, 1.6);
    let peak = rng.range(0.85, 1.0);
    let place = |(gx, gy): (f32, f32)| {
        let (x, y) = ((gx - 0.5 + shear * (gy - 0.5)) * sx, (gy - 0.5) * sy);
        (cx + c * x - sn * y, cy + sn * x + c * y)
    };

    let mut dist = vec![f32::MAX; side * side];
    for stroke in glyph(d) {
        let pts: Vec<(f32, f32)> = stroke.into_iter().map(place).collect();
        for w in pts.windows(2) {
            let (a, b) = (w[0], w[1]);
            let r = half_width + 1.0;
            let x0 = (a.0.min(b.0) - r).floor().max(0.0) as usize;
            let x1 = ((a.0.max(b.0) + r).ceil().max(0.0) as usize).min(side);
            let y0 = (a.1.min(b.1) - r).floor().max(0.0) as usize;
            let y1 = ((a.1.max(b.1) + r).ceil().max(0.0) as usize).min(side);
            for y in y0..y1 {
                for x in x0..x1 {
                    let d = seg_dist(x as f32 + 0.5, y as f32 + 0.5, a, b);
                    let slot = &mut dist[y * side + x];
                    *slot = slot.min(d);
                }
            }
        }
    }
    for (o, d) in out.iter_mut().zip(&dist) {
        *o = ((half_width + 0.5 - d).clamp(0.0, 1.0) * peak * 255.0) as u8;
    }
}

/// `n` jittered digits 0-9, balanced across classes, 28x28.
pub fn synthetic(n: usize, seed: u64) -> Dataset {
    const SIDE: usize = 28;
    let mut rng = Rng::new(seed);
    let mut pixels = vec![0u8; n * SIDE * SIDE];
    let labels: Vec<u8> = (0..n).map(|i| (i % 10) as u8).collect();
    for (img, &d) in pixels.chunks_exact_mut(SIDE * SIDE).zip(&labels) {
        render_digit(d, SIDE, &mut rng, img);
    }
    let mut ds = Dataset::new("synthetic", SIDE, SIDE, pixels, labels);
    ds.classes = 10;
    ds
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn idx_images(n: u32, rows: u32, cols: u32) -> Vec<u8> {
        let mut b = Vec::new();
        for v in [0x0803, n, rows, cols] {
            b.extend_from_slice(&v.to_be_bytes());
        }
        b.extend((0..n * rows * cols).map(|i| i as u8));
        b
    }

    fn idx_labels(labels: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        for v in [0x0801, labels.len() as u32] {
            b.extend_from_slice(&v.to_be_bytes());
        }
        b.extend_from_slice(labels);
        b
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("snn-{tag}-{}", std::process::id()));
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn idx_parse() {
        let (r, c, px) = parse_idx_images(&idx_images(2, 2, 3)).unwrap();
        assert_eq!((r, c), (2, 3));
        assert_eq!(px, (0..12).collect::<Vec<u8>>());
        assert_eq!(parse_idx_labels(&idx_labels(&[3, 7])).unwrap(), vec![3, 7]);
    }

    #[test]
    fn idx_rejects_bad_input() {
        assert!(parse_idx_images(&idx_labels(&[1])).is_err());
        assert!(parse_idx_labels(&idx_images(1, 1, 1)).is_err());
        let mut short = idx_images(2, 2, 2);
        short.pop();
        assert!(parse_idx_images(&short).is_err());
        assert!(parse_idx_images(&[0, 0]).is_err());
    }

    #[test]
    fn mnist_dir_plain_and_gz() {
        let dir = temp_dir("mnist");
        fs::write(dir.join("t10k-images-idx3-ubyte"), idx_images(3, 2, 2)).unwrap();
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        gz.write_all(&idx_labels(&[0, 4, 9])).unwrap();
        fs::write(dir.join("t10k-labels-idx1-ubyte.gz"), gz.finish().unwrap()).unwrap();

        let ds = mnist(&dir, false);
        let missing = mnist(&dir, true);
        fs::remove_dir_all(&dir).ok();
        let ds = ds.unwrap();
        assert_eq!((ds.len(), ds.width, ds.height, ds.classes), (3, 2, 2, 10));
        assert_eq!(ds.image(1), &[4, 5, 6, 7]);
        assert_eq!(ds.label(2), 9);
        assert!(missing.is_err());
    }

    #[test]
    fn png_dir_labels_resizes_and_inverts() {
        let dir = temp_dir("png");
        for (class, bg) in [("a", 0u8), ("b", 255u8)] {
            fs::create_dir_all(dir.join(class)).unwrap();
            let mut img = image::GrayImage::from_pixel(10, 10, image::Luma([bg]));
            img.put_pixel(5, 5, image::Luma([255 - bg]));
            img.save(dir.join(class).join("x.png")).unwrap();
        }
        let ds = png_dir(&dir, 28);
        fs::remove_dir_all(&dir).ok();
        let ds = ds.unwrap();
        assert_eq!((ds.len(), ds.width, ds.classes), (2, 28, 2));
        assert_eq!((ds.label(0), ds.label(1)), (0, 1));
        // the light-background image was inverted to a dark background
        assert_eq!(ds.image(1)[0], 0);
    }

    #[test]
    fn synthetic_is_seeded_balanced_and_centred() {
        let a = synthetic(40, 1);
        assert_eq!(a.image(7), synthetic(40, 1).image(7));
        assert_ne!(a.image(7), synthetic(40, 2).image(7));
        assert_eq!((a.len(), a.classes), (40, 10));
        assert_eq!((0..40).filter(|&i| a.label(i) == 3).count(), 4);
        for i in 0..40 {
            let img = a.image(i);
            let lit = img.iter().filter(|&&v| v > 128).count();
            assert!(
                (25..400).contains(&lit),
                "digit {} has {lit} lit pixels",
                a.label(i)
            );
            assert!(img[0] == 0 && img[783] == 0, "corners stay dark");
        }
    }

    #[test]
    fn synthetic_digits_differ() {
        let ds = synthetic(200, 3);
        let mut mean = vec![[0f32; 784]; 10];
        for i in 0..ds.len() {
            for (m, &v) in mean[ds.label(i) as usize].iter_mut().zip(ds.image(i)) {
                *m += v as f32;
            }
        }
        let dist = |a: &[f32; 784], b: &[f32; 784]| {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>()
        };
        for a in 0..10 {
            for b in a + 1..10 {
                assert!(dist(&mean[a], &mean[b]) > 0.1 * dist(&mean[a], &[0.0; 784]));
            }
        }
    }

    #[test]
    fn split_and_truncate() {
        let mut ds = synthetic(30, 0);
        let test = ds.split_off(10);
        assert_eq!((ds.len(), test.len()), (20, 10));
        assert_eq!(test.label(0), 0);
        ds.truncate(5);
        assert_eq!(ds.len(), 5);
        assert_eq!(ds.image(4).len(), 784);
    }
}
