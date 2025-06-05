//! Weight initialisation and plain-text persistence.

use crate::rng::Rng;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// `n` weights uniform in [lo, hi).
pub fn random(n: usize, lo: f32, hi: f32, rng: &mut Rng) -> Vec<f32> {
    (0..n).map(|_| rng.range(lo, hi)).collect()
}

/// Writes `rows cols` on the first line, then one row per line.
pub fn save(path: &Path, rows: usize, cols: usize, w: &[f32]) -> io::Result<()> {
    assert_eq!(w.len(), rows * cols);
    let mut f = BufWriter::new(fs::File::create(path)?);
    writeln!(f, "{rows} {cols}")?;
    for row in w.chunks_exact(cols.max(1)) {
        let line: Vec<String> = row.iter().map(|v| v.to_string()).collect();
        writeln!(f, "{}", line.join(" "))?;
    }
    f.flush()
}

/// Reads a file written by [`save`]; returns (rows, cols, weights).
pub fn load(path: &Path) -> io::Result<(usize, usize, Vec<f32>)> {
    parse(&fs::read_to_string(path)?)
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Saves weight rows as a grid of `side`² tiles, `lo` black and `hi` white.
pub fn save_png(path: &Path, w: &[f32], side: usize, lo: f32, hi: f32) -> io::Result<()> {
    let n = w.len() / (side * side);
    let cols = ((n as f32).sqrt().ceil() as usize).max(1);
    let rows = n.div_ceil(cols).max(1);
    let (iw, ih) = (cols * (side + 1) - 1, rows * (side + 1) - 1);
    let mut px = vec![0u8; iw * ih];
    for (k, tile) in w.chunks_exact(side * side).enumerate() {
        let (ox, oy) = ((k % cols) * (side + 1), (k / cols) * (side + 1));
        for (i, &v) in tile.iter().enumerate() {
            let g = ((v - lo) / (hi - lo)).clamp(0.0, 1.0) * 255.0;
            px[(oy + i / side) * iw + ox + i % side] = g as u8;
        }
    }
    image::GrayImage::from_raw(iw as u32, ih as u32, px)
        .expect("buffer matches dimensions")
        .save(path)
        .map_err(io::Error::other)
}

pub fn parse(text: &str) -> io::Result<(usize, usize, Vec<f32>)> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| invalid("empty weight file".into()))?;
    let dims: Vec<usize> = header
        .split_whitespace()
        .map(|s| {
            s.parse()
                .map_err(|_| invalid(format!("bad dimension {s:?}")))
        })
        .collect::<io::Result<_>>()?;
    let [rows, cols] = dims[..] else {
        return Err(invalid(format!("expected `rows cols`, got {header:?}")));
    };

    let mut w = Vec::with_capacity(rows * cols);
    for (r, line) in lines.enumerate() {
        let before = w.len();
        for s in line.split_whitespace() {
            w.push(
                s.parse::<f32>()
                    .map_err(|_| invalid(format!("row {r}: bad value {s:?}")))?,
            );
        }
        if w.len() - before != cols {
            return Err(invalid(format!(
                "row {r}: {} values, expected {cols}",
                w.len() - before
            )));
        }
    }
    if w.len() != rows * cols {
        return Err(invalid(format!(
            "{} rows, expected {rows}",
            w.len() / cols.max(1)
        )));
    }
    Ok((rows, cols, w))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_is_in_range_and_seeded() {
        let a = random(1000, -0.5, 0.5, &mut Rng::new(4));
        assert!(a.iter().all(|w| (-0.5..0.5).contains(w)));
        assert_eq!(a, random(1000, -0.5, 0.5, &mut Rng::new(4)));
    }

    #[test]
    fn save_load_round_trip() {
        let w = random(3 * 5, -1.2, 2.0, &mut Rng::new(5));
        let path = std::env::temp_dir().join(format!("snn-weights-{}.txt", std::process::id()));
        save(&path, 3, 5, &w).unwrap();
        let loaded = load(&path);
        fs::remove_file(&path).ok();
        assert_eq!(loaded.unwrap(), (3, 5, w));
    }

    #[test]
    fn png_tiles_weight_rows() {
        let w: Vec<f32> = (0..5 * 4)
            .map(|i| if i % 4 == 0 { 1.0 } else { -1.0 })
            .collect();
        let path = std::env::temp_dir().join(format!("snn-weights-{}.png", std::process::id()));
        save_png(&path, &w, 2, -1.0, 1.0).unwrap();
        let img = image::open(&path).map(|i| i.to_luma8());
        fs::remove_file(&path).ok();
        // 5 tiles of 2x2 in a 3x2 grid with 1px gaps
        let img = img.unwrap();
        assert_eq!(img.dimensions(), (8, 5));
        assert_eq!(img.get_pixel(0, 0)[0], 255);
        assert_eq!(img.get_pixel(1, 0)[0], 0);
        assert_eq!(img.get_pixel(3, 0)[0], 255);
    }

    #[test]
    fn rejects_malformed_files() {
        assert!(parse("").is_err());
        assert!(parse("2\n1 2\n").is_err());
        assert!(parse("1 2\n1 x\n").is_err());
        assert!(parse("1 2\n1 2 3\n").is_err());
        assert!(parse("2 2\n1 2\n").is_err());
        assert_eq!(parse("1 2\n1 2\n").unwrap(), (1, 2, vec![1.0, 2.0]));
    }
}
