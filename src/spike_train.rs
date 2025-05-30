use ndarray::Array2;
use rand::Rng;
use image::{DynamicImage, GrayImage};
use std::f64;

pub fn encode_stochastic(img: &GrayImage, time_steps: usize) -> Vec<Vec<f64>> {

    let mut train = Vec::new();
    let (height, width) = img.dimensions();    
    let normalized = normalize_image(img);
    
    for y in 0..height {
        for x in 0..width {

            let intensity = normalized[[y as usize, x as usize]];
            let mut spike_train = Vec::new();
            
            for _ in 0..time_steps {
                let random_val: f64 = rand::thread_rng().gen();
                let spike = if random_val < intensity { 1.0 } else { 0.0 };
                spike_train.push(spike);
            }
            
            train.push(spike_train);
        }
    }
    
    train
}

pub fn encode_deterministic(pot: &Array2<f64>, time_steps: usize) -> Vec<Vec<f64>> {

    let mut train = Vec::new();
    let (rows, cols) = pot.dim();
    
    for i in 0..rows {
        for j in 0..cols {

            let potential = pot[[i, j]];
            let mut spike_train = vec![0.0; time_steps];
            let freq = interpolate(potential, -2.0, 5.0, 1.0, 10.0);

            if freq > 0 {
                let freq_step = (time_steps as f64 / freq).ceil() as usize;
                let mut k = freq_step;
                
                while k < time_steps {
                    spike_train[k] = 1.0;
                    k += freq_step;
                }
            }
            
            train.push(spike_train);
        }
    }
    
    train
}

fn normalize_image(img: &GrayImage) -> Array2<f64> {

    let (height, width) = img.dimensions();
    let mut array = Array2::zeros((height as usize, width as usize));

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y)[0] as f64 / 255.0;
            array[[y as usize, x as usize]] = pixel;
        }
    }
    
    let norm = array.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        array = array / norm;
    }
    
    array
}

fn interpolate(value: f64, in_min: f64, in_max: f64, out_min: f64, out_max: f64) -> f64 {
    if in_max == in_min {
        return out_min;
    }
    
    let ratio = (value - in_min) / (in_max - in_min);
    out_min + ratio * (out_max - out_min)
}

pub fn encode_image_stochastic(img: &DynamicImage, time_steps: usize) -> Vec<Vec<f64>> {
    let gray_img = img.to_luma8();
    encode_stochastic(&gray_img, time_steps)
}

pub fn encode_image_deterministic(img: &DynamicImage, time_steps: usize) -> Vec<Vec<f64>> {
    let gray_img = img.to_luma8();
    let (height, width) = gray_img.dimensions();
    let mut pot = Array2::zeros((height as usize, width as usize));
    
    for y in 0..height {
        for x in 0..width {
            let pixel = gray_img.get_pixel(x, y)[0] as f64 / 255.0;
            pot[[y as usize, y as usize]] = pixel;
        }
    }
    
    encode_deterministic(&pot, time_steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_interpolation() {
        assert_eq!(interpolate(0.0, -1.0, 1.0, 0.0, 10.0), 5.0);
        assert_eq!(interpolate(-1.0, -1.0, 1.0, 0.0, 10.0), 0.0);
        assert_eq!(interpolate(1.0, -1.0, 1.0, 0.0, 10.0), 10.0);
    }

    #[test]
    fn test_deterministic_encoding() {
        let pot = arr2(&[[0.5, 0.3], [0.8, 0.1]]);
        let train = encode_deterministic(&pot, 10);
        
        assert_eq!(train.len(), 4);
        assert_eq!(train[0].len(), 10);
    }
}

