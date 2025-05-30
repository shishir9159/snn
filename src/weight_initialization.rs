use ndarray::Array2;
use image::DynamicImage;
use std::fs;

pub fn load_learned_weights(image_paths: &[&str]) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {

    let mut weights = Vec::new();
    
    for path in image_paths {
        let img = image::open(path)?;
        let gray_img = img.to_luma8();
        let (height, width) = gray_img.dimensions();
        
        let mut weight_vector = Vec::new();
        for y in 0..height {
            for x in 0..width {

                let pixel_value = gray_img.get_pixel(x, y)[0];
                let weight = if pixel_value == 0 {
                    -0.9
                } else {
                    1.0
                };

                weight_vector.push(weight);
            }
        }
        
        weights.push(weight_vector);
    }
    
    Ok(weights)
}

pub fn load_learned_weights_from_dir(dir_path: &str, image_names: &[&str]) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    
    let mut weights = Vec::new();
    
    for name in image_names {
        let full_path = format!("{}/{}", dir_path, name);
        let img = image::open(&full_path)?;
        let gray_img = img.to_luma8();
        let (height, width) = gray_img.dimensions();
        
        let mut weight_vector = Vec::new();
        for y in 0..height {
            for x in 0..width {
    
                let pixel_value = gray_img.get_pixel(x, y)[0];
                let weight = if pixel_value == 0 {
                    -0.9
                } else {
                    1.0
                };
    
                weight_vector.push(weight);
            }
        }
        
        weights.push(weight_vector);
    }
    
    Ok(weights)
}

pub fn initialize_random_weights(num_neurons: usize, num_inputs: usize, w_min: f64, w_max: f64) -> Array2<f64> {

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut weights = Array2::zeros((num_neurons, num_inputs));
    
    for i in 0..num_neurons {
        for j in 0..num_inputs {
            weights[[i, j]] = rng.gen_range(w_min..=w_max);
        }
    }
    
    weights
}

pub fn initialize_xavier_weights(num_neurons: usize, num_inputs: usize) -> Array2<f64> {

    use rand::Rng;

    let mut rng = rand::thread_rng();
    let mut weights = Array2::zeros((num_neurons, num_inputs));
    
    let limit = (6.0 / (num_neurons + num_inputs) as f64).sqrt();
    
    for i in 0..num_neurons {
        for j in 0..num_inputs {
            weights[[i, j]] = rng.gen_range(-limit..=limit);
        }
    }
    
    weights
}

pub fn initialize_he_weights(num_neurons: usize, num_inputs: usize) -> Array2<f64> {

    use rand::Rng;

    let mut rng = rand::thread_rng();
    let mut weights = Array2::zeros((num_neurons, num_inputs));
    
    let std_dev = (2.4 / num_inputs as f64).sqrt();
    
    for i in 0..num_neurons {
        for j in 0..num_inputs {
            weights[[i, j]] = rng.gen_range(-std_dev..=std_dev);
        }
    }
    
    weights
}

pub fn save_weights(weights: &Array2<f64>, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {

    let mut content = String::new();
    let (rows, cols) = weights.dim();
    
    content.push_str(&format!("{} {}\n", rows, cols));
    
    for i in 0..rows {
        for j in 0..cols {
            content.push_str(&format!("{} ", weights[[i, j]]));
        }

        content.push('\n');
    }
    
    fs::write(file_path, content)?;
    Ok(())
}

pub fn load_weights(file_path: &str) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(file_path)?;
    let lines: Vec<&str> = content.lines().collect();
    
    if lines.is_empty() {
        return Err("Empty file".into());
    }
    
    let first_line: Vec<&str> = lines[0].split_whitespace().collect();
    if first_line.len() != 2 {
        return Err("Invalid format".into());
    }
    
    let rows: usize = first_line[0].parse()?;
    let cols: usize = first_line[1].parse()?;
    
    let mut weights = Array2::zeros((rows, cols));
    
    for (i, line) in lines.iter().enumerate().skip(1) {
        if i - 1 >= rows {
            break;
        }
        
        let values: Vec<f64> = line.split_whitespace()
            .map(|s| s.parse().unwrap_or(0.0))
            .collect();
        
        for (j, &value) in values.iter().enumerate() {
            if j < cols {
                weights[[i - 1, j]] = value;
            }
        }
    }
    
    Ok(weights)
}

pub fn create_weight_matrix(
    learned_weights: &[Vec<f64>],
    num_neurons: usize,
    num_inputs: usize,
    w_min: f64,
    w_max: f64
) -> Array2<f64> {
    let mut weights = initialize_random_weights(num_neurons, num_inputs, w_min, w_max);
    
    for (i, learned_weight) in learned_weights.iter().enumerate() {
        if i < num_neurons && learned_weight.len() == num_inputs {
            for j in 0..num_inputs {
                weights[[i, j]] = learned_weight[j];
            }
        }
    }
    weights
}