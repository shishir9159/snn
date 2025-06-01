use ndarray::Array2;
use crate::neuron::Neuron;
use crate::spike_train::encode_image_stochastic;
use crate::receptive_field::apply_receptive_field;
use crate::weight_initialization::create_weight_matrix;
use image::{DynamicImage, GrayImage};

#[derive(Debug, Clone)]
pub struct ClassificationParams {
    pub pth: f64,
    pub epochs: usize,
    pub time_steps: usize,
    pub t_back: i32,
    pub t_fore: i32,
    pub num_images: usize,
    pub num_input_neurons: usize,
    pub num_output_neurons: usize,
    pub w_max: f64,
    pub w_min: f64,
}

impl Default for ClassificationParams {
    fn default() -> Self {
        Self {
            time_steps: 200,
            t_back: -20,
            t_fore: 20,
            pth: 150.0,
            num_input_neurons: 784, // 28x28 for MNIST
            num_output_neurons: 8,
            epochs: 1,
            num_images: 6,
            w_max: 0.5,
            w_min: -0.5,
        }
    }
}

// winner-takes-it-all
pub struct ClassificationNetwork {
    params: ClassificationParams,
    output_neurons: Vec<Neuron>,
    synapse_weights: Array2<f64>,
    learned_weights: Vec<Vec<f64>>,
}

impl ClassificationNetwork {
    // classification network
    pub fn new(params: ClassificationParams, learned_weights: Vec<Vec<f64>>) -> Self {

        let mut output_neurons = Vec::new();
        
        for _ in 0..params.num_output_neurons {
            output_neurons.push(Neuron::new());
        }
        
        let synapse_weights = create_weight_matrix(
            &learned_weights,
            params.num_output_neurons,
            params.num_input_neurons,
            params.w_min,
            params.w_max,
        );
        
        Self {
            params,
            output_neurons,
            synapse_weights,
            learned_weights,
        }
    }
    
    pub fn classify_image(&mut self, img: &DynamicImage) -> Vec<f64> {

        for neuron in &mut self.output_neurons {
            neuron.initial();
        }
        
        let gray_img = img.to_luma8();
        let _pot = apply_receptive_field(&gray_img);        
        let spike_train = encode_image_stochastic(img, self.params.time_steps);
        
        let mut flat_spike_train = Vec::new();

        for t in 0..self.params.time_steps {
            for pixel_spikes in &spike_train {
                if t < pixel_spikes.len() {
                    flat_spike_train.push(pixel_spikes[t]);
                } else {
                    flat_spike_train.push(0.0);
                }
            }
        }
        
        let mut input_spikes = Array2::zeros((self.params.num_input_neurons, self.params.time_steps));
        for t in 0..self.params.time_steps {
            for i in 0..self.params.num_input_neurons {
                let idx = t * self.params.num_input_neurons + i;
                if idx < flat_spike_train.len() {
                    input_spikes[[i, t]] = flat_spike_train[idx];
                }
            }
        }
        
        let mut spike_counts = vec![0.0; self.params.num_output_neurons];
        let mut f_spike = false;
        
        for t in 0..self.params.time_steps {
            let mut active_potentials = vec![0.0; self.params.num_output_neurons];
            
            for (j, neuron) in self.output_neurons.iter_mut().enumerate() {
                if neuron.t_rest < t as i32 {
                    let input_sum = (0..self.params.num_input_neurons)
                        .map(|i| self.synapse_weights[[j, i]] * input_spikes[[i, t]])
                        .sum::<f64>();
                    
                    neuron.update_potential(input_sum);
                    if neuron.p > neuron.prest {
                        neuron.p -= neuron.d;
                    }
                    active_potentials[j] = neuron.p;
                }
            }
            
            if !f_spike {
                if let Some(&max_pot) = active_potentials.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
                    if max_pot > self.params.pth {
                        f_spike = true;
                        if let Some(winner) = active_potentials.iter().position(|&x| x == max_pot) {
                            for (s, neuron) in self.output_neurons.iter_mut().enumerate() {
                                if s != winner {
                                    neuron.inhibit();
                                }
                            }
                        }
                    }
                }
            }
            
            for (j, neuron) in self.output_neurons.iter_mut().enumerate() {
                let spike = neuron.check();
                if spike == 1 {
                    spike_counts[j] += 1.0;
                    neuron.t_rest = t as i32 + neuron.t_ref;
                }
            }
        }
        
        spike_counts
    }
    
    pub fn classify_images(&mut self, images: &[DynamicImage]) -> Vec<Vec<f64>> {
        let mut results = Vec::new();
        
        for img in images {
            let result = self.classify_image(img);
            results.push(result);
        }
        
        results
    }
    
    pub fn train(&mut self, training_images: &[DynamicImage], labels: &[usize]) {
        for epoch in 0..self.params.epochs {
            for (img, &label) in training_images.iter().zip(labels.iter()) {
                let _result = self.classify_image(img);
                println!("Epoch {}: Training on image with label {}", epoch, label);
            }
        }
    }
    
    pub fn get_weights(&self) -> &Array2<f64> {
        &self.synapse_weights
    }
    
    pub fn update_weights(&mut self, new_weights: Array2<f64>) {
        self.synapse_weights = new_weights;
    }
}

pub fn load_images_from_dir(dir_path: &str, image_names: &[&str]) -> Result<Vec<DynamicImage>, Box<dyn std::error::Error>> {

    let mut images = Vec::new();
    
    for name in image_names {
        let full_path = format!("{}/{}", dir_path, name);
        let img = image::open(&full_path)?;
        images.push(img);
    }
    
    Ok(images)
}

pub fn create_test_images() -> Vec<DynamicImage> {

    let mut images = Vec::new();
    
    for _i in 0..6 {
        let img = GrayImage::new(28, 28);
        images.push(DynamicImage::ImageLuma8(img));
    }
    
    images
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_initialization::initialize_random_weights;

    #[test]
    fn test_classification_params() {
        let params = ClassificationParams::default();
        assert_eq!(params.time_steps, 200);
        assert_eq!(params.num_output_neurons, 8);
    }

    #[test]
    fn test_network_creation() {
        let params = ClassificationParams::default();
        let learned_weights = vec![vec![1.0; 784]];
        let network = ClassificationNetwork::new(params, learned_weights);
        assert_eq!(network.output_neurons.len(), 8);
    }
}



