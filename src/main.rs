mod classification;
mod neuron;
mod receptive_field;
mod spike_train;
mod weight_initialization;

use classification::{ClassificationNetwork, ClassificationParams, load_images_from_dir};
use weight_initialization::load_learned_weights_from_dir;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let params = ClassificationParams {
        time_steps: 200,
        t_back: -20,
        t_fore: 20,
        pth: 150.0,
        num_input_neurons: 784,
        num_output_neurons: 8,
        epochs: 1,
        num_images: 6,
        w_max: 0.5,
        w_min: -0.5,
    };

    let image_names = ["1.png", "2.png", "3.png", "4.png", "5.png", "6.png"];
    let learned_weights = match load_learned_weights_from_dir("training_split", &image_names) {

        Ok(weights) => {
            println!("Loaded {} learned weight vectors", weights.len());
            weights
        }

        Err(_) => {
            println!("Could not load learned weights, using random initialization");
            vec![]
        }
    };
    
    let mut network = ClassificationNetwork::new(params, learned_weights);
    
    let test_images = match load_images_from_dir("test_split", &image_names) {

        Ok(images) => {
            println!("loaded {} test images", images.len());
            images
        }

        Err(_) => {
            println!("could not load test images, using example test images");
            classification::create_test_images()
        }
    };

    let results = network.classify_images(&test_images);
    for (i, result) in results.iter().enumerate() {

        println!("image {}: {:?}", i+1, result);
    
        
        if let Some(max_idx) = result.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
            println!("-> predicted class: {}", max_idx.0);
        }
    }
    
    let mut neuron = neuron::Neuron::new();
    let spike_train = vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let spikes = neuron.simulate(&spike_train, spike_train.len());
    
    println!("input spike train: {:?}", spike_train);
    println!("output spikes: {:?}", spikes);
    println!("total spikes: {}", spikes.iter().sum::<i32>());
    
    use image::GrayImage;
    use receptive_field::apply_receptive_field;
    
    let test_img = GrayImage::new(28, 28);
    let pot = apply_receptive_field(&test_img);
    println!("trying receptive field to a 28x28 image");
    println!("potential map dimensions: {:?}", pot.dim());

    Ok(())
}
