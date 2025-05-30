use ndarray::Array2;
use image::GrayImage;

pub fn apply_receptive_field(img: &GrayImage) -> Array2<f64> {
    let (height, width) = img.dimensions();

    // potential matrix
    let mut pot = Array2::zeros((height as usize, width as usize));
    
    let filter = [
        [-0.5, -0.125,  0.25,  -0.125, -0.5  ],
        [-0.125,  0.25,  0.625,  0.25,  -0.125],
        [ 0.25,   0.625, 1.0,    0.625,  0.25 ],
        [-0.125,  0.25,  0.625,  0.25,  -0.125],
        [-0.5,   -0.125, 0.25,  -0.125, -0.5  ]
    ];

    let ox = 2;
    let oy = 2;
    let range = [-2, -1, 0, 1, 2];
    
    for i in 0..height as usize {
        for j in 0..width as usize {
            let mut sum = 0.0;

            for (m_idx, &m) in range.iter().enumerate() {
                for (n_idx, &n) in range.iter().enumerate() {
                    let y = i as i32 + m;
                    let x = j as i32 + n;
                    
                    if y >= 0 && y < height as i32 && x >= 0 && x < width as i32 {
                        let pixel_value = img.get_pixel(x as u32, y as u32)[0] as f64 / 255.0;
                        sum += filter[(ox + m) as usize][(oy + n) as usize] * pixel_value;
                    }
                }
            }

            pot[[i, j]] = sum;
        }
    }
    
    pot
}

pub fn apply_receptive_field_to_array(input: &Array2<f64>) -> Array2<f64> {

    let (height, width) = input.dim();
    let mut pot = Array2::zeros((height, width));
    
    let filter = [
        [-0.5, -0.125,  0.25,  -0.125, -0.5  ],
        [-0.125,  0.25,  0.625,  0.25,  -0.125],
        [ 0.25,   0.625, 1.0,    0.625,  0.25 ],
        [-0.125,  0.25,  0.625,  0.25,  -0.125],
        [-0.5,   -0.125, 0.25,  -0.125, -0.5  ]
    ];
    
    let range = [-2, -1, 0, 1, 2];
    let ox = 2;
    let oy = 2;
    
    for i in 0..height {
        for j in 0..width {
            let mut sum = 0.0;
            
            for (m_idx, &m) in range.iter().enumerate() {
                for (n_idx, &n) in range.iter().enumerate() {
                    let y = i as i32 + m;
                    let x = j as i32 + n;
                    
                    if y >= 0 && y < height as i32 && x >= 0 && x < width as i32 {
                        sum += filter[(ox + m) as usize][(oy + n) as usize] * input[[y as usize, x as usize]];
                    }
                }
            }
            
            pot[[i, j]] = sum;
        }
    }
    
    pot
}

pub fn create_center_surround_filter(size: usize) -> Array2<f64> {

    let mut filter = Array2::zeros((size, size));
    let center = size / 2;
    
    for i in 0..size {
        for j in 0..size {
            let distance = ((i as f64 - center as f64).powi(2) + (j as f64 - center as f64).powi(2)).sqrt();
            if distance <= center as f64 {
                filter[[i, j]] = 1.0;
            } else {
                filter[[i, j]] = -0.5;
            }
        }
    }
    
    filter
}

// applies a custom filter
pub fn apply_custom_filter(img: &GrayImage, filter: &Array2<f64>) -> Array2<f64> {

    let (height, width) = img.dimensions();
    let (f_height, f_width) = filter.dim();
    let mut result = Array2::zeros((height as usize, width as usize));
    
    let pad_w = f_width / 2;
    let pad_h = f_height / 2;
    
    for i in 0..height as usize {
        for j in 0..width as usize {

            let mut sum = 0.0;
            
            for fi in 0..f_height {
                for fj in 0..f_width {

                    let y = i as i32 + fi as i32 - pad_h as i32;
                    let x = j as i32 + fj as i32 - pad_w as i32;
                    
                    if y >= 0 && y < height as i32 && x >= 0 && x < width as i32 {
                        let pixel_value = img.get_pixel(x as u32, y as u32)[0] as f64 / 255.0;
                        sum += filter[[fi, fj]] * pixel_value;
                    }
                }
            }
            
            result[[i, j]] = sum;
        }
    }
    
    result
}


#[cfg(test)]
mod tests {

    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_center_surround_filter() {
        
        let filter = create_center_surround_filter(5);
        assert_eq!(filter.dim(), (5, 5));

        // center should be positive
        assert_eq!(filter[[2, 2]], 1.0);
    }

    #[test]
    fn test_custom_filter() {
        let filter = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(filter.dim(), (2, 2));
    }
}
