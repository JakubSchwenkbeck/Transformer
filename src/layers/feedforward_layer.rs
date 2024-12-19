#![allow(dead_code)]
#![allow(unused_imports)]

use crate::activation::activation_functions::gelu;
use crate::settings::HIDDEN_SIZE;
use ndarray::{array, Array1, Array2, Array3};
use rand::Rng;
use std::ops::Add;

pub struct FeedForwardLayer {
    weights1: Array2<f32>,
    bias1: Array1<f32>, // weights and biases for first linear layer

    weights2: Array2<f32>,
    bias2: Array1<f32>, // weights and biases for second linear layer

    dropout_rate: f32, // Dropout rate
    initialized: bool,
}
impl FeedForwardLayer {
    // init with random values
    pub fn new(
        _batch_size: usize,
        input_size: usize,
        output_size: usize,
        dropout_rate: f32,
    ) -> FeedForwardLayer {
        let hidden_size = input_size * 4; // Define the hidden layer size

        // He (Kaiming) initialization for weights
        let weights1 = he_initialization(input_size, hidden_size); // Shape: (input_size, hidden_size)
        let bias1 = bias_initialization(hidden_size); // Shape: (hidden_size,)

        let weights2 = he_initialization(hidden_size, output_size); // Shape: (hidden_size, output_size)
        let bias2 = bias_initialization(output_size); // Shape: (output_size,)
        FeedForwardLayer {
            weights1,
            bias1,
            weights2,
            bias2,
            dropout_rate,
            initialized: true,
        }
    }
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
    pub fn forward_t(&self, input: &Array2<f32>, train: bool) -> Array2<f32> {
        // First linear layer
        let first_dot = input.dot(&self.weights1);
        let first_output = first_dot.add(&self.bias1);
        let first_activation = gelu(&first_output);

        // Dropout (only during training)
        let first_activation = if train {
            self.apply_dropout(&first_activation)
        } else {
            first_activation
        };

        // Second linear layer
        first_activation.dot(&self.weights2).add(&self.bias2)
    }
    /// Forward pass through the feed-forward layer.
    ///
    /// # Parameters:
    /// - `x`: Input tensor of shape (batch_size, seq_length, d_model).
    ///
    /// # Returns:
    /// - Output tensor of shape (batch_size, seq_length, d_model).
    pub fn forward(&self, x: Array3<f32>) -> Array3<f32> {
        let batch_size = x.shape()[0];
        let seq_length = x.shape()[1];
        let d_model = x.shape()[2];

        // Flatten the input to 2D: (batch_size * seq_length, d_model)
        let reshaped_x = x.to_shape((batch_size * seq_length, d_model));

        match reshaped_x {
            Ok(valid_reshaped_x) => {
                let dot = valid_reshaped_x.dot(&self.weights1);

                let add = dot + &self.bias1;

                // First linear layer + gelu

                let hidden = gelu(&add.to_owned());

                let dot2 = hidden.dot(&self.weights2);

                // Second linear layer
                let output = dot2 + &self.bias2;

                // Reshape back to 3D: (batch_size, seq_length, d_model)
                output
                    .to_shape((batch_size, seq_length, d_model))
                    .unwrap()
                    .to_owned()
                // Use the `hidden` result here for further processing.
            }
            Err(ref e) => {
                eprintln!("Shape error: {}", e);
                eprintln!(
                    "Shape of input : {:?}   -=-   Shape of weights : {:?} ",
                    reshaped_x.unwrap().shape(),
                    seq_length
                );
                // Or return unchanged?
                x
            }
        }
    }

    fn apply_dropout(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut rng = rand::rng();
        input.map(|&x| {
            if rng.random::<f32>() < self.dropout_rate {
                0.0
            } else {
                x
            }
        })
    }
}

fn he_initialization(input_size: usize, output_size: usize) -> Array2<f32> {
    let mut rng = rand::rng();
    // He initialization: scale by sqrt(2 / input_size)
    let scale = (2.0 / input_size as f32).sqrt();
    let values: Vec<f32> = (0..(input_size * output_size))
        .map(|_| rng.random_range(-scale..scale))
        .collect();

    // Create an Array2 from the values vector
    Array2::from_shape_vec((input_size, output_size), values).unwrap()
}

fn bias_initialization(size: usize) -> Array1<f32> {
    Array1::zeros(size)
}

fn test_bias_initialization() {
    let size = 5;

    let bias = bias_initialization(size);

    // Check that the dimensions are correct (size x 1)
    assert_eq!(bias.shape(), &[size,]);

    // Check that all values in the bias array are 0.0
    for &value in bias.iter() {
        assert_eq!(value, 0.0);
    }
}

#[test]
fn test_feedforward_forward() {
    // Define a dummy input with shape (batch_size, seq_length, d_model)
    let input = array![
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ],
        [
            [1.3, 1.4, 1.5, 1.6],
            [1.7, 1.8, 1.9, 2.0],
            [2.1, 2.2, 2.3, 2.4],
        ]
    ];

    // Create a FeedForwardLayer instance
    let feed_forward_layer = FeedForwardLayer::new(2, 4, 4, 0.1);

    // Feed forward through the layer
    let feed_forward_output = feed_forward_layer.forward(input.clone());

    // Assert the output shape
    assert_eq!(feed_forward_output.shape(), &[2, 3, 4]);

    // Optionally, check if the output is transformed (e.g., not equal to input)
    assert!(!feed_forward_output.iter().eq(input.iter())); // Check if output is different from input
}
