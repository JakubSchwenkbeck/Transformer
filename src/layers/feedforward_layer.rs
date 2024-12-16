#![allow(dead_code)]

use crate::activation::activation_functions::gelu;
use ndarray::{Array2, Array3};
use rand::Rng;
use std::ops::Add;

pub struct FeedForwardLayer {
    weights1: Array2<f32>,
    bias1: Array2<f32>, // weights and biases for first linear layer

    weights2: Array2<f32>,
    bias2: Array2<f32>, // weights and biases for second linear layer

    dropout_rate: f32, // Dropout rate
}
impl FeedForwardLayer {
    // init with random values
    pub fn new(
        input_size: usize,
        output_size: usize,
        hidden_dim: usize,
        dropout_rate: f32,
    ) -> FeedForwardLayer {
        // He (Kaiming) initialization for weights
        let weights1 = he_initialization(input_size, hidden_dim);
        let bias1 = bias_initialization(hidden_dim);

        let weights2 = he_initialization(hidden_dim, output_size);
        let bias2 = bias_initialization(output_size);

        FeedForwardLayer {
            weights1,
            bias1,
            weights2,
            bias2,
            dropout_rate,
        }
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
            let reshaped_x = x.to_shape((batch_size * seq_length, d_model)).unwrap();

            // First linear layer + ReLU
            let hidden = (reshaped_x.dot(&self.weights1) + &self.bias1).mapv(gelu);

            // Second linear layer
            let output = hidden.dot(&self.weights2) + &self.bias2;

            // Reshape back to 3D: (batch_size, seq_length, d_model)
            output.to_shape((batch_size, seq_length, d_model)).unwrap().to_owned()
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

fn bias_initialization(size: usize) -> Array2<f32> {
    Array2::zeros((size, 1)) // Biases are usually initialized to zero
}
fn test_bias_initialization() {
    let size = 5;

    let bias = bias_initialization(size);

    // Check that the dimensions are correct (size x 1)
    assert_eq!(bias.shape(), &[size, 1]);

    // Check that all values in the bias array are 0.0
    for &value in bias.iter() {
        assert_eq!(value, 0.0);
    }
}
