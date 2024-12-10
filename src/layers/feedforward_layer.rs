#![allow(dead_code)]

use crate::activation::activation_functions::gelu;
use ndarray::Array2;
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
        dropout: f32,
    ) -> FeedForwardLayer {
        let weights1 = Array2::<f32>::zeros((input_size, hidden_dim));
        let bias1 = Array2::<f32>::zeros((input_size, input_size));
        let weights2 = Array2::<f32>::zeros((hidden_dim, output_size));
        let bias2 = Array2::<f32>::zeros((input_size, input_size));

        FeedForwardLayer {
            weights1,
            bias1,
            weights2,
            bias2,
            dropout_rate: dropout,
        }
    }

    pub fn forward_t(&self, input: &Array2<f32>) -> Array2<f32> {
        let first_dot = input.dot(&self.weights1);
        let first_output = first_dot.add(&self.bias1);
        let first_activation = gelu(&first_output);

        gelu(&first_activation.dot(&self.weights2).add(&self.bias2))
    }
}
