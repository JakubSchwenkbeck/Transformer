#![allow(dead_code)]
use ndarray::Array2;
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
}
