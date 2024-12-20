use crate::activation::activation_functions::gelu;
use contracts::requires;
use ndarray::{Array1, Array2, Array3};
use rand::Rng;
use std::ops::Add;

pub struct FeedForwardLayer {
    weights1: Array2<f32>,
    bias1: Array1<f32>, // Weights and biases for the first linear layer
    weights2: Array2<f32>,
    bias2: Array1<f32>,           // Weights and biases for the second linear layer
    dropout_rate: f32,            // Dropout rate
    pub(crate) input_size: usize, // Input feature size
    pub(crate) output_size: usize, // Output feature size
    initialized: bool,
}

impl FeedForwardLayer {
    /// Initializes the FeedForwardLayer with random weights and biases.
    ///
    /// # Parameters:
    /// - `_batch_size`: Batch size (not stored, used for verification if needed).
    /// - `input_size`: Number of input features (d_model).
    /// - `output_size`: Number of output features (d_model).
    /// - `dropout_rate`: Probability of dropping a unit in dropout (0.0 to 1.0).
    #[requires(input_size > 0, "Input size must be greater than 0")]
    #[requires(output_size > 0, "Output size must be greater than 0")]
    #[requires((0.0..=1.0).contains(&dropout_rate), "Dropout rate must be in range [0.0, 1.0]")]
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
            input_size,
            output_size,
            initialized: true,
        }
    }

    /// Verifies that the layer is properly initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Performs a forward pass in training mode.
    ///
    /// # Parameters:
    /// - `input`: 2D input tensor of shape (batch_size * seq_length, input_size).
    /// - `train`: Whether to apply dropout.
    ///
    /// # Returns:
    /// - Output tensor of shape (batch_size * seq_length, output_size).
    #[requires(input.shape()[1] == self.input_size, "Input feature size must match layer's input size")]
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

    /// Performs a forward pass in evaluation mode.
    ///
    /// # Parameters:
    /// - `x`: Input tensor of shape (batch_size, seq_length, input_size).
    ///
    /// # Returns:
    /// - Output tensor of shape (batch_size, seq_length, output_size).
    #[requires(x.shape()[2] == self.input_size, "Input feature size must match layer's input size")]
    #[requires(!x.is_empty(), "Input tensor must not be empty")]
    pub fn forward(&self, x: Array3<f32>) -> Array3<f32> {
        let batch_size = x.shape()[0];
        let seq_length = x.shape()[1];
        let d_model = x.shape()[2];

        let reshaped_x = x.to_shape((batch_size * seq_length, d_model));

        match reshaped_x {
            Ok(valid_reshaped_x) => {
                let dot = valid_reshaped_x.dot(&self.weights1);
                let add = dot + &self.bias1;

                // First linear layer + GELU activation
                let hidden = gelu(&add.to_owned());
                let dot2 = hidden.dot(&self.weights2);

                // Second linear layer
                let output = dot2 + &self.bias2;

                // Reshape back to 3D
                output
                    .to_shape((batch_size, seq_length, self.output_size))
                    .unwrap()
                    .to_owned()
            }
            Err(ref e) => {
                eprintln!("Shape error: {}", e);
                x // Fallback to the original input on failure
            }
        }
    }

    /// Applies dropout to the input.
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

/// He initialization function.
fn he_initialization(input_size: usize, output_size: usize) -> Array2<f32> {
    let mut rng = rand::rng();
    let scale = (2.0 / input_size as f32).sqrt();
    let values: Vec<f32> = (0..(input_size * output_size))
        .map(|_| rng.random_range(-scale..scale))
        .collect();
    Array2::from_shape_vec((input_size, output_size), values).unwrap()
}

/// Initializes bias vectors with zeros.
fn bias_initialization(size: usize) -> Array1<f32> {
    Array1::zeros(size)
}
