use crate::activation::activation_functions::gelu;
use crate::data::learnable::LearnableWeights;
use contracts::requires;
use ndarray::{Array1, Array2, Array3};
use rand::Rng;
use std::ops::Add;

pub struct FeedForwardLayer<'a> {
    weights1: &'a Array2<f32>,
    bias1: &'a Array1<f32>, // Weights and biases for the first linear layer
    weights2: &'a Array2<f32>,
    bias2: &'a Array1<f32>, // Weights and biases for the second linear layer
    dropout_rate: f32,      // Dropout rate
    pub(crate) input_size: usize, // Input feature size
    pub(crate) output_size: usize, // Output feature size
    initialized: bool,
    pub learnables:  &'a LearnableWeights,
}

impl<'a> FeedForwardLayer<'a> {
    /// Initializes the FeedForwardLayer with weights from LearnableWeights.
    ///
    /// # Parameters:
    /// - `learnable_weights`: An instance of LearnableWeights to be used for this layer.
    /// - `dropout_rate`: Probability of dropping a unit in dropout (0.0 to 1.0).
    //#[requires(learnable_weights.is_initialized(), "LearnableWeights must be initialized")]
    pub fn new(learnable_weights: &'a LearnableWeights, dropout_rate: f32) -> FeedForwardLayer<'a> {
        FeedForwardLayer {
            weights1: &learnable_weights.linear1_weights,
            bias1: &learnable_weights.bias1,
            weights2: &learnable_weights.linear2_weights,
            bias2: &learnable_weights.bias2,
            dropout_rate,
            input_size: learnable_weights.linear1_weights.shape()[0],
            output_size: learnable_weights.linear2_weights.shape()[1],
            initialized: true,
            learnables: learnable_weights,
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
    #[requires(input.shape()[0] > 0, "Input tensor must not be empty")]
    #[requires(input.shape()[1] == self.input_size, "Input tensor's second dimension must match input_size")]
    pub fn forward_t(&self, input: &Array2<f32>, train: bool) -> Array2<f32> {
        // First linear layer
        let first_dot = input.dot(self.weights1);
        let first_output = first_dot.add(self.bias1);
        let first_activation = gelu(&first_output);

        // Dropout (only during training)
        let first_activation = if train {
            self.apply_dropout(&first_activation)
        } else {
            first_activation
        };

        // Second linear layer
        first_activation.dot(self.weights2).add(self.bias2)
    }

    /// Performs a forward pass in evaluation mode.
    ///
    /// # Parameters:
    /// - `x`: Input tensor of shape (batch_size, seq_length, input_size).
    ///
    /// # Returns:
    /// - Output tensor of shape (batch_size, seq_length, output_size).
    #[requires(x.shape()[0] > 0, "Input tensor must not be empty")]
    pub fn forward(&self, x: Array3<f32>) -> Array3<f32> {
        let batch_size = x.shape()[0];
        let seq_length = x.shape()[1];
        let d_model = x.shape()[2];

        let reshaped_x = x.to_shape((batch_size * seq_length, d_model));

        match reshaped_x {
            Ok(valid_reshaped_x) => {
                let dot = valid_reshaped_x.dot(self.weights1);
                let add = dot + self.bias1;

                // First linear layer + GELU activation
                let hidden = gelu(&add.to_owned());
                let dot2 = hidden.dot(self.weights2);

                // Second linear layer
                let output = dot2 + self.bias2;

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
