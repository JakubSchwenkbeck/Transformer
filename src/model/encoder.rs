use crate::attention::multihead_attention::multi_head_attention;
use crate::layers::feedforward_layer::FeedForwardLayer;
use crate::layers::normalization::layer_norm;
use ndarray::{Array2, Array3};
use std::ops::Add;

/// Implements a Transformer Encoder layer.
///
/// # Parameters:
/// - `input`: Input tensor of shape (batch_size, seq_length, d_model).
/// - `gamma`: Learned scale parameter for layer normalization.
/// - `beta`: Learned shift parameter for layer normalization.
/// - `epsilon`: Small constant for numerical stability in layer normalization.
/// - `feed_forward_layer`: Feed-forward layer instance.
///
/// # Returns:
/// - Output tensor of shape (batch_size, seq_length, d_model) after passing through the encoder layer.
pub fn encoding(
    input: Array3<f32>,                    // Input tensor
    gamma: Array2<f32>,                    // Scale parameter for layer norm
    beta: Array2<f32>,                     // Shift parameter for layer norm
    epsilon: f32,                          // Small constant for numerical stability
    feed_forward_layer: &FeedForwardLayer, // Feed-forward layer instance
) -> Array3<f32> {
    let batch_size = input.shape()[0];
    let seq_length = input.shape()[1];
    let d_model = input.shape()[2];

    // Multi-Head Attention
    let dummy_learned_matrices = Array2::<f32>::zeros((d_model, d_model)); // Replace with actual learned parameters
    let attention_output = multi_head_attention(
        input.clone(),                  // Q
        input.clone(),                  // K
        input.clone(),                  // V
        4,                              // Number of heads
        false,                          // No masking
        dummy_learned_matrices.clone(), // W_Q
        dummy_learned_matrices.clone(), // W_K
        dummy_learned_matrices.clone(), // W_V
        dummy_learned_matrices.clone(), // W_O
    );

    // Add & Normalize (Residual Connection + Layer Norm)
    let attention_residual = attention_output.add(&input); // Residual connection
    let reshaped_attention = attention_residual
        .to_shape((batch_size * seq_length, d_model)) // Flatten to 2D
        .unwrap();
    let attention_norm = layer_norm(
        &reshaped_attention.to_owned(), // Convert to 2D for layer_norm
        &gamma,
        &beta,
        epsilon,
    )
    .to_shape((batch_size, seq_length, d_model))
    .unwrap()
    .to_owned();

    // Feed-Forward Network
    let feed_forward_output = feed_forward_layer.forward(attention_norm.clone());

    //  Add & Normalize (Residual Connection + Layer Norm)
    let feed_forward_residual = feed_forward_output.add(&attention_norm); // Residual connection
    let reshaped_ff_attention = feed_forward_residual
        .to_shape((batch_size * seq_length, d_model)) // Flatten to 2D
        .unwrap();
    let output = layer_norm(
        &reshaped_ff_attention.to_owned(), // Convert to 2D for layer_norm
        &gamma,
        &beta,
        epsilon,
    )
    .to_shape((batch_size, seq_length, d_model))
    .unwrap()
    .to_owned();

    output
}
