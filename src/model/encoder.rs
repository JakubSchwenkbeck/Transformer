#![allow(unused_imports)]
use crate::attention::multihead_attention::multi_head_attention;
use crate::data::learnable::initialize_weights;
use crate::layers::feedforward_layer::FeedForwardLayer;
use crate::layers::normalization::layer_norm;
use crate::settings::{BATCH_SIZE, EMBEDDING_SIZE, NUM_HEADS};
use contracts::{ensures, requires};
use ndarray::{array, Array2, Array3};
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

#[requires(input.shape().len() == 3, "Input tensor must have 3 dimensions (batch_size, seq_length, embed_size)")]
#[requires(input.shape()[2] == gamma.shape()[1], "Gamma dimensions do not match input feature size")]
#[requires(gamma.shape()[0] == 1, "Gamma must have exactly one row")]
#[requires(input.shape()[2] == beta.shape()[1], "Beta dimensions do not match input feature size")]
#[requires(beta.shape()[0] == 1, "Beta must have exactly one row")]
#[requires(epsilon > 0.0, "Epsilon must be positive and non-zero")]
#[requires(feed_forward_layer.is_initialized(), "Feed-forward layer is not properly initialized")]
#[requires(input.shape()[1] > 0, "Sequence length must be greater than zero")]

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
    assert_eq!(
        gamma.shape()[1],
        d_model,
        "Gamma dimensions do not match input feature size"
    );
    assert_eq!(gamma.shape()[0], 1, "Gamma must have exactly one row");
    assert_eq!(
        beta.shape()[1],
        d_model,
        "Beta dimensions do not match input feature size"
    );
    assert_eq!(beta.shape()[0], 1, "Beta must have exactly one row");
    assert!(epsilon > 0.0, "Epsilon must be positive and non-zero");
    assert!(
        feed_forward_layer.is_initialized(),
        "Feed-forward layer is not properly initialized"
    );
    assert!(seq_length > 0, "Sequence length must be greater than zero");

    // Multi-Head Attention
    let weights = feed_forward_layer.learnables;
    let attention_output = multi_head_attention(
        input.clone(),                  // Q
        input.clone(),                  // K
        input.clone(),                  // V
        NUM_HEADS,                              // Number of heads
        false,                          // No masking
        weights.query_weights.clone(),  // W_Q
        weights.key_weights.clone(),    // W_K
        weights.value_weights.clone(),  // W_V
        weights.output_projection.clone(), // W_O
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

    // Add & Normalize (Residual Connection + Layer Norm)
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

    assert_eq!(
        output.shape(),
        input.shape(),
        "Output tensor must have the same shape as the input tensor"
    );

    output
}
