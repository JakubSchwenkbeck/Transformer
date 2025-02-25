#![allow(warnings)]
use crate::attention::multihead_attention::multi_head_attention;
use crate::data::learnable::initialize_weights;
use crate::layers::feedforward_layer::FeedForwardLayer;
use crate::layers::normalization::layer_norm;
use crate::model::encoder::encoding;
use contracts::requires;
use ndarray::{array, Array2, Array3};
use std::ops::Add;
use crate::settings::NUM_HEADS;

#[requires(input.shape().len() == 3, "Input tensor must have 3 dimensions (batch_size, seq_length, d_model)")]
#[requires(encoder_output.shape().len() == 3, "Encoder output tensor must have 3 dimensions (batch_size, seq_length, d_model)")]
#[requires(input.shape() == encoder_output.shape(), "Input tensor and encoder output tensor must have the same shape")]
#[requires(input.shape()[2] == gamma.shape()[1], "Gamma dimensions do not match input feature size")]
#[requires(gamma.shape()[0] == 1, "Gamma must have exactly one row")]
#[requires(input.shape()[2] == beta.shape()[1], "Beta dimensions do not match input feature size")]
#[requires(beta.shape()[0] == 1, "Beta must have exactly one row")]
#[requires(epsilon > 0.0, "Epsilon must be positive and non-zero")]
#[requires(feed_forward_layer.is_initialized(), "Feed-forward layer is not properly initialized")]
#[requires(input.shape()[1] > 0, "Sequence length must be greater than zero")]
pub fn decoding(
    input: Array3<f32>, // Input tensor (usually from the previous decoder layer or initial input)
    encoder_output: Array3<f32>, // Encoder output (for the encoder-decoder attention)
    gamma: Array2<f32>, // Scale parameter for layer norm
    beta: Array2<f32>,  // Shift parameter for layer norm
    epsilon: f32,       // Small constant for numerical stability
    feed_forward_layer: &FeedForwardLayer, // Feed-forward layer instance
) -> Array3<f32> {
    let batch_size = input.shape()[0];
    let seq_length = input.shape()[1];
    let d_model = input.shape()[2];
    let weights = feed_forward_layer.learnables;
    // Self-Attention (Masked Multi-Head Attention in the Decoder)
    let attention_output = multi_head_attention(
        input.clone(),                  // Q
        input.clone(),                  // K
        input.clone(),                  // V
        NUM_HEADS,                              // Number of heads
        true,                           // Masking enabled for decoder
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
    let attention_norm: Array3<f32> = layer_norm(
        &reshaped_attention.to_owned(), // Convert to 2D for layer_norm
        &gamma,
        &beta,
        epsilon,
    )
    .to_shape((batch_size, seq_length, d_model))
    .unwrap()
    .to_owned();

    //  Encoder-Decoder Attention (Cross-Attention Layer)
    let cross_attention_output = multi_head_attention(
        attention_norm.clone(),         // Q from the previous step
        encoder_output.clone(),         // K from the encoder output
        encoder_output.clone(),         // V from the encoder output
        NUM_HEADS,                              // Number of heads
        false,                          // No masking
        weights.query_weights.clone(),  // W_Q
        weights.key_weights.clone(),    // W_K
        weights.value_weights.clone(),  // W_V
        weights.output_projection.clone(), // W_O
    );

    //  Add & Normalize (Residual Connection + Layer Norm)
    let cross_attention_residual = cross_attention_output.add(&attention_norm); // Residual connection
    let reshaped_cross_attention = cross_attention_residual
        .to_shape((batch_size * seq_length, d_model)) // Flatten to 2D
        .unwrap();
    let cross_attention_norm: Array3<f32> =
        layer_norm(&reshaped_cross_attention.to_owned(), &gamma, &beta, epsilon)
            .to_shape((batch_size, seq_length, d_model))
            .unwrap()
            .to_owned();

    //  Feedforward Layer
    let ff_output = feed_forward_layer.forward(cross_attention_norm.clone());

    //  Add & Normalize (Residual Connection + Layer Norm)
    let ff_residual = ff_output.add(&cross_attention_norm); // Residual connection
    let reshaped_ff = ff_residual
        .to_shape((batch_size * seq_length, d_model)) // Flatten to 2D
        .unwrap();
    let ff_norm: Array3<f32> = layer_norm(&reshaped_ff.to_owned(), &gamma, &beta, epsilon)
        .to_shape((batch_size, seq_length, d_model))
        .unwrap()
        .to_owned();

    ff_norm // decoder ouput
}
