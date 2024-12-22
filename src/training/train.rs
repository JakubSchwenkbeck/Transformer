use crate::data::learnable::LearnableWeights;
use crate::math::linear_algebra::flatten_3d_array;
use crate::settings::*;
use ndarray::{Array2, Array3};

/// Compute gradients for the transformer model's learnable weights.
pub fn compute_gradients(
    weights: &mut LearnableWeights,
    inputs: &Array3<f32>,
    targets: &Array2<f32>,
    predictions: &Array2<f32>,
    vocabsize: usize,
) -> LearnableWeights {
    let mut gradients = LearnableWeights::new(
        OUTPUT_SIZE,
        HIDDEN_SIZE,
        vocabsize, // Ensure the vocab size is correct
        D_MODEL,
        D_K,
        FFN_DIM,
    );

    // Compute the loss and its derivative
    let loss = predictions - targets;
    let d_loss = &loss * 2.0 / (BATCH_SIZE as f32);

    // Compute gradients for the output projection weights
    gradients.output_projection_vocab = predictions.t().dot(&d_loss);

    // Flattened inputs for further computations
    let flattened_inputs = flatten_3d_array(inputs.clone());

    // Compute gradients for the feedforward network weights
    let d_linear2 = d_loss.dot(&weights.linear2_weights.t());
    gradients.linear2_weights = flattened_inputs.t().dot(&d_linear2);
    gradients.bias2 = d_linear2.sum_axis(ndarray::Axis(0));

    let d_linear1 = d_linear2.dot(&weights.linear1_weights.t());
    gradients.linear1_weights = flattened_inputs.t().dot(&d_linear1);
    gradients.bias1 = d_linear1.sum_axis(ndarray::Axis(0));

    // Compute gradients for the attention mechanism weights
    let d_attention_output = d_loss.dot(&weights.output_projection.t());
    gradients.output_projection = flattened_inputs.t().dot(&d_attention_output);
    let d_value = d_attention_output.dot(&weights.value_weights.t());
    gradients.value_weights = flattened_inputs.t().dot(&d_value);
    let d_key = d_attention_output.dot(&weights.key_weights.t());
    gradients.key_weights = flattened_inputs.t().dot(&d_key);
    let d_query = d_attention_output.dot(&weights.query_weights.t());
    gradients.query_weights = flattened_inputs.t().dot(&d_query);

    // Compute gradients for the embedding layer
    gradients.embedding = inputs.mean_axis(ndarray::Axis(0)).unwrap(); // Ensure shape consistency with model.embedding

    // Compute gradients for layer normalization parameters (scale and shift)
    gradients.layer_norm_scale = d_linear1.mean_axis(ndarray::Axis(0)).unwrap().to_vec();
    gradients.layer_norm_shift = d_linear1.sum_axis(ndarray::Axis(0)).to_vec();

    gradients
}

pub fn update_weights(
    model: &mut LearnableWeights,
    gradients: &LearnableWeights,
    learning_rate: f32,
) {
    println!(
        "EMBEDDING OLD :{:?}, EMBEDDING NEW: {:?}",
        model.embedding.shape(),
        gradients.embedding.shape()
    );
    // Ensure the gradients and model weights have compatible shapes (reshape if necessary)
    model.embedding = &model.embedding - &(&gradients.embedding * learning_rate);
    model.query_weights = &model.query_weights - &(&gradients.query_weights * learning_rate);
    model.key_weights = &model.key_weights - &(&gradients.key_weights * learning_rate);
    model.value_weights = &model.value_weights - &(&gradients.value_weights * learning_rate);
    model.output_projection =
        &model.output_projection - &(&gradients.output_projection * learning_rate);
    model.linear1_weights = &model.linear1_weights - &(&gradients.linear1_weights * learning_rate);
    model.linear2_weights = &model.linear2_weights - &(&gradients.linear2_weights * learning_rate);

    // Handle potential shape mismatches with bias updates
    model.bias1 = &model.bias1 - &(&gradients.bias1 * learning_rate);
    model.bias2 = &model.bias2 - &(&gradients.bias2 * learning_rate);

    // Handle Layer Norm scales and shifts (ensure correct dimensions)
    model
        .layer_norm_scale
        .iter_mut()
        .zip(gradients.layer_norm_scale.iter())
        .for_each(|(a, g)| *a -= g * learning_rate);
    model
        .layer_norm_shift
        .iter_mut()
        .zip(gradients.layer_norm_shift.iter())
        .for_each(|(a, g)| *a -= g * learning_rate);

    // Update output projection vocabulary weights (handle shapes)
    model.output_projection_vocab =
        &model.output_projection_vocab - &(&gradients.output_projection_vocab * learning_rate);
}
