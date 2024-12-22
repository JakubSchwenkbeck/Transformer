use crate::data::learnable::LearnableWeights;
use crate::settings::*;
use ndarray::{Array1, Array2};

pub fn compute_gradients(
    _logits: &Array2<f32>,
    _target_sequence: &Array1<usize>,
    vocab_size: usize,
    _model: &LearnableWeights,
) -> LearnableWeights {
    // TODO: compute gradients for all the learnable weights in the model

    // FTEMPO!!! :OR NOW ONLY COPY WEIGHTS...
    LearnableWeights::new(
        OUTPUT_SIZE,
        HIDDEN_SIZE,
        vocab_size,
        EMBEDDING_SIZE,
        EMBEDDING_SIZE,
        HIDDEN_SIZE,
    )
}

pub fn update_weights(
    model: &mut LearnableWeights,
    gradients: &LearnableWeights,
    learning_rate: f32,
) {
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
