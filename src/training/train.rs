use crate::data::learnable::LearnableWeights;
use crate::settings::*;
use ndarray::{Array1, Array2, Axis};
use crate::attention::softmax::softmax_matrix;

pub fn compute_gradients(
    logits: &Array2<f32>,
    target_sequence: &Array1<usize>,
    vocab_size: usize,
    model: &LearnableWeights,
) -> LearnableWeights {
    // Compute the softmax probabilities
    let probabilities = softmax_matrix(logits);

    // One-hot encode the target sequence (row = token in sequence, col is vocab index)
    println!("Target seqeuence: {:?}",target_sequence);
    let mut target_one_hot = Array2::<f32>::zeros((target_sequence.len(), vocab_size));
    for (i, &target_idx) in target_sequence.iter().enumerate() {
        target_one_hot[[i, target_idx]] = 1.0;
    }

    // Compute the loss gradient with respect to logits
    let d_logits = &probabilities - &target_one_hot;
    println!("d_logits shape: {:?}", logits.dim());

    // Backpropagate the gradient through the output projection layer
    let d_output_projection = d_logits.dot(&model.output_projection.t());
    // Compute gradients for output_projection_vocab
    let d_output_projection_vocab = d_logits.sum_axis(Axis(0)).insert_axis(Axis(1));

    // Backpropagate through layer normalization
    // Compute gradients for layer normalization parameters
    let d_hidden = d_logits.dot(&model.linear2_weights.t());
    let d_layer_norm_scale = d_hidden.mean_axis(Axis(0)).unwrap().to_vec();
    let d_layer_norm_shift = d_hidden.std_axis(Axis(0), 0.0).to_vec();

    // Compute gradients for the embedding
    let d_embedding = d_hidden.dot(&model.embedding.t());

    // Compute gradients for attention weights (query, key, value)
    let d_query_weights = d_hidden.dot(&model.query_weights.t());
    let d_key_weights = d_hidden.dot(&model.key_weights.t());
    let d_value_weights = d_hidden.dot(&model.value_weights.t());

    // Compute gradients for linear layers
    let d_linear1_weights = d_hidden.dot(&model.linear1_weights.t());
    let d_linear2_weights = model.linear2_weights.t().dot(&d_logits);

    // Compute biases
    let d_bias1 = d_hidden.sum_axis(Axis(0));
    let d_bias2 = d_logits.sum_axis(Axis(0));

    // Package gradients into a LearnableWeights structure
    LearnableWeights {
        embedding: d_embedding,
        query_weights: d_query_weights,
        key_weights: d_key_weights,
        value_weights: d_value_weights,
        output_projection: d_output_projection,
        linear1_weights: d_linear1_weights,
        linear2_weights: d_linear2_weights,
        bias1: d_bias1,
        bias2: d_bias2,
        layer_norm_scale: d_layer_norm_scale, // Converted to Vec<f32>
        layer_norm_shift: d_layer_norm_shift, // Converted to Vec<f32>
        output_projection_vocab: d_output_projection_vocab, // Correct shape as Array2<f32>
    }
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
