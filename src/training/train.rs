use crate::data::learnable::LearnableWeights;
use crate::math::linear_algebra::flatten_3d_array;
use crate::settings::*;
use ndarray::{Array1, Array2, Array3};
use crate::training::loss_function::cross_entropy_loss;

/// Compute gradients for the transformer model's learnable weights.
pub fn compute_gradients(
    weights: &mut LearnableWeights,
    inputs: &Array3<f32>,
    targets: &Array2<f32>,
    predictions: &Array2<f32>,
) -> LearnableWeights {
    let mut gradients = LearnableWeights::new(
        D_MODEL, // output_size = D_MODEL
        FFN_DIM, // hidden_size = FFN_DIM
        D_MODEL, // vocab_size
        D_MODEL, // embedding_dim = D_MODEL
        D_K,     // attention_dim
        FFN_DIM, // ffn_dim
    );

    // Compute the loss and its derivative

   // println!("PRED : {:?}",predictions);
   // println!("TARGET : {:?}",targets);


    let loss = predictions - targets;
    let d_loss = 0.0001 * &loss * 2.0 / (BATCH_SIZE as f32);
    // Debugging: Print loss and its derivative

  //  println!("Derivative of loss (d_loss): {:?}", d_loss);

    // Compute gradients for the output projection weights
    gradients.output_projection_vocab = predictions.t().dot(&d_loss);

    // Debugging: Print the output projection gradient
  //  println!("Gradient for output_projection_vocab: {:?}", gradients.output_projection_vocab);

    // Flattened inputs for further computations
    let flattened_inputs = flatten_3d_array(inputs.clone()); // Flatten [1, 88, 88] -> [88, 88]

    // Debugging: Print flattened inputs
   // println!("Flattened inputs: {:?}", flattened_inputs);

    // Compute gradients for the feedforward network weights
    // d_linear2 corresponds to the gradient w.r.t. the second linear layer
    let d_linear2 = d_loss.dot(&weights.linear2_weights.t()); // Shape: [88, 128]
    gradients.linear2_weights = flattened_inputs.t().dot(&d_linear2); // Shape: [88, 128]
    gradients.bias2 = d_linear2.sum_axis(ndarray::Axis(0)); // Sum across sequences to get bias gradient

    // Debugging: Print the gradient for the second linear layer
   // println!("Gradient for linear2_weights: {:?}", gradients.linear2_weights);
   // println!("Bias2 gradient: {:?}", gradients.bias2);

    // d_linear1 corresponds to the gradient w.r.t. the first linear layer
    let d_linear1 = d_linear2.dot(&weights.linear1_weights.t()); // Shape: [88, 88]

    gradients.linear1_weights = flattened_inputs.t().dot(&d_linear1); // Shape: [88, 128] (for linear1)
    gradients.bias1 = d_linear1.sum_axis(ndarray::Axis(0)); // Sum across sequences to get bias gradient

    // Debugging: Print the gradient for the first linear layer
   // println!("Gradient for linear1_weights: {:?}", gradients.linear1_weights);
   // println!("Bias1 gradient: {:?}", gradients.bias1);

    // Compute gradients for the attention mechanism weights
    let d_attention_output = d_loss.dot(&weights.output_projection.t()); // Shape: [88, 88]
    gradients.output_projection = flattened_inputs.t().dot(&d_attention_output); // Shape: [88, 88]

    // Debugging: Print attention output gradients
   // println!("Gradient for output_projection: {:?}", gradients.output_projection);

    let d_value = d_attention_output.dot(&weights.value_weights.t()); // Shape: [88, 88]
    gradients.value_weights = flattened_inputs.t().dot(&d_value); // Shape: [88, 88]

    // Debugging: Print value weights gradient
   // println!("Gradient for value_weights: {:?}", gradients.value_weights);

    let d_key = d_attention_output.dot(&weights.key_weights.t()); // Shape: [88, 88]
    gradients.key_weights = flattened_inputs.t().dot(&d_key); // Shape: [88, 88]

    // Debugging: Print key weights gradient
   // println!("Gradient for key_weights: {:?}", gradients.key_weights);

    let d_query = d_attention_output.dot(&weights.query_weights.t()); // Shape: [88, 88]
    gradients.query_weights = flattened_inputs.t().dot(&d_query); // Shape: [88, 88]

    // Debugging: Print query weights gradient
   // println!("Gradient for query_weights: {:?}", gradients.query_weights);

    // Compute gradients for the embedding layer
    gradients.embedding = inputs.mean_axis(ndarray::Axis(0)).unwrap(); // Ensure shape consistency with model.embedding

    // Debugging: Print embedding gradients
   // println!("Embedding gradients: {:?}", gradients.embedding);

    // Compute gradients for layer normalization parameters (scale and shift)
    gradients.layer_norm_scale = d_linear1.mean_axis(ndarray::Axis(0)).unwrap().to_vec();
    gradients.layer_norm_shift = d_linear1.sum_axis(ndarray::Axis(0)).to_vec();

    // Debugging: Print layer norm gradients
  //  println!("Layer norm scale gradient: {:?}", gradients.layer_norm_scale);
   // println!("Layer norm shift gradient: {:?}", gradients.layer_norm_shift);

    gradients
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
