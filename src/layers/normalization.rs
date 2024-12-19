use ndarray::{Array2, Axis};

/// Performs layer normalization on a 2D array (batch size x embedding size).
///
/// # Parameters:
/// - `x`: The input 2D array (batch_size x embedding_size).
/// - `gamma`: The learned scaling parameter (embedding_size).
/// - `beta`: The learned bias parameter (embedding_size).
/// - `epsilon`: Small constant for numerical stability.
///
/// # Returns:
/// A 2D array of the same shape as `x` after applying Layer Normalization.
pub fn layer_norm(
    x: &Array2<f32>,
    gamma: &Array2<f32>,
    beta: &Array2<f32>,
    epsilon: f32,
) -> Array2<f32> {
    // Step 1: Calculate mean and variance across the features (axis=1)
    let mean = x.mean_axis(Axis(1)).unwrap();
    let variance = x.var_axis(Axis(1), 0.0);
    println!("Mean: {:?}", mean);
    println!("Variance: {:?}", variance);

    let expanded_mean = mean.insert_axis(Axis(1)); // Expands [6] to [6, 1]
    let expanded_variance = variance.insert_axis(Axis(1)); // Expands [6] to [6, 1]
    println!("EXPMean: {:?}", expanded_mean);
    println!("EXPVariance: {:?}", expanded_variance);

    // Add epsilon to expanded variance
    let normalized = (x - &expanded_mean) / (expanded_variance + epsilon).mapv(f32::sqrt);

    println!("Normalized {}", normalized);
    // Step 2: Normalize the input
    //let normalized = (x - &reshaped_mean) / (reshaped_variance + epsilon).mapv(f32::sqrt);

    // Step 3: Apply scaling (gamma) and shifting (beta)
    normalized * gamma + beta
}
