#![allow(unused_imports)]
use contracts::{ensures, requires};
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
#[requires(x.shape().len() == 2, "Input array must be 2-dimensional")]
#[requires(gamma.shape().len() == 2 && gamma.shape()[0] == 1, "Gamma must be a 2-dimensional array with a single row")]
#[requires(beta.shape().len() == 2 && beta.shape()[0] == 1, "Beta must be a 2-dimensional array with a single row")]
#[requires(epsilon > 0.0, "Epsilon must be positive and non-zero")]
#[ensures(ret.shape() == x.shape(), "The resulting array must have the same shape as the input array")]
#[ensures(ret.iter().all(|&x| x.is_finite()), "All elements in the resulting array must be finite")]
pub fn layer_norm(
    x: &Array2<f32>,
    gamma: &Array2<f32>,
    beta: &Array2<f32>,
    epsilon: f32,
) -> Array2<f32> {
    // Calculate mean and variance across the features (axis=1)
    let mean = x.mean_axis(Axis(1)).unwrap();
    let variance = x.var_axis(Axis(1), 0.0);

    let expanded_mean = mean.insert_axis(Axis(1)); // Expands [6] to [6, 1]
    let expanded_variance = variance.insert_axis(Axis(1)); // Expands [6] to [6, 1]

    // Add epsilon to expanded variance
    let normalized = (x - &expanded_mean) / (expanded_variance + epsilon).mapv(f32::sqrt);

    normalized * gamma + beta
}
