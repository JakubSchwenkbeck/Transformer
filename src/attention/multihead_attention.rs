#![allow(warnings)] // work in progress
use ndarray::{Array3, Array4};

pub fn multi_head_attention(
    q: Array3<f32>,            // Query: Shape (B, L_Q, d_model)
    k: Array3<f32>,            // Key: Shape (B, L_K, d_model)
    v: Array3<f32>,            // Value: Shape (B, L_K, d_model)
    num_heads: usize,          // Number of attention heads
    mask: Option<Array3<f32>>, // Mask for attention (optional)
) -> Array3<f32> {
    // 1. Project the Q, K, V tensors into different subspaces for each head.
    // 2. Apply scaled dot-product attention on each head in parallel.
    // 3. Concatenate the outputs of all heads.
    // 4. Apply a final linear transformation to the concatenated result.

    // Step 1: Linearly project Q, K, V for each head
    let (q_heads, k_heads, v_heads) = split_into_heads(&q, &k, &v, num_heads);

    // Step 2: Compute the attention for each head
    let attention_outputs = compute_attention_for_heads(q_heads, k_heads, v_heads, mask);

    // Step 3: Concatenate the output of all attention heads
    let concatenated_output = concat_heads(attention_outputs, num_heads);

    // Step 4: Apply the final linear transformation (W_O) to the concatenated output
    let output = linear_transformation(concatenated_output);

    output
}

pub fn linear_transformation(concatenated_output: Array3<f32>) -> Array3<f32> {
    todo!()
}

pub fn concat_heads(
    attention_outputs: Vec<Array3<f32>>, // List of attention outputs for each head
    num_heads: usize,                    // Number of heads
) -> Array3<f32> {
    todo!()
}

pub fn compute_attention_for_heads(
    q_heads: Array4<f32>,      // Shape: (B, num_heads, L_Q, d_k)
    k_heads: Array4<f32>,      // Shape: (B, num_heads, L_K, d_k)
    v_heads: Array4<f32>,      // Shape: (B, num_heads, L_K, d_k)
    mask: Option<Array3<f32>>, // Mask for attention (optional)
) -> Vec<Array3<f32>> {
    todo!()
}

/// Splits Q, K, and V into multiple heads.
///
/// # Arguments
/// * `q` - Query matrix with shape (batch_size, seq_len, feature_dim)
/// * `k` - Key matrix with shape (batch_size, seq_len, feature_dim)
/// * `v` - Value matrix with shape (batch_size, seq_len, feature_dim)
/// * `num_heads` - The number of attention heads
///
/// # Returns
/// Returns a tuple of 4D arrays: (Q_heads, K_heads, V_heads), where each has shape
/// (batch_size, num_heads, seq_len, head_dim)
pub fn split_into_heads(
    q: &Array3<f32>,
    k: &Array3<f32>,
    v: &Array3<f32>,
    num_heads: usize,
) -> (Array4<f32>, Array4<f32>, Array4<f32>) {
    let batch_size = q.shape()[0];
    let seq_len = q.shape()[1];
    let feature_dim = q.shape()[2];

    // Compute the head dimension (feature_dim / num_heads)
    let head_dim = feature_dim / num_heads;

    // Reshape Q, K, and V into a 4D tensor with shape (batch_size, num_heads, seq_len, head_dim)
    let q_heads = q
        .view()
        .to_shape((batch_size, seq_len, num_heads, head_dim))
        .unwrap()
        .permuted_axes([0, 2, 1, 3]) // Swap axes to get shape (batch_size, num_heads, seq_len, head_dim)
        .to_owned();

    let k_heads = k
        .view()
        .to_shape((batch_size, seq_len, num_heads, head_dim))
        .unwrap()
        .permuted_axes([0, 2, 1, 3]) // Swap axes to get shape (batch_size, num_heads, seq_len, head_dim)
        .to_owned();

    let v_heads = v
        .view()
        .to_shape((batch_size, seq_len, num_heads, head_dim))
        .unwrap()
        .permuted_axes([0, 2, 1, 3]) // Swap axes to get shape (batch_size, num_heads, seq_len, head_dim)
        .to_owned();

    (q_heads, k_heads, v_heads)
}
