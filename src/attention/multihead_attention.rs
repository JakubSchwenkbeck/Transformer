#![allow(warnings)]
use crate::attention::scaled_dot_attention::scaled_dot_product_attention;
use crate::math::linear_algebra::apply_projection;
use ndarray::{s, Array2, Array3, Array4};

pub fn multi_head_attention(
    q: Array3<f32>,   // Query: Shape (B, L_Q, d_model)
    k: Array3<f32>,   // Key: Shape (B, L_K, d_model)
    v: Array3<f32>,   // Value: Shape (B, L_K, d_model)
    num_heads: usize, // Number of attention heads
    mask: bool,       // Mask for attention (optional)
    w_q: Array2<f32>, // Learned projection matrix for Q: Shape (d_model, d_model)
    w_k: Array2<f32>, // Learned projection matrix for K: Shape (d_model, d_model)
    w_v: Array2<f32>, // Learned projection matrix for V: Shape (d_model, d_model)
    w_o: Array2<f32>, // Final linear projection: Shape (d_model, d_model)
) -> Array3<f32> {
    // Linearly project Q, K, V using w_q, w_k, w_v (common projection for all heads)
    let q_projected = apply_projection(&q, &w_q); // Shape: (B, L_Q, d_model)
    let k_projected = apply_projection(&k, &w_k); // Shape: (B, L_K, d_model)
    let v_projected = apply_projection(&v, &w_v); // Shape: (B, L_K, d_model)

    // Split into multiple heads
    let (q_heads, k_heads, v_heads) =
        split_into_heads(&q_projected, &k_projected, &v_projected, num_heads);

    // Compute attention for each head
    let attention_outputs = compute_attention_for_heads(q_heads, k_heads, v_heads, mask);

    // Concatenate the outputs of all heads
    let concatenated_output = concat_heads(attention_outputs, num_heads);

    // Apply the final linear transformation (w_o) to the concatenated output
    apply_projection(&concatenated_output, &w_o)
}

/// Compute attention for each head in the multi-head attention.
pub fn compute_attention_for_heads(
    q_heads: Array4<f32>, // Shape: (B, num_heads, L_Q, head_dim)
    k_heads: Array4<f32>, // Shape: (B, num_heads, L_K, head_dim)
    v_heads: Array4<f32>, // Shape: (B, num_heads, L_K, head_dim)
    mask: bool,           // Mask for attention (optional)
) -> Vec<Array3<f32>> {
    let num_heads = q_heads.shape()[1];

    // Initialize a Vec to store the attention results for each head
    let mut attention_results = Vec::with_capacity(num_heads);

    // Iterate over each head and apply scaled dot-product attention
    for head_idx in 0..num_heads {
        // Slice Q, K, V for the current head
        let q_head = q_heads.slice(s![.., head_idx, .., ..]).to_owned(); // Shape: (B, L_Q, head_dim)
        let k_head = k_heads.slice(s![.., head_idx, .., ..]).to_owned(); // Shape: (B, L_K, head_dim)
        let v_head = v_heads.slice(s![.., head_idx, .., ..]).to_owned(); // Shape: (B, L_K, head_dim)

        // Compute attention for the current head
        let attention_output = scaled_dot_product_attention(q_head, k_head, v_head, mask);

        // Store the result in the output vector
        attention_results.push(attention_output);
    }

    attention_results
}

/// Splits Q, K, and V into multiple heads.
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

    assert!(
        feature_dim % num_heads == 0,
        "feature_dim must be divisible by num_heads"
    );

    // Reshape Q, K, and V into a 4D tensor with shape (batch_size, num_heads, seq_len, head_dim)
    let q_heads = q
        .view()
        .to_shape((batch_size, seq_len, num_heads, head_dim))
        .unwrap()
        .permuted_axes([0, 2, 1, 3])
        .to_owned();

    let k_heads = k
        .view()
        .to_shape((batch_size, seq_len, num_heads, head_dim))
        .unwrap()
        .permuted_axes([0, 2, 1, 3])
        .to_owned();

    let v_heads = v
        .view()
        .to_shape((batch_size, seq_len, num_heads, head_dim))
        .unwrap()
        .permuted_axes([0, 2, 1, 3])
        .to_owned();

    (q_heads, k_heads, v_heads)
}

/// Concatenates the attention outputs from all heads.
pub fn concat_heads(
    attention_outputs: Vec<Array3<f32>>, // List of attention outputs for each head
    num_heads: usize,                    // Number of heads
) -> Array3<f32> {
    let batch_size = attention_outputs[0].shape()[0];
    let seq_length = attention_outputs[0].shape()[1];
    let head_dim = attention_outputs[0].shape()[2]; // Feature size per head

    // Concatenate the outputs from all heads
    let mut concatenated = Array3::<f32>::zeros((batch_size, seq_length, num_heads * head_dim));

    for i in 0..num_heads {
        let head_output = &attention_outputs[i];
        concatenated
            .slice_mut(s![.., .., i * head_dim..(i + 1) * head_dim])
            .assign(head_output);
    }

    concatenated
}
