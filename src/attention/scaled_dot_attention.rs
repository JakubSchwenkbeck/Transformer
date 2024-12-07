#![allow(unused_variables)]
use crate::attention::softmax::softmax_3d;
use crate::math::linear_algebra::{matmul, tensor_product};
use ndarray::{Array3, Axis, ShapeError};

pub fn scaled_dot_product_attention(
    q: Array3<f32>, // Query
    k: Array3<f32>, // Key
    v: Array3<f32>, // Value
    mask: bool,
) -> Array3<f32> {
    let scores = scaled_dot_product(q, k, v.clone(), mask);
    let sm_scores = softmax_3d(&scores);
    tensor_product(&sm_scores, &v)
}

pub fn scaled_dot_product(
    q: Array3<f32>, // Shape: (B, L_Q, d_k)
    k: Array3<f32>, // Shape: (B, L_K, d_k)
    v: Array3<f32>, // Shape: (B, L_K, d_v)
    mask: bool,
) -> Array3<f32> {
    let batch_size = q.shape()[0];
    assert_eq!(q.shape()[0], k.shape()[0], "Batch Size mismatch");
    let d_k: f32 = q.shape()[2] as f32;
    let d_v = v.shape()[2];
    let L_Q = q.shape()[1]; // (L_Q and L_K might be equal all along?)
    let L_K = k.shape()[1];
    let mut scores = query_key_product(q, k).unwrap();

    // Scale the scores by sqrt(d_k)
    scores /= d_k.sqrt();
    if mask {
        let mask = Array3::from_shape_fn((batch_size, L_Q, L_K), |(b, i, j)| {
            if i >= j {
                0.0
            } else {
                f32::NEG_INFINITY
            }
        });
        // Ensure the mask has the shape (B, L_Q, L_K)
        assert_eq!(mask.shape(), &[batch_size, L_Q, L_K]);

        // Add the mask to the scores: apply a large negative number to masked positions
        // This ensures that after softmax, these positions will have zero attention.
        for b in 0..batch_size {
            for i in 0..L_Q {
                for j in 0..L_K {
                    if mask[(b, i, j)] == 0.0 {
                        // Applying a large negative value to masked positions
                        scores[(b, i, j)] = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }
    scores
}
pub fn query_key_product(
    q: Array3<f32>, // Shape: (B, L_Q, d_k)
    k: Array3<f32>, // Shape: (B, L_K, d_k)
) -> Result<Array3<f32>, ShapeError> {
    // Ensure the last dimensions of q and k are compatible for dot product
    assert_eq!(
        q.shape()[2],
        k.shape()[2],
        "Query and Key must have the same depth for dot product"
    );

    // Perform the batch matrix multiplication q * k^T
    let mut scores = Array3::<f32>::zeros((q.shape()[0], q.shape()[1], k.shape()[1])); // (B, L_Q, L_K)

    for b in 0..q.shape()[0] {
        let q_batch = q.index_axis(Axis(0), b); // Shape: (L_Q, d_k)
        let k_batch = k.index_axis(Axis(0), b); // Shape: (L_K, d_k)

        // Matrix multiplication
        scores
            .index_axis_mut(Axis(0), b)
            //.assign(&(q_batch.dot(&k_batch.t()))); // using my own implementation:
            .assign(&(matmul(&q_batch.to_owned(), &k_batch.t().to_owned()).unwrap()));
    }

    Ok(scores)
}
