#![allow(unused_variables)]
use crate::attention::softmax::softmax_3d;
use crate::math::linear_algebra::matmul;
use ndarray::{array, Array3, Axis, ShapeError};

pub fn scaled_dot_product_attention(
    q: Array3<f32>, // Query
    k: Array3<f32>, // Key
    v: Array3<f32>, // Value
) -> Array3<f32> {
    let scores = scaled_dot_product(q, k, v, None);
    let sm_scores = softmax_3d(&scores);
    sm_scores
}

pub fn scaled_dot_product(
    q: Array3<f32>,             // Shape: (B, L_Q, d_k)
    k: Array3<f32>,             // Shape: (B, L_K, d_k)
    v: Array3<f32>,             // Shape: (B, L_K, d_v)
    _mask: Option<Array3<f32>>, // Shape: (B, L_Q, L_K)
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

pub fn test_attention_matrices() {
    // Query matrix Q: Shape (2, 3, 4) -> Batch size 2, sequence length 3, d_k 4
    let q: Array3<f32> = array![
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ],
        [
            [13.0, 14.0, 15.0, 16.0],
            [17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0]
        ]
    ];

    // Key matrix K: Shape (2, 3, 4) -> Batch size 2, sequence length 3, d_k 4
    let k: Array3<f32> = array![
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0]
        ],
        [
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0]
        ]
    ];

    // Value matrix V: Shape (2, 3, 5) -> Batch size 2, sequence length 3, d_v 5
    let v: Array3<f32> = array![
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0]
        ],
        [
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
            [26.0, 27.0, 28.0, 29.0, 30.0]
        ]
    ];

    let res = scaled_dot_product(q.clone(), k.clone(), v.clone(), None);
    println!(
        "The Query Matrix : \n {:?} \n with shape {:?} \n ",
        q,
        q.shape()
    );
    println!(
        "The Key Matrix : \n {:?} \n with shape {:?} \n ",
        k,
        k.shape()
    );
    println!(
        "The Value Matrix : \n {:?} \n with shape {:?} \n ",
        v,
        v.shape()
    );
    println!(
        "The scaled Query and Key Product for Attention : \n {:?} \n with shape {:?} ",
        res,
        res.shape()
    );
}
