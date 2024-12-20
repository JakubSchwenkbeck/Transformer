use ndarray::{array, s, Array1, Array2, Array3};
use std::time::Instant;
use Transformer::math::linear_algebra::{apply_projection, dotproduct, matmul, tensor_product}; // Assuming you're using ndarray for matrices

#[test]
fn test_matmul_valid_input() {
    // Arrange: Define input matrices
    let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 2x3 matrix
    let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]; // 3x2 matrix

    // Expected result after matrix multiplication
    let expected = array![[58.0, 64.0], [139.0, 154.0]]; // 2x2 matrix

    // Act: Perform the multiplication
    let result = matmul(&a, &b); // Assuming matmul returns Result

    // Assert: Check the result is Ok and matches the expected result
    match result {
        Ok(res) => assert_eq!(res, expected),
        Err(e) => panic!("Matrix multiplication failed: {}", e),
    }
}

#[test]
fn test_dotproduct() {
    let a: Array1<f32> = array![1.0, 2.0, 3.0];
    let b: Array1<f32> = array![4.0, 5.0, 5.0];

    let expected = (4 + 10 + 15) as f32;
    let result = dotproduct(&a, &b);

    assert_eq!(result, expected);
}
#[test]
fn test_floats_dotproduct() {
    let a: Array1<f32> = array![2.9, 7.68, 2.333];
    let b: Array1<f32> = array![0.74, 1.2, 5.111];

    let expected = (2.9 * 0.74 + 7.68 * 1.2 + 2.333 * 5.111) as f32;
    let result = dotproduct(&a, &b);

    assert_eq!(result, expected);
}
#[test]
fn test_empty_dotproduct() {
    let a: Array1<f32> = array![];
    let b: Array1<f32> = array![];

    let expected = 0.0;
    let result = dotproduct(&a, &b);

    assert_eq!(result, expected);
}

#[test]
#[should_panic]
fn test_mismatch_dotproduct() {
    let a: Array1<f32> = array![2.9, 7.68, 2.333, 1.0];
    let b: Array1<f32> = array![0.74, 1.2, 5.111];

    // This call is expected to panic due to a mismatch in dimensions.
    let _result = dotproduct(&a, &b);
}

#[test]
fn test_tensor_product_single_batch() {
    let a = array![
        [[1.0, 2.0], [3.0, 4.0]] // Shape: (1, 2, 2)
    ];
    let b = array![
        [[5.0, 6.0], [7.0, 8.0]] // Shape: (1, 2, 2)
    ];
    let expected = array![
        [[19.0, 22.0], [43.0, 50.0]] // Shape: (1, 2, 2), result of 2x2 matrix multiplication
    ];

    let result = tensor_product(&a, &b);
    assert_eq!(result, expected);
}

#[test]
fn test_tensor_product_multi_batch() {
    let a = array![
        [[1.0, 0.0], [0.0, 1.0]], // Batch 1: identity matrix
        [[2.0, 3.0], [4.0, 5.0]]  // Batch 2: General 2x2 matrix
    ]; // Shape: (2, 2, 2)

    let b = array![
        [[1.0, 2.0], [3.0, 4.0]], // Batch 1: Random 2x2 matrix
        [[0.0, 1.0], [1.0, 0.0]]  // Batch 2: Another 2x2 matrix
    ]; // Shape: (2, 2, 2)

    let expected = array![
        [[1.0, 2.0], [3.0, 4.0]], // Batch 1: Multiplication with Identity gives same matrix
        [[3.0, 2.0], [5.0, 4.0]]  // Batch 2: Custom result
    ];

    let result = tensor_product(&a, &b);
    assert_eq!(result, expected);
}
#[test]
fn test_tensor_product_larger_matrices() {
    let a = array![
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], // Shape: (1, 2, 3)
    ];

    let b = array![
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]] // Shape: (1, 3, 2)
    ];

    let expected = array![
        [[58.0, 64.0], [139.0, 154.0]] // Shape: (1, 2, 2)
    ];

    let result = tensor_product(&a, &b);
    assert_eq!(result, expected);
}

#[test]
fn test_tensor_product_performance() {
    let batch_size = 32;
    let seq_length = 128;
    let embedding_size = 512;

    let a = Array3::<f32>::ones((batch_size, seq_length, embedding_size));
    let b = Array3::<f32>::ones((batch_size, embedding_size, seq_length));

    // Timing the tensor product operation for larger tensors
    let start = Instant::now();
    let _ = tensor_product(&a, &b);
    let duration = start.elapsed();

    // Ensure the operation completes within reasonable time limits
    assert!(duration.as_secs() < 1, "Tensor product took too long");
}
#[test]
fn test_tensor_product_shapes() {
    // Given tensors with matching dimensions
    let a = array![
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  // Shape: (1, 2, 3)
        ];

    let b = array![
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]  // Shape: (1, 3, 2)
        ];

    let result = tensor_product(&a, &b);

    // Assert shape consistency: result should be (1, 2, 2)
    assert_eq!(result.shape(), [1, 2, 2]);
}

#[test]
#[should_panic]
fn test_mismatch_tensor_product() {
    let a = array![
        [[1.0, 2.0], [3.0, 4.0]] // Shape: (1, 2, 2)
    ];
    let b = array![
        [[5.0, 6.0], [7.0, 8.0],[2.0,1.0]] // Shape: (1, 2, 2)
    ];
    // This call is expected to panic due to a mismatch in dimensions.
    let _result = tensor_product(&a, &b);
}
#[test]
fn test_projection() {
    // Step 1: Define the input matrix X (B, T, D)
    let x = Array3::from_shape_vec(
        (2, 3, 4), // B = 2, T = 3, D = 4
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ],
    )
    .unwrap();

    // Step 2: Define the projection matrices W_Q, W_K, W_V (D, d_k)
    let w_q = Array2::from_shape_vec(
        (4, 2), // D = 4, d_k = 2
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();

    let w_k = Array2::from_shape_vec(
        (4, 2), // D = 4, d_k = 2
        vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();

    let w_v = Array2::from_shape_vec(
        (4, 2), // D = 4, d_v = 2
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    )
    .unwrap();

    // Step 3: Perform the projections Q, K, V
    let q = apply_projection(&x, &w_q); // Shape: [B, T, d_k]
    let k = apply_projection(&x, &w_k); // Shape: [B, T, d_k]
    let v = apply_projection(&x, &w_v); // Shape: [B, T, d_v]

    // Step 4: Define the expected projections for Q, K, V
    let expected_q_1 = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0]).unwrap();

    let expected_k_1 = Array2::from_shape_vec((3, 2), vec![2.0, 1.0, 6.0, 5.0, 10.0, 9.0]).unwrap();

    let expected_v_1 =
        Array2::from_shape_vec((3, 2), vec![3.0, 2.0, 7.0, 6.0, 11.0, 10.0]).unwrap();

    let expected_q_2 =
        Array2::from_shape_vec((3, 2), vec![13.0, 14.0, 17.0, 18.0, 21.0, 22.0]).unwrap();

    let expected_k_2 =
        Array2::from_shape_vec((3, 2), vec![14.0, 13.0, 18.0, 17.0, 22.0, 21.0]).unwrap();

    let expected_v_2 =
        Array2::from_shape_vec((3, 2), vec![15.0, 14.0, 19.0, 18.0, 23.0, 22.0]).unwrap();

    // Step 5: Assert that the computed values match the expected ones
    assert_eq!(q.slice(s![0, .., ..]), expected_q_1);
    assert_eq!(q.slice(s![1, .., ..]), expected_q_2);

    assert_eq!(k.slice(s![0, .., ..]), expected_k_1);
    assert_eq!(k.slice(s![1, .., ..]), expected_k_2);

    assert_eq!(v.slice(s![0, .., ..]), expected_v_1);
    assert_eq!(v.slice(s![1, .., ..]), expected_v_2);
}
