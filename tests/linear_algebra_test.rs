use ndarray::{array, Array1};
use Transformer::math::linear_algebra::{dotproduct, matmul, tensor_product}; // Assuming you're using ndarray for matrices

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
fn test_matmul_invalid_input() {
    // Arrange: Define input matrices with mismatched dimensions
    let a = array![[1.0, 2.0], [3.0, 4.0]]; // 2x2 matrix
    let b = array![[5.0, 6.0]]; // 1x2 matrix (mismatched dimensions)

    // Act: Perform the multiplication, expecting an error
    let result = matmul(&a, &b);

    // Assert: Ensure the result is an error due to incompatible dimensions
    assert_eq!(
        result,
        Err("Matrix dimensions are incompatible for multiplication.")
    );
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
