use Transformer::utils::linear_algebra::matmul;
use ndarray::array; // Assuming you're using ndarray for matrices

#[test]
fn test_matmul_valid_input() {
    // Arrange: Define input matrices
    let a = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]; // 2x3 matrix
    let b = array![
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0]
        ]; // 3x2 matrix

    // Expected result after matrix multiplication
    let expected = array![
            [58.0, 64.0],
            [139.0, 154.0]
        ]; // 2x2 matrix

    // Act: Perform the multiplication
    let result = matmul(&a, &b); // Assuming matmul returns Result

    // Assert: Check the result matches the expectation
    assert_eq!(result, expected);
}

#[test]
#[should_panic] // If you expect the function to panic on invalid input
fn test_matmul_invalid_input() {
    let a = array![
            [1.0, 2.0],
            [3.0, 4.0]
        ]; // 2x2 matrix
    let b = array![
            [5.0, 6.0]
        ]; // 1x2 matrix (mismatched dimensions)

    matmul(&a, &b); // Should panic or return an error
}


