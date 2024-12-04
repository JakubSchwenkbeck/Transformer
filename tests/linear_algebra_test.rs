use Transformer::math::linear_algebra::matmul;
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

    // Assert: Check the result is Ok and matches the expected result
    match result {
        Ok(res) => assert_eq!(res, expected),
        Err(e) => panic!("Matrix multiplication failed: {}", e),
    }
}

#[test]
fn test_matmul_invalid_input() {
    // Arrange: Define input matrices with mismatched dimensions
    let a = array![
            [1.0, 2.0],
            [3.0, 4.0]
        ]; // 2x2 matrix
    let b = array![
            [5.0, 6.0]
        ]; // 1x2 matrix (mismatched dimensions)

    // Act: Perform the multiplication, expecting an error
    let result = matmul(&a, &b);

    // Assert: Ensure the result is an error due to incompatible dimensions
    assert_eq!(result, Err("Matrix dimensions are incompatible for multiplication."));
}
