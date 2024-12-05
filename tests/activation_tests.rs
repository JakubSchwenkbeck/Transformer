use ndarray::array;
use Transformer::activation::activation_functions::{gelu, relu};

#[test]
pub fn test_relu() {
    let input = array![[1.0, -1.0], [0.5, -0.5]];
    let result = relu(&input);

    assert_eq!(result, array![[1.0, 0.0], [0.5, 0.0]]);
}
#[test]
pub fn one_test_gelu() {
    let binding = gelu(&array![[1.0]]);
    let result = binding.get((0, 0)).unwrap();

    // Define an acceptable tolerance (epsilon)
    let epsilon = 1e-6; // allow up to 1e-6 difference

    // Assert that the result is within the tolerance of the expected value
    assert!(
        (result - 0.841192f32).abs() < epsilon,
        "Test failed: result was {}",
        result
    );
}

#[test]
fn test_gelu_2x2() {
    // Create a 2x2 matrix
    let input = array![[1.0, 2.0], [3.0, 4.0]];

    // Apply GELU element-wise
    let result = gelu(&input);

    // Define an acceptable tolerance (epsilon)
    let epsilon = 1e-6;

    // Expected results for GELU(1.0), GELU(2.0), GELU(3.0), GELU(4.0)
    let expected = array![[0.841192, 1.9545977], [2.9963627, 3.99993]];

    // Assert that each element in the result is within tolerance of the expected value
    for (r, expected_row) in result.outer_iter().zip(expected.outer_iter()) {
        for (res, exp) in r.iter().zip(expected_row.iter()) {
            assert!(
                (res - exp).abs() < epsilon,
                "Test failed: result was {}, expected {}",
                res,
                exp
            );
        }
    }
}
