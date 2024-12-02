
use ndarray::Array2;
use ndarray::linalg::general_mat_mul;


/// Performs matrix multiplication between two 2D arrays.
///
/// # Parameters
/// - `a`: A reference to the first matrix (of type `Array2<f32>`).
/// - `b`: A reference to the second matrix (of type `Array2<f32>`).
///
/// # Returns
/// An `Array2<f32>` representing the result of the matrix multiplication.

pub fn matmul(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let mut result = Array2::<f32>::zeros((a.nrows(), b.ncols())); // get a's rows and b's cols, panic if mismatch!
    general_mat_mul(1.0, a, b, 0.0, &mut result); // compute res <= 1* AB + 0 * res
    result
}
