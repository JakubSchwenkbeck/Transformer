
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

pub fn matmul(a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>, &'static str> {
    if a.ncols() != b.nrows() {
        return Err("Matrix dimensions are incompatible for multiplication.");
    }
    let mut result = Array2::<f32>::zeros((a.nrows(), b.ncols()));
    general_mat_mul(1.0, a, b, 0.0, &mut result);
    Ok(result)
}

