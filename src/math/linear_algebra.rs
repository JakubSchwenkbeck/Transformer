use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array1, Array2, Array3};

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

pub fn dotproduct(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    a.dot(b)
}

/// Computes the tensor product (batched matrix multiplication) of two 3D tensors `a` and `b`.
///
/// # Parameters:
/// - `a`: 3D tensor of shape (batch_size, m, k).
/// - `b`: 3D tensor of shape (batch_size, k, n).
///
/// # Returns:
/// A 3D tensor of shape (batch_size, m, n) containing the result of the batched matrix multiplication.
///
/// # Panics:
/// - If the batch sizes of `a` and `b` don't match.
/// - If the inner dimensions (`k` in `a` and `b`) don't align for matrix multiplication.
pub fn tensor_product(a: &Array3<f32>, b: &Array3<f32>) -> Array3<f32> {
    // Check that batch sizes match and if dimension align
    assert_eq!(a.shape()[0], b.shape()[0], "Batch sizes must match");
    assert_eq!(a.shape()[2], b.shape()[1], "Inner dimensions must align");

    let batch_size = a.shape()[0];
    let m = a.shape()[1]; // Number of rows in each matrix in a.
    let n = b.shape()[2]; // Number of columns in each matrix in b.

    // Initialize a 3D tensor for the result, filled with zeros.
    // Its shape corresponds to (batch_size, m, n).
    let mut result = Array3::<f32>::zeros((batch_size, m, n));

    for i in 0..batch_size {
        // - `s![i, .., ..]` selects the `i`th matrix (2D slice) in the batch.

        let a_slice = a.slice(s![i, .., ..]);
        let b_slice = b.slice(s![i, .., ..]);
        let mut result_slice = result.slice_mut(s![i, .., ..]); // Mutable slice of the result matrix for this batch.

        general_mat_mul(1.0, &a_slice, &b_slice, 0.0, &mut result_slice);
    }

    result
}
/// Applies a linear projection to a 3D tensor using a weight matrix.
///
/// # Arguments
/// - `x`: The input 3D tensor (e.g., [batch, seq_len, input_dim]).
/// - `w`: The weight matrix for the projection (e.g., [input_dim, output_dim]).
///
/// # Returns
/// A new 3D tensor with the projection applied (e.g., [batch, seq_len, output_dim]).
pub fn apply_projection(x: &Array3<f32>, w: &Array2<f32>) -> Array3<f32> {
    let batch_size = x.shape()[0];
    let seq_len = x.shape()[1];
    let d_model = x.shape()[2]; // Should be the same as w.shape()[0]
    assert_eq!(d_model, w.shape()[0]);
    let d_k = w.shape()[1]; // Output dimension (head dimension)

    // Initialize the result tensor with shape (batch_size, seq_len, d_k)
    let mut result = Array3::<f32>::zeros((batch_size, seq_len, d_k));

    // Perform matrix multiplication for each batch
    for i in 0..batch_size {
        let x_slice = x.slice(s![i, .., ..]); // Slice the i-th batch (shape: (seq_len, d_model))
        let mul = matmul(&x_slice.to_owned(), w); // Perform matrix multiplication
        if mul.is_ok() {
            result.slice_mut(s![i, .., ..]).assign(&mul.unwrap());
        }
    }

    result
}
