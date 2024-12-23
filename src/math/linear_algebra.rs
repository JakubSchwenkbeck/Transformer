#![allow(warnings)]
use contracts::{ensures, requires};
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
#[requires(a.ncols() == b.nrows(), "Matrix dimensions are incompatible for multiplication.")]
#[ensures(ret.is_ok(), "Matrix multiplication should be successful")]
#[ensures(ret.as_ref().unwrap().nrows() > 0, "The resulting matrix must have more than 0 rows.")]
#[ensures(ret.as_ref().unwrap().ncols() > 0, "The resulting matrix must have more than 0 columns.")]
pub fn matmul(a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>, &'static str> {
    if a.ncols() != b.nrows() {
        return Err("Matrix dimensions are incompatible for multiplication.");
    }
    let mut ret = Array2::<f32>::zeros((a.nrows(), b.ncols()));
    general_mat_mul(1.0, a, b, 0.0, &mut ret);
    Ok(ret)
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
#[requires(a.shape().len() == 3, "Input tensor a must have 3 dimensions")]
#[requires(b.shape().len() == 3, "Input tensor b must have 3 dimensions")]
#[requires(a.shape()[0] == b.shape()[0], "Batch sizes must match")]
#[requires(a.shape()[2] == b.shape()[1], "Inner dimensions must align for matrix multiplication")]
//#[ensures(ret.shape().len() == 3, "The resulting tensor must have 3 dimensions.")]
//#[ensures(ret.iter().all(|&x| x.is_finite()), "All elements in the resulting tensor must be finite.")]
pub fn tensor_product(a: &Array3<f32>, b: &Array3<f32>) -> Array3<f32> {
    // Check that batch sizes match and if dimension align
    assert_eq!(a.shape()[0], b.shape()[0], "Batch sizes must match");
    assert_eq!(a.shape()[2], b.shape()[1], "Inner dimensions must align");

    let batch_size = a.shape()[0];
    let m = a.shape()[1]; // Number of rows in each matrix in a.
    let n = b.shape()[2]; // Number of columns in each matrix in b.

    // Initialize a 3D tensor for the result, filled with zeros.
    // Its shape corresponds to (batch_size, m, n).
    let mut ret = Array3::<f32>::zeros((batch_size, m, n));

    for i in 0..batch_size {
        // - `s![i, .., ..]` selects the `i`th matrix (2D slice) in the batch.

        let a_slice = a.slice(s![i, .., ..]);
        let b_slice = b.slice(s![i, .., ..]);
        let mut ret_slice = ret.slice_mut(s![i, .., ..]); // Mutable slice of the result matrix for this batch.

        general_mat_mul(1.0, &a_slice, &b_slice, 0.0, &mut ret_slice);
    }

    ret
}

/// Applies a linear projection to a 3D tensor using a weight matrix.
///
/// # Arguments
/// - `x`: The input 3D tensor (e.g., [batch, seq_len, input_dim]).
/// - `w`: The weight matrix for the projection (e.g., [input_dim, output_dim]).
///
/// # Returns
/// A new 3D tensor with the projection applied (e.g., [batch, seq_len, output_dim]).
#[requires(x.shape().len() == 3, "Input tensor x must have 3 dimensions")]
#[requires(w.shape().len() == 2, "Weight matrix w must have 2 dimensions")]
#[requires(x.shape()[2] == w.shape()[0], "Input feature size must match the weight matrix's rows")]
//#[ensures(ret.shape().len() == 3, "The resulting tensor must have 3 dimensions.")]
//#[ensures(ret.iter().all(|&x| x.is_finite()), "All elements in the resulting tensor must be finite.")]
pub fn apply_projection(x: &Array3<f32>, w: &Array2<f32>) -> Array3<f32> {
    let batch_size = x.shape()[0];
    let seq_len = x.shape()[1];
    let d_model = x.shape()[2]; // Should be the same as w.shape()[0]
    assert_eq!(d_model, w.shape()[0]);
    let d_k = w.shape()[1]; // Output dimension (head dimension)

    // Initialize the ret tensor with shape (batch_size, seq_len, d_k)
    let mut ret = Array3::<f32>::zeros((batch_size, seq_len, d_k));

    // Perform matrix multiplication for each batch
    for i in 0..batch_size {
        let x_slice = x.slice(s![i, .., ..]); // Slice the i-th batch (shape: (seq_len, d_model))
        let mul = matmul(&x_slice.to_owned(), w); // Perform matrix multiplication
        if mul.is_ok() {
            ret.slice_mut(s![i, .., ..]).assign(&mul.unwrap());
        }
    }

    ret
}

/// Flattens a 3D array into a 2D array.
///
/// # Parameters
/// - `batch`: A 3D tensor of shape (batch_size, seq_length, embed_size).
///
/// # Returns
/// A 2D tensor of shape (batch_size * seq_length, embed_size).
#[requires(batch.shape().len() == 3, "Input tensor must have 3 dimensions")]
#[ensures(ret.shape().len() == 2, "The resulting tensor must have 2 dimensions.")]
//#[ensures(ret.iter().all(|&x| x.is_finite()), "All elements in the resulting tensor must be finite.")]
pub fn flatten_3d_array(batch: Array3<f32>) -> Array2<f32> {
    let (batch_size, seq_length, embed_size) = batch.dim();
    batch
        .to_shape((batch_size * seq_length, embed_size))
        .unwrap()
        .to_owned()
}
