#![allow(unused_imports)]

use contracts::{ensures, requires};
// {array} import is not recognized as it is used in #[test]
use ndarray::{array, s, Array, Array1, Array2, Array3, ArrayView1, Axis};

//noinspection ALL
#[requires(!vec.is_empty(), "Input vector must not be empty.")]
#[ensures(ret.len() == vec.len(), "Output vector must have the same length as the input vector.")]
pub fn softmax_vector(vec: ArrayView1<f32>) -> Array1<f32> {
    let max = vec.fold(f32::NEG_INFINITY, |a, &b| a.max(b)); // Stabilize by subtracting max
    let exp_vec = vec.mapv(|x| (x - max).exp());
    let sum: f32 = exp_vec.sum();
    exp_vec / sum
}
#[requires(!vec.is_empty(), "Input vector must not be empty.")]
pub fn softmax_vec(vec: Vec<f32>) -> Array1<f32> {
    let array = Array1::from(vec); // Convert Vec<f32> to Array1<f32>
    softmax_vector(array.view())
}

#[requires(mat.shape().len() == 2, "Input matrix must be 2-dimensional.")]
pub fn softmax_matrix(mat: &Array2<f32>) -> Array2<f32> {
    convert_to_array2(mat.map_axis(Axis(1), softmax_vector))
}

#[requires(attention_scores.shape().len() == 3, "Input tensor must be 3-dimensional.")]
pub fn softmax_3d(attention_scores: &Array3<f32>) -> Array3<f32> {
    let batch_size = attention_scores.shape()[0];
    let mut softmax_result = Array3::<f32>::zeros(attention_scores.raw_dim());

    for b in 0..batch_size {
        // Extract the 2D slice for the current batch
        let mat = attention_scores.slice(s![b, .., ..]);

        // Convert the slice to a 2D array to pass into softmax_matrix
        let mat_2d = mat.to_owned(); // Clone into Array2
        let softmaxed = softmax_matrix(&mat_2d); // Apply softmax

        // Store the result back into the appropriate slice
        softmax_result.slice_mut(s![b, .., ..]).assign(&softmaxed);
    }

    softmax_result
}
#[requires(!array1d.is_empty(), "Input array must not be empty.")]
#[requires(array1d.iter().all(|row| !row.is_empty()), "All rows must be non-empty.")]
fn convert_to_array2(array1d: Array<Array1<f32>, ndarray::Ix1>) -> Array2<f32> {
    // Check if the input array is non-empty
    assert!(!array1d.is_empty(), "Input array must not be empty.");

    // Stack all Array1<f32> along a new axis (axis 0) to create an Array2<f32>
    let rows = array1d.len();
    let cols = array1d[0].len(); // Assume all rows have the same length
    let mut data = Vec::new();

    for row in array1d.iter() {
        assert_eq!(
            row.len(),
            cols,
            "All rows must have the same number of columns."
        );
        data.extend_from_slice(row.as_slice().unwrap()); // Flatten each row and push into Vec
    }

    Array2::from_shape_vec((rows, cols), data).unwrap() // Convert Vec to Array2
}

#[test]
pub fn convert_to_array2_test() {
    let data = Array::from(vec![
        Array1::from(vec![1.0, 2.0, 3.0]),
        Array1::from(vec![4.0, 5.0, 6.0]),
        Array1::from(vec![7.0, 8.0, 9.0]),
    ]);

    // Convert to Array2<f32>
    let matrix = convert_to_array2(data);

    let d2array: Array2<f32> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

    assert_eq!(matrix.nrows(), 3);
    assert_eq!(matrix.ncols(), 3);
    assert_eq!(matrix, d2array);
}

#[test]
#[should_panic]
pub fn convert_to_array2_test_panic() {
    let data = Array::from(vec![]);
    let _ = convert_to_array2(data);
}
