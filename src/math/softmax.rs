use ndarray::{array, Array, Array1, Array2, ArrayView1, Axis};




fn softmax_vector(vec: ArrayView1<f32>) -> Array1<f32> {
    let max = vec.fold(f32::NEG_INFINITY, |a, &b| a.max(b)); // Stabilize by subtracting max
    let exp_vec = vec.mapv(|x| (x - max).exp());
    let sum: f32 = exp_vec.sum();
    exp_vec / sum
}


fn softmax_matrix(mat: &Array2<f32>) -> Array2<f32> {
    convert_to_array2(mat.map_axis(Axis(1), |row| softmax_vector(row)))
}





fn convert_to_array2(array1d: Array<Array1<f32>, ndarray::Ix1>) -> Array2<f32> {
    // Check if the input array is non-empty
    assert!(!array1d.is_empty(), "Input array must not be empty.");

    // Stack all Array1<f32> along a new axis (axis 0) to create an Array2<f32>
    let rows = array1d.len();
    let cols = array1d[0].len(); // Assume all rows have the same length
    let mut data = Vec::new();

    for row in array1d.iter() {
        assert_eq!(row.len(), cols, "All rows must have the same number of columns.");
        data.extend_from_slice(row.as_slice().unwrap()); // Flatten each row and push into Vec
    }

    Array2::from_shape_vec((rows, cols), data).unwrap() // Convert Vec to Array2
}

#[test]
pub fn convert_to_array2_test(){


    let data = Array::from(vec![
        Array1::from(vec![1.0, 2.0, 3.0]),
        Array1::from(vec![4.0, 5.0, 6.0]),
        Array1::from(vec![7.0, 8.0, 9.0]),
    ]);

    // Convert to Array2<f32>
    let matrix = convert_to_array2(data);

    let d2array :Array2<f32> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

    assert_eq!(matrix.nrows(), 3);
    assert_eq!(matrix.ncols(), 3);
    assert_eq!(matrix,d2array);

}
