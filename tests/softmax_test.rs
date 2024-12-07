#![allow(clippy::excessive_precision)]

use ndarray::{array, Array, Array1, Array2, Array3, Ix1};
use Transformer::attention::softmax::{softmax_3d, softmax_matrix, softmax_vector};

#[test]
pub fn test_softmax_vector() {
    let vec: Array<f32, Ix1> = Array1::from(vec![-1.0, 0.0, 3.0, 5.0]);

    let res = softmax_vector(vec.view());

    let expected: Array1<f32> =
        Array1::from(vec![0.0021656966, 0.0058869733, 0.11824302, 0.87370431]);

    assert_eq!(res, expected);

    let vec_sum = res.sum();
    assert!(
        1.0 - 1e-7 < vec_sum && vec_sum < 1.0 + 1e-7,
        "Sum too far off"
    );
}

#[test]
pub fn test_softmax_matrix() {
    let mat1: Array2<f32> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let expected1: Array2<f32> = array![
        [0.0900305732, 0.2447284711, 0.6652409558],
        [0.0900305732, 0.2447284711, 0.6652409558],
        [0.0900305732, 0.2447284711, 0.6652409558]
    ];

    for i in 0..expected1.nrows() {
        let vec_sum = expected1.row(i).sum();
        assert!(
            1.0 - 1e-6 < vec_sum && vec_sum < 1.0 + 1e-6,
            "Sum too far off"
        );
    }

    let res1: Array2<f32> = softmax_matrix(&mat1);

    assert_eq!(res1, expected1);

    let mat2: Array2<f32> = array![[-1.0, 3.0, 7.0], [0.0, 2.5, 4.0], [7.0, 8.0, 9.0]];
    let expected2: Array2<f32> = array![
        [0.00032932044, 0.0179802867, 0.98169035],
        [0.0147534745, 0.17973413, 0.805512412],
        [0.0900305732, 0.2447284711, 0.6652409558]
    ];
    for i in 0..expected2.nrows() {
        let vec_sum = expected2.row(i).sum();
        assert!(
            1.0 - 1e-6 < vec_sum && vec_sum < 1.0 + 1e-6,
            "Sum too far off"
        );
    }
    let res2: Array2<f32> = softmax_matrix(&mat2);
    assert_eq!(res2, expected2);
}

#[test]
pub fn test_softmax_3d() {
    let attention_scores: Array3<f32> = array![
        [[2.0, 1.0, 0.1], [1.0, 2.0, 0.3]],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    ]; // Shape: (2, 2, 3)

    let res = softmax_3d(&attention_scores);

    let expected: Array3<f32> = array![
        [[0.659, 0.242, 0.099], [0.237, 0.645, 0.118]],
        [[0.300, 0.332, 0.368], [0.300, 0.332, 0.368]]
    ];

    for i in 0..expected.shape()[0] {
        // Iterate over the first dimension
        for j in 0..expected.shape()[1] {
            // Iterate over the second dimension
            for k in 0..expected.shape()[2] {
                // Iterate over the third dimension

                assert!(
                    (res[[i, j, k]] - expected[[i, j, k]]).abs() < 0.05,
                    "scaled dot attention too far off"
                );
            }
        }
    }
}
