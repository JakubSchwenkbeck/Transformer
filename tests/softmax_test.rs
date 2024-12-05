#![allow(clippy::excessive_precision)]

use ndarray::{array, Array, Array1, Array2, Ix1};
use Transformer::attention::softmax::{softmax_matrix, softmax_vector};

#[test]
pub fn test_softmax_vector() {
    let vec: Array<f32, Ix1> = Array1::from(vec![-1.0, 0.0, 3.0, 5.0]);

    let res = softmax_vector(vec.view());

    let expected: Array1<f32> =
        Array1::from(vec![0.0021656966, 0.0058869733, 0.11824302, 0.87370431]);

    assert_eq!(res, expected);
}

#[test]
pub fn test_softmax_matrix() {
    let mat1: Array2<f32> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let expected1: Array2<f32> = array![
        [0.0900305732, 0.2447284711, 0.6652409558],
        [0.0900305732, 0.2447284711, 0.6652409558],
        [0.0900305732, 0.2447284711, 0.6652409558]
    ];

    let res1: Array2<f32> = softmax_matrix(&mat1);

    assert_eq!(res1, expected1);

    let mat2: Array2<f32> = array![[-1.0, 3.0, 7.0], [0.0, 2.5, 4.0], [7.0, 8.0, 9.0]];
    let expected2: Array2<f32> = array![
        [0.00032932044, 0.0179802867, 0.98169035],
        [0.0147534745, 0.17973413, 0.805512412],
        [0.0900305732, 0.2447284711, 0.6652409558]
    ];

    let res2: Array2<f32> = softmax_matrix(&mat2);

    assert_eq!(res2, expected2);
}
