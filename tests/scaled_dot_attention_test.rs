use ndarray::{array, s, Array1, Array3};
use Transformer::attention::scaled_dot_attention::{
    query_key_product, scaled_dot_product, scaled_dot_product_attention,
};
use Transformer::attention::softmax::softmax_3d;

#[test]
fn test_query_key_product() {
    let q = array![
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    ]; // Shape: (2, 2, 3)

    let k = array![
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]
    ]; // Shape: (2, 3, 3)

    let expected = array![
        [[14.0, 32.0, 50.0], [32.0, 77.0, 122.0]],      // Batch 1
        [[266.0, 338.0, 410.0], [365.0, 464.0, 563.0]]  // Batch 2
    ]; // Shape: (2, 2, 3)

    let result = query_key_product(q.clone(), k.clone());
    assert_eq!(result.unwrap(), expected);
}

// test values by https://medium.com/@saraswatp/understanding-scaled-dot-product-attention-in-transformer-models-5fe02b0f150c
#[test]
fn test_scaled_dot() {
    let a: Array3<f32> = array![[
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [0.1, 0.2, 0.3],
        [1.3, 1.4, 1.5]
    ]];
    // simple case, Q = V = K
    let res = scaled_dot_product(a.clone(), a.clone(), a.clone(), false);

    let expected: Array3<f32> = array![[
        [0.081, 0.185, 0.289, 0.392, 0.081, 0.496],
        [0.185, 0.445, 0.705, 0.964, 0.185, 1.224],
        [0.289, 0.705, 1.122, 1.538, 0.289, 1.954],
        [0.392, 0.964, 1.538, 2.109, 0.392, 2.682],
        [0.081, 0.185, 0.289, 0.392, 0.081, 0.496],
        [0.496, 1.224, 1.954, 2.682, 0.496, 3.384]
    ]];

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

    assert_eq!(res.shape()[2], expected.shape()[2]);
}

// test values by https://medium.com/@saraswatp/understanding-scaled-dot-product-attention-in-transformer-models-5fe02b0f150c
#[test]
fn softmax_scaled_dot_test() {
    let a: Array3<f32> = array![[
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [0.1, 0.2, 0.3],
        [1.3, 1.4, 1.5]
    ]];
    // simple case, Q = V = K
    let res = softmax_3d(&scaled_dot_product(a.clone(), a.clone(), a.clone(), false));
    let result: Array1<f32> = res.slice(s![0, 0, ..]).to_owned();
    let expected: Array1<f32> = array![0.145, 0.162, 0.171, 0.189, 0.145, 0.21];
    for i in 0..expected.len() {
        assert!(
            (result[i] - expected[i]) < 0.001,
            "Softmax scaled dot is too far off!"
        );
    }
}

#[test]
fn full_scaled_dot_attention_test() {
    let a: Array3<f32> = array![[
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [0.1, 0.2, 0.3],
        [1.3, 1.4, 1.5]
    ]];
    // simple case, Q = V = K
    let res = scaled_dot_product_attention(a.clone(), a.clone(), a.clone(), false);
    let result: Array1<f32> = res.slice(s![0, 0, ..]).to_owned();
    let output = [0.7836, 0.8836, 0.9836];
    for i in 0..result.len() {
        assert!(
            (result[i] - output[i]) < 0.001,
            "Softmax scaled dot is too far off!"
        );
    }
}
