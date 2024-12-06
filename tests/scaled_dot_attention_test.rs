use ndarray::array;
use Transformer::attention::scaled_dot_attention::query_key_product;
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
