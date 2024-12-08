use ndarray::Array3;
use Transformer::attention::multihead_attention::*;

/// Helper function to create a tensor filled with a specific value.
fn create_input(batch_size: usize, seq_len: usize, feature_dim: usize) -> Array3<f32> {
    Array3::<f32>::ones((batch_size, seq_len, feature_dim))
}

#[test]
fn test_split_into_heads_basic() {
    let q = create_input(2, 5, 8); // batch_size = 2, seq_len = 5, feature_dim = 8
    let k = create_input(2, 5, 8);
    let v = create_input(2, 5, 8);
    let num_heads = 2;

    let (q_heads, k_heads, v_heads) = split_into_heads(&q, &k, &v, num_heads);

    // Assert that the shape of the outputs is (batch_size, num_heads, seq_len, head_dim)
    assert_eq!(q_heads.shape(), &[2, 2, 5, 4]); // (batch_size, num_heads, seq_len, head_dim)
    assert_eq!(k_heads.shape(), &[2, 2, 5, 4]);
    assert_eq!(v_heads.shape(), &[2, 2, 5, 4]);
}

#[test]
fn test_split_into_heads_single_head() {
    let q = create_input(2, 5, 8); // batch_size = 2, seq_len = 5, feature_dim = 8
    let k = create_input(2, 5, 8);
    let v = create_input(2, 5, 8);
    let num_heads = 1;

    let (q_heads, k_heads, v_heads) = split_into_heads(&q, &k, &v, num_heads);

    // With 1 head, the head dimension is equal to the feature dimension.
    assert_eq!(q_heads.shape(), &[2, 1, 5, 8]);
    assert_eq!(k_heads.shape(), &[2, 1, 5, 8]);
    assert_eq!(v_heads.shape(), &[2, 1, 5, 8]);
}

#[test]
fn test_split_into_heads_edge_case() {
    let q = create_input(1, 5, 8); // batch_size = 1, seq_len = 5, feature_dim = 8
    let k = create_input(1, 5, 8);
    let v = create_input(1, 5, 8);
    let num_heads = 4;

    let (q_heads, k_heads, v_heads) = split_into_heads(&q, &k, &v, num_heads);

    // With 4 heads, the head dimension should be 8 / 4 = 2
    assert_eq!(q_heads.shape(), &[1, 4, 5, 2]);
    assert_eq!(k_heads.shape(), &[1, 4, 5, 2]);
    assert_eq!(v_heads.shape(), &[1, 4, 5, 2]);
}

#[test]
fn test_split_into_heads_invalid_num_heads() {
    let q = create_input(2, 5, 8); // batch_size = 2, seq_len = 5, feature_dim = 8
    let k = create_input(2, 5, 8);
    let v = create_input(2, 5, 8);

    // Invalid case: num_heads > feature_dim
    let num_heads = 16;
    let result = std::panic::catch_unwind(|| {
        split_into_heads(&q, &k, &v, num_heads);
    });
    assert!(
        result.is_err(),
        "Expected panic when num_heads > feature_dim"
    );

    // Invalid case: feature_dim is not divisible by num_heads
    let num_heads = 3;
    let result = std::panic::catch_unwind(|| {
        split_into_heads(&q, &k, &v, num_heads);
    });
    assert!(
        result.is_err(),
        "Expected panic when feature_dim is not divisible by num_heads"
    );
}
