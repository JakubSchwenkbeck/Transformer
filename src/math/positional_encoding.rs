/// Computes the sinusoidal positional encoding for a given position and dimension.
///
/// This encoding is used in Transformer models to represent token positions
/// in a sequence. It alternates between sine and cosine based on the dimension index.
///
/// # Arguments
/// - `pos` - Token position in the sequence (must be >= 0).
/// - `index` - Dimension index.
/// - `embedding_size` - Dimensionality of the embedding space.
///
/// # Returns
/// The positional encoding value (as `f32`).
pub fn sinusoidal_pos_encoding(pos: usize, index: usize, embedding_size: usize) -> f32 {
    if pos == 0 {
        return 0.0;
    };
    let divisor = 10000f32.powf(2.0 * (index as f32 / embedding_size as f32)); // 100000^(2*i / embedding size)

    if index % 2 == 0 {
        // for an even index, use sin, else cos!
        (pos as f32 / divisor).sin()
    } else {
        (pos as f32 / divisor).cos()
    }
}
