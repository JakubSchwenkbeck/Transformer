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
