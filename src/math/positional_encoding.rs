pub fn sinusoidal_pos_encoding(pos: usize, index: usize, embedding: usize) -> f32 {
    let divisor = 10000f32.powf(2.0 * (index / embedding) as f32);

    if index % 2 == 0 {
        (pos as f32 / divisor).sin()
    } else {
        (pos as f32 / divisor).cos()
    }
}
