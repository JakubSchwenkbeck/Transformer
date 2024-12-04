use ndarray::Array2;

pub fn relu(a: &Array2<f32>) -> Array2<f32> {
    a.mapv(|x| if x > 0.0 { x } else { 0.0 }) // mapValue returns a new and owned Array!!

    // Relu :   B(i,j) = max(0,A(i,j))
}

use std::f32::consts::PI;
// GELU (Gaussian Error Linear Unit)
//Formula:
// GELU(x)=x⋅Φ(x)
// with GELU(x)≈0.5x(1+tanh(sqrt 2/π)(x+0.044715x^3)))
// Empirically better than Relu
pub fn gelu(a: &Array2<f32>) -> Array2<f32> {
    fn gelu_calc(y: f32) -> f32 {
        y * 0.5f32 * (1f32 + ((2f32 / PI).sqrt() * (y + 0.044715f32 * y.powi(3))).tanh())
    }

    a.mapv(gelu_calc)
}
