use Transformer::utils::linear_algebra;
use ndarray::{array};
fn main() {

    let a = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]; // 2x3 matrix

    let b = array![
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0]
    ]; // 3x2 matrix

    let result = linear_algebra::matmul(&a, &b);

    // Print the result
    println!("Result of matrix multiplication:\n{}", result);

}
