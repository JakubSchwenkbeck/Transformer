use ndarray::{array, s, Array2, Array3};
use Transformer::attention::scaled_dot_attention::scaled_dot_product;
use Transformer::attention::softmax::softmax_3d;

fn main() {
    println!("runs successfully!");

    let a: Array3<f32> = array![[
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [0.1, 0.2, 0.3],
        [1.3, 1.4, 1.5]
    ]];

    let scores = scaled_dot_product(a.clone(), a.clone(), a.clone(), false);
    let sm_scores = softmax_3d(&scores);
    // Words corresponding to the input
    let words = ["the", "cat", "sat", "on", "the", "mat"];

    display_attention_weights(sm_scores.slice(s![0, .., ..]).to_owned(), &words);
}
fn display_attention_weights(scores: Array2<f32>, words: &[&str]) {
    println!("Attention Weights (Softmax Scores):\n");

    // Print column headers
    print!("{:<6}", ""); // Empty corner for alignment
    for word in words {
        print!(" {:<5}", word);
    }
    println!(); // New line for clarity

    // Iterate through rows and display each with the corresponding word
    for (i, row) in scores.outer_iter().enumerate() {
        print!("{:<6}", words[i]); // Row label
        for &val in row.iter() {
            print!("{:<6.3}", val); // Print score with 3 decimal places
        }
        println!(); // New line after each row
    }
}
