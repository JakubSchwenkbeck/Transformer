use crate::attention::scaled_dot_attention::scaled_dot_product;
use crate::attention::softmax::softmax_3d;
use ndarray::{array, s, Array2};

pub fn example() {
    let words = vec!["The", "cat", "sat", "on", "the", "mat"];
    let q = array![[
        [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
        [0.8, 1.0, 0.9, 0.7, 0.3, 0.2],
        [0.6, 0.9, 1.0, 0.8, 0.5, 0.3],
        [0.4, 0.7, 0.8, 1.0, 0.7, 0.6],
        [0.2, 0.3, 0.5, 0.7, 1.0, 0.9],
        [0.1, 0.2, 0.3, 0.6, 0.9, 1.0]
    ]];

    let scores = scaled_dot_product(q.clone(), q.clone(), q.clone(), true);
    let sm_scores = softmax_3d(&scores);
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
