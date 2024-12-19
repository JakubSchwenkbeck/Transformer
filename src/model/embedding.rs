#![allow(dead_code)]

use std::collections::HashMap;
use ndarray::{s, Array, Array2, ArrayView2, Axis};
use rand::Rng;

pub struct Embedding {
    vocab_size: usize,
    embed_size: usize,
    weights: Array2<f32>,
}

impl Embedding {
    pub fn new(vocab_size: usize, embed_size: usize) -> Self {
        // Initialize with random values for simplicity
        let mut rng = rand::rng();
        let weights =
            Array::from_shape_fn((vocab_size, embed_size), |_| rng.random_range(-1.0..1.0));
        Embedding {
            vocab_size,
            embed_size,
            weights,
        }
    }

    // Inside Embedding::forward
    pub fn forward(&self, tokens: Vec<usize>) -> Array2<f32> {
        let mut token_embeddings: Vec<f32> = Vec::new();

        // For each token, get the corresponding embedding and append it to the token_embeddings vector
        for &token in &tokens {
            token_embeddings.extend(self.weights.slice(s![token, ..]).to_vec());
        }

        // Create the Array2 from the flattened token_embeddings vector
        Array2::from_shape_vec((tokens.len(), self.embed_size), token_embeddings).unwrap()
    }

    pub fn predict_tokens(probabilities: ArrayView2<f32>, vocab : &HashMap<String, usize> ) -> Array2<f32> {
        let mut predicted_tokens = Vec::new();

        for probs in probabilities.axis_iter(Axis(0)) { // Iterate over the rows (sequence tokens)
            let max_index = probs.argmax().unwrap(); // Find the index of the maximum probability
            predicted_tokens.push(vocab[max_index]); // Map the index to the vocabulary

        }
        predicted_tokens
    }

    pub fn retrieve_tokens(
        decoded_embeddings: Array2<f32>,
        weights: &Array2<f32>,  // Embedding matrix
        vocab: &HashMap<String, usize>, // Token to index mapping
    ) -> Vec<String> {
        // Reverse the vocab to get a mapping from index to token
        let index_to_token: HashMap<usize, String> = vocab.iter().map(|(k, &v)| (v, k.clone())).collect();

        let mut predicted_tokens = Vec::new();

        for decoded in decoded_embeddings.axis_iter(Axis(0)) {
            // Compute cosine similarity with all embeddings in `weights`
            let similarities: Vec<f32> = weights
                .axis_iter(Axis(0))
                .map(|embedding| embedding.dot(&decoded) / (embedding.norm() * decoded.norm()))
                .collect();

            // Find the index of the maximum similarity
            let best_match = similarities
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;

            // Map index to the corresponding token
            if let Some(token) = index_to_token.get(&best_match) {
                predicted_tokens.push(token.clone());
            }
        }

        predicted_tokens
    }
}
