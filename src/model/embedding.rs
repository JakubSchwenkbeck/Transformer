#![allow(dead_code)]

use crate::math::positional_encoding::sinusoidal_pos_encoding;
use ndarray::{s, Array, Array2, ArrayView, ArrayView2, Axis, Ix1};
use std::cmp::Ordering;
//use rand::Rng;
use rand::Rng;
use std::collections::HashMap;

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
    /// Compute the sinusoidal positional encodings for a given sequence length.
    pub fn get_positional_encodings(&self, seq_len: usize) -> Array2<f32> {
        let mut pos_encodings = Array2::zeros((seq_len, self.embed_size));

        for pos in 0..seq_len {
            for i in 0..self.embed_size {
                pos_encodings[(pos, i)] = sinusoidal_pos_encoding(pos, i, self.embed_size);
            }
        }
        pos_encodings
    }

    /// Forward pass through the embedding layer, adding positional encodings to token embeddings.
    pub fn forward(&self, tokens: Vec<usize>) -> Array2<f32> {
        let seq_len = tokens.len();
        let mut token_embeddings: Vec<f32> = Vec::new();

        // For each token, get the corresponding embedding and append it to the token_embeddings vector
        for &token in &tokens {
            token_embeddings.extend(self.weights.slice(s![token, ..]).to_vec());
        }

        // Create the Array2 from the flattened token_embeddings vector
        let token_embeddings =
            Array2::from_shape_vec((seq_len, self.embed_size), token_embeddings).unwrap();
        let pos_encodings = self.get_positional_encodings(seq_len);
        token_embeddings + pos_encodings
    }

    pub fn retrieve_tokens(
        &self,
        decoded_embeddings: Array2<f32>,
        vocab: &HashMap<String, usize>, // Token to index mapping
    ) -> Vec<String> {
        // Reverse the vocab to get a mapping from index to token
        let index_to_token: HashMap<usize, String> =
            vocab.iter().map(|(k, &v)| (v, k.clone())).collect();

        let mut predicted_tokens = Vec::new();

        for decoded in decoded_embeddings.axis_iter(Axis(0)) {
            let decoded_norm = norm(decoded);
            if decoded_norm < 1e-6 {
                continue; // Skip null rows
            }
            // Compute cosine similarity with all embeddings in `weights`
            let similarities: Vec<f32> = self
                .weights
                .axis_iter(Axis(0))
                .map(|embedding| embedding.dot(&decoded) / (norm(embedding) * norm(embedding)))
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

pub fn norm(vector: ArrayView<f32, Ix1>) -> f32 {
    vector.mapv(|x| x * x).sum().sqrt()
}
pub fn predict_tokens(
    probabilities: ArrayView2<f32>,
    vocab: &HashMap<String, usize>,
) -> Vec<String> {
    // Reverse the vocab to get a mapping from index to token
    let index_to_token: HashMap<usize, String> =
        vocab.iter().map(|(k, &v)| (v, k.clone())).collect();

    let mut predicted_tokens = Vec::new();

    for probs in probabilities.axis_iter(Axis(0)) {
        // Iterate over the rows (sequence tokens)
        // Find the index of the maximum probability
        let max_index = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal)) // TODO : Need better MaM handling than asserting equal
            .unwrap()
            .0;

        // Map the index to the corresponding token
        if let Some(token) = index_to_token.get(&max_index) {
            predicted_tokens.push(token.clone());
        }
    }

    predicted_tokens
}
