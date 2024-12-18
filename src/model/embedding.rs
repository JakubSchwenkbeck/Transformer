#![allow(dead_code)]
use ndarray::{s, Array, Array2};
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
}
