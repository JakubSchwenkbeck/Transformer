use crate::settings::*;
use ndarray::{Array1, Array2};

pub struct LearnableWeights {
    // Embedding Layer
    pub embedding: Array2<f32>, // (vocab_size, embedding_dim)

    // Attention Mechanism
    pub query_weights: Array2<f32>, // (embedding_dim, attention_dim)
    pub key_weights: Array2<f32>,   // (embedding_dim, attention_dim)
    pub value_weights: Array2<f32>, // (embedding_dim, attention_dim)
    pub output_projection: Array2<f32>, // (attention_dim, embedding_dim)

    // Feedforward Network
    pub linear1_weights: Array2<f32>, // (embedding_dim, ffn_dim)
    pub linear2_weights: Array2<f32>, // (ffn_dim, embedding_dim)
    pub bias1: Array1<f32>,
    pub bias2: Array1<f32>,

    // Layer Normalization
    pub layer_norm_scale: Vec<f32>, // (embedding_dim,)
    pub layer_norm_shift: Vec<f32>, // (embedding_dim,)

    // Output Layer
    pub output_projection_vocab: Array2<f32>, // (embedding_dim, vocab_size)
}

impl LearnableWeights {
    pub fn new(
        output_size: usize,
        hidden_size: usize,
        vocab_size: usize,
        embedding_dim: usize,
        attention_dim: usize,
        ffn_dim: usize,
    ) -> Self {
        LearnableWeights {
            // Embedding Layer
            embedding: Array2::ones((vocab_size, embedding_dim)),

            // Attention Mechanism
            query_weights: Array2::ones((embedding_dim, attention_dim)),
            key_weights: Array2::ones((embedding_dim, attention_dim)),
            value_weights: Array2::ones((embedding_dim, attention_dim)),
            output_projection: Array2::ones((attention_dim, embedding_dim)),

            // Feedforward Network
            linear1_weights: Array2::ones((embedding_dim, ffn_dim)),
            linear2_weights: Array2::ones((ffn_dim, embedding_dim)),
            bias1: Array1::zeros(hidden_size),
            bias2: Array1::zeros(output_size),

            // Layer Normalization
            layer_norm_scale: vec![1.0; embedding_dim], // Initialize scale to 1
            layer_norm_shift: vec![0.0; embedding_dim], // Initialize shift to 0

            // Output Layer
            output_projection_vocab: Array2::zeros((embedding_dim, vocab_size)),
        }
    }
}
pub fn initialize_weights() -> LearnableWeights {
    LearnableWeights::new(
        OUTPUT_SIZE, // output_size
        HIDDEN_SIZE, // hidden_size
        D_MODEL,     // vocab_size
        D_MODEL,     // embedding_dim
        D_K,         // attention_dim (for keys, queries)
        D_V,         // ffn_dim (could align with embedding_dim or specific)
    )
}
