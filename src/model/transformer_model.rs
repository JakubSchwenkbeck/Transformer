#![allow(warnings)]
use crate::attention::softmax::{softmax_matrix, softmax_vec, softmax_vector};
use crate::data::tokenizer::Tokenizer;
use crate::layers::feedforward_layer::FeedForwardLayer;
use crate::math::linear_algebra::flatten_3d_array;
use crate::model::decoder::decoding;
use crate::model::embedding::{predict_tokens, Embedding};
use crate::model::encoder::encoding;
use crate::settings::*;
use ndarray::{Array1, Array2, Array3};
use rand::Rng;
use std::collections::HashMap;

pub fn transformer_model(
    sentence: &str,                 // Input sentence
    vocab: &HashMap<String, usize>, // Vocabulary
) -> Vec<String> {
    // Initialize Tokenizer and Embedding layer
    let tokenizer = Tokenizer::new(vocab.clone());
    let embedding = Embedding::new(vocab.len(), EMBEDDING_SIZE); // Initialize embedding layer

    // Tokenize and embed the input sentence
    let tokens = tokenizer.tokenize(sentence);
    let embeddings = embedding.forward(tokens.clone());

    // Convert embeddings to Array3 (batch_size, seq_length, embed_size)
    let input_tensor = Array3::from_shape_fn(
        (BATCH_SIZE, tokens.len(), EMBEDDING_SIZE),
        |(batch, seq, _)| embeddings[[seq, batch]],
    );

    // Initialize gamma and beta for layer normalization
    let gamma = Array2::ones((1, EMBEDDING_SIZE)); // Example gamma (scale parameter)
    let beta = Array2::zeros((1, EMBEDDING_SIZE)); // Example beta (shift parameter)

    // Initialize the feed-forward layer with correct types
    let feed_forward_layer =
        FeedForwardLayer::new(BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE, DROPOUT_RATE);

    // Perform encoding (transformer encoder)
    let encoded = encoding(
        input_tensor.clone(),
        gamma.clone(),
        beta.clone(),
        EPSILON,
        &feed_forward_layer,
    );

    // Perform decoding (transformer decoder)
    let decoded = decoding(
        input_tensor,
        encoded.clone(),
        gamma,
        beta,
        EPSILON,
        &feed_forward_layer,
    );

    // Apply final linear transformation
    let output_projection = Array2::ones((OUTPUT_SIZE, vocab.len())); // All ones weights
    let logits = flatten_3d_array(decoded).dot(&output_projection); // Linear layer

    // Apply softmax to logits
    let probabilities = softmax_matrix(&logits);

    // Convert probabilities back to text using the tokenizer
    predict_tokens(probabilities.view(), vocab)
}