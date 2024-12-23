#![allow(warnings)]
use crate::attention::softmax::{softmax_matrix, softmax_vec, softmax_vector};
use crate::data::learnable::{initialize_weights, LearnableWeights};
use crate::data::tokenizer::Tokenizer;
use crate::layers::feedforward_layer::FeedForwardLayer;
use crate::math::linear_algebra::flatten_3d_array;
use crate::model::decoder::decoding;
use crate::model::embedding::{predict_tokens, Embedding};
use crate::model::encoder::encoding;
use crate::settings::*;
use crate::training::loss_function::cross_entropy_loss;
use ndarray::{Array1, Array2, Array3};
use rand::Rng;
use std::collections::HashMap;

pub fn transformer_model(
    sentence: &str,       // Input sentence
    tokenizer: Tokenizer, // Vocabulary
) -> Vec<String> {
    // Initialize Tokenizer and Embedding layer

    let embedding = Embedding::new(tokenizer.vocab.len(), EMBEDDING_SIZE); // Initialize embedding layer

    // Tokenize and embed the input sentence
    let tokens = tokenizer.tokenize(sentence);
    let embeddings = embedding.forward(tokens.clone());

    // Convert embeddings to Array3 (batch_size, seq_length, embed_size)
    let input_tensor = Array3::from_shape_fn(
        (BATCH_SIZE, tokens.len(), EMBEDDING_SIZE),
        |(_, seq, embed)| embeddings[[seq, embed]],
    );

    // Initialize gamma and beta for layer normalization
    let gamma = Array2::ones((1, EMBEDDING_SIZE)); // Example gamma (scale parameter)
    let beta = Array2::zeros((1, EMBEDDING_SIZE)); // Example beta (shift parameter)

    let learnable_weights = LearnableWeights::new(
        OUTPUT_SIZE,
        HIDDEN_SIZE,
        tokenizer.vocab.len(),
        EMBEDDING_SIZE,
        EMBEDDING_SIZE,
        HIDDEN_SIZE,
    );
    // Initialize the feed-forward layer with correct types
    let feed_forward_layer = FeedForwardLayer::new(&learnable_weights, DROPOUT_RATE);

    // Perform encoding with N stacked layers
    let mut encoded = input_tensor.clone();
    for _ in 0..NUM_LAYERS {
        encoded = encoding(
            encoded,
            gamma.clone(),
            beta.clone(),
            EPSILON,
            &feed_forward_layer,
        );
    }

    // Perform decoding with N stacked layers
    let mut decoded = input_tensor.clone();
    for _ in 0..NUM_LAYERS {
        decoded = decoding(
            decoded,
            encoded.clone(),
            gamma.clone(),
            beta.clone(),
            EPSILON,
            &feed_forward_layer,
        );
    }

    // Apply final linear transformation
    let output_projection = Array2::ones((OUTPUT_SIZE, tokenizer.vocab.len())); // All ones weights
    let logits = flatten_3d_array(decoded).dot(&output_projection); // Linear layer

    /*
    let targets = Array1::from(vec![1, 2, 3]);
    let loss = cross_entropy_loss(&logits.clone(), &targets,5);
    println!("LOSS: {:?}", loss);

     */

    // Apply softmax to logits
    let probabilities = softmax_matrix(&logits);

    // Convert probabilities back to text using the tokenizer
    predict_tokens(probabilities.view(), &tokenizer.vocab)
}
