use ndarray::{Array2, Array3};
use std::collections::HashMap;
use Transformer::data::tokenizer::{example_tokens, Tokenizer};
use Transformer::example::example;
use Transformer::layers::feedforward_layer::FeedForwardLayer;
use Transformer::model::decoder::decoding;
use Transformer::model::embedding::Embedding;
use Transformer::model::encoder::encoding;
use Transformer::settings::{BATCH_SIZE, DROPOUT_RATE, INPUT_SIZE, OUTPUT_SIZE};

fn main() {
    println!("runs successfully!");
    println!("============= ATTENTION WEIGHT EXAMPLE =============");
    example();
    println!("============= TOKENIZER EXAMPLE =============");

    example_tokens();

    println!(" \n \n \n ENCODER/DECODER  \n");
    // Example vocabulary
    let vocab = vec![
        ("hello".to_string(), 4),
        ("world".to_string(), 5),
        ("my".to_string(), 6),
        ("name".to_string(), 7),
        ("is".to_string(), 8),
    ]
    .into_iter()
    .collect::<HashMap<String, usize>>();

    // Initialize Tokenizer and Embedding layer
    let tokenizer = Tokenizer::new(vocab);
    let embedding = Embedding::new(10, 64); // Example vocab size and embedding size
                                            // Input sentence
    let sentence = "hello world";

    // Tokenize and embed the input
    let tokens = tokenizer.tokenize(sentence);
    let embeddings = embedding.forward(tokens.clone());

    // Convert embeddings to Array3 (batch_size, seq_length, embed_size)
    let input_tensor = Array3::from_shape_fn((1, tokens.len(), 64), |(batch, seq, _)| {
        embeddings[[seq, batch]]
    });

    // Initialize gamma and beta for layer normalization
    let gamma = Array2::ones((1, 64)); // Example gamma (scale parameter)
    let beta = Array2::zeros((1, 64)); // Example beta (shift parameter)

    // Initialize the feed-forward layer with correct types

    let feed_forward_layer =
        FeedForwardLayer::new(BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE, DROPOUT_RATE);

    // Perform encoding (transformer layer)
    let epsilon = 1e-6; // Small epsilon for numerical stability
    let encoded = encoding(
        input_tensor,
        gamma.clone(),
        beta.clone(),
        epsilon,
        &feed_forward_layer,
    );
    // Perform decoding (transformer layer)
    let decoded = decoding(
        encoded.clone(),
        encoded.clone(),
        gamma,
        beta,
        epsilon,
        &feed_forward_layer,
    );

    // Print the encoded and decoded output tensors
    println!("Encoded: {:?}", encoded);
    println!("Decoded: {:?}", decoded);
}
