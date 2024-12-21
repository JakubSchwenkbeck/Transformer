use ndarray::{Array2, Array3};
use std::collections::HashMap;
use Transformer::data::tokenizer::{example_tokens, Tokenizer};
use Transformer::example::example;
use Transformer::layers::feedforward_layer::FeedForwardLayer;
use Transformer::math::linear_algebra::flatten_3d_array;
use Transformer::model::decoder::decoding;
use Transformer::model::embedding::Embedding;
use Transformer::model::encoder::encoding;
use Transformer::model::transformer_model::transformer_model;
use Transformer::settings::{BATCH_SIZE, DROPOUT_RATE, EMBEDDING_SIZE, INPUT_SIZE, OUTPUT_SIZE};

fn main() {
    println!("runs successfully!");
    println!("============= ATTENTION WEIGHT EXAMPLE =============");
    example();
    println!("============= TOKENIZER EXAMPLE =============");

    example_tokens();

    println!(" \n \n \n ENCODER/DECODER  \n");

    let vocab = HashMap::from([
        ("hello".to_string(), 0),
        ("world".to_string(), 1),
        ("rust".to_string(), 2),
        ("transformer".to_string(), 3),
        ("learning".to_string(), 4),
        ("model".to_string(), 5),
    ]);

    // Initialize Tokenizer and Embedding layer
    let tokenizer = Tokenizer::new(vocab.clone());
    let embedding = Embedding::new(6, 12); // Example vocab size and embedding size
                                           // Input sentence
    let sentence = "hello world rust";

    // Tokenize and embed the input
    let tokens = tokenizer.tokenize(sentence);
    let embeddings = embedding.forward(tokens.clone());
    println!("embeddings: {:?}", embeddings);

    let input_tensor = Array3::from_shape_fn(
        (BATCH_SIZE, tokens.len(), EMBEDDING_SIZE),
        |(_, seq, embed)| embeddings[[seq, embed]],
    );

    println!("INPUT : {}", input_tensor.clone());
    // Initialize gamma and beta for layer normalization
    let gamma = Array2::ones((1, 12)); // Example gamma (scale parameter)
    let beta = Array2::zeros((1, 12)); // Example beta (shift parameter)

    // Initialize the feed-forward layer with correct types

    let feed_forward_layer =
        FeedForwardLayer::new(BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE, DROPOUT_RATE);

    // Perform encoding (transformer layer)
    let epsilon = 1e-6; // Small epsilon for numerical stability
    let encoded = encoding(
        input_tensor.clone(),
        gamma.clone(),
        beta.clone(),
        epsilon,
        &feed_forward_layer,
    );
    // Perform decoding (transformer layer)
    let decoded = decoding(
        input_tensor,
        encoded.clone(),
        gamma,
        beta,
        epsilon,
        &feed_forward_layer,
    );

    // Print the encoded and decoded output tensors
    println!("Encoded: {:?}", encoded);
    println!("Decoded: {:?}", decoded);

    let tokens = embedding.retrieve_tokens(flatten_3d_array(decoded), &vocab);

    println!("Tokens: {:?}", tokens);

    let predicted_token = transformer_model(sentence, &vocab);

    println!("Predicted Token: {:?}", predicted_token);
}
