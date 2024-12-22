#![allow(warnings)]
use crate::attention::softmax::{softmax_matrix, softmax_vec, softmax_vector};
use crate::data::dataset::{gen_data, Dataset};
use crate::data::learnable::{initialize_weights, LearnableWeights};
use crate::data::tokenizer::Tokenizer;
use crate::layers::feedforward_layer::FeedForwardLayer;
use crate::math::linear_algebra::flatten_3d_array;
use crate::model::decoder::decoding;
use crate::model::embedding::{predict_index, predict_tokens, Embedding};
use crate::model::encoder::encoding;
use crate::settings::*;
use crate::training::loss_function::cross_entropy_loss;
use crate::training::train::{compute_gradients, update_weights};
use ndarray::{Array1, Array2, Array3};
use rand::prelude::SliceRandom;
use rand::Rng;
use std::collections::HashMap;

fn train_model(
    dataset: &Dataset,                       // The training data
    tokenizer: Tokenizer,                    // Vocabulary
    mut learnable_weights: LearnableWeights, // Initial weights
    num_epochs: usize,                       // Number of training epochs
    learning_rate: f32,                      // Learning rate
) -> Vec<String> {
    let vocab_size = tokenizer.vocab.len(); // Vocabulary size
    let mut outputs = Vec::new();           // To store outputs for progress tracking

    // Loop over the number of epochs
    for epoch in 0..num_epochs {
        println!("\n=== Epoch {}/{} ===", epoch + 1, num_epochs);

        // Shuffle the dataset indices
        let mut data_indices: Vec<usize> = (0..dataset.inputs.len()).collect();
        data_indices.shuffle(&mut rand::thread_rng());

        let mut total_loss = 0.0; // Accumulate loss for this epoch
        let mut num_batches = 0;

        // Loop over the training data
        for (step, &idx) in data_indices.iter().enumerate() {
            let input = &dataset.inputs[idx];
            let target = &dataset.targets[idx];

            // Convert to Array1 for processing
            let input_seq = Array1::from(input.clone());
            let target_seq = Array1::from(target.clone());

            // Forward pass: Model prediction
            let (out, logits) = training_model(
                input,
                target_seq.clone(),
                &mut learnable_weights,
                vocab_size,
                tokenizer.vocab.clone(),
            );

            // Compute loss
            let loss = cross_entropy_loss(&logits, &target_seq, vocab_size);
            total_loss += loss; // Accumulate loss for averaging
            num_batches += 1;

            // Backward pass: Compute gradients
            let gradients = compute_gradients(&logits, &target_seq, vocab_size, &learnable_weights);

            // Update weights
            update_weights(&mut learnable_weights, &gradients, learning_rate);

            // Periodically log training progress
            if step % 10 == 0 {
                let decoded_output = tokenizer.detokenize(out.to_vec());
                println!(
                    "Step {}: Loss = {:.4}, Output = {:?}, Expected = {:?}",
                    step,
                    loss,
                    decoded_output,
                    tokenizer.detokenize(target.to_vec())
                );
                outputs.push(decoded_output);
            }
        }

        // End of epoch: Print average loss and track improvement
        let avg_loss = total_loss / num_batches as f32;
        println!("Epoch {} completed with average loss: {:.4}", epoch + 1, avg_loss);

        // For debugging or tracking, we could save weights periodically here.
    }

    println!("\nTraining completed!");
    outputs
}

pub fn train() {
    let (tokenizer, dataset) = gen_data();

    let learnable_weights = LearnableWeights::new(
        OUTPUT_SIZE,
        HIDDEN_SIZE,
        tokenizer.vocab.len(),
        EMBEDDING_SIZE,
        EMBEDDING_SIZE,
        HIDDEN_SIZE,
    );

    // Define the number of epochs and learning rate
    let num_epochs = 10;
    let learning_rate = 0.001;

    // Train the model
    let outputs = train_model(
        &dataset,
        tokenizer,
        learnable_weights,
        num_epochs,
        learning_rate,
    );

    // Print some of the outputs after training
    for output in outputs.iter().take(5) {
        println!("Output: {}", output);
    }
}

pub fn training_model(
    tokens: &Vec<usize>,
    target_seq: Array1<usize>,
    learnable_weights: &mut LearnableWeights,
    vocab_size: usize,
    vocab: HashMap<String, usize>,
) -> (Vec<usize>, Array2<f32>) {
    // Initialize Tokenizer and Embedding layer
    let embedding = Embedding::new(vocab_size, EMBEDDING_SIZE); // Initialize embedding layer

    // Embed the input sentence
    let embeddings = embedding.forward(tokens.clone());

    // Convert embeddings to Array3 (batch_size, seq_length, embed_size)
    let input_tensor = Array3::from_shape_fn(
        (BATCH_SIZE, tokens.len(), EMBEDDING_SIZE),
        |(_, seq, embed)| embeddings[[seq, embed]],
    );

    // Initialize gamma and beta for layer normalization
    let gamma = Array2::ones((1, EMBEDDING_SIZE)); // Example gamma (scale parameter)
    let beta = Array2::zeros((1, EMBEDDING_SIZE)); // Example beta (shift parameter)

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
    let output_projection = &learnable_weights.output_projection; // All ones weights
    println!("Decoded shape: {:?}", decoded.dim());
    println!("Flattened decoded shape: {:?}", flatten_3d_array(decoded.clone()).dim());
    println!("Output projection shape: {:?}", output_projection.dim());
    println!("Transposed output projection shape: {:?}", output_projection.t().dim());


    let logits = flatten_3d_array(decoded).dot(&output_projection.to_owned()); // Linear layer

    // Apply softmax to logits
    let probabilities = softmax_matrix(&logits);

    // Convert probabilities back to text using the tokenizer
    let tokens = predict_index(probabilities.view(), &vocab);

    (tokens, logits.clone())
}
