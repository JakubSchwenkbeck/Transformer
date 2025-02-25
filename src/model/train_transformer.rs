use crate::attention::softmax::softmax_matrix;
use crate::data::dataset::{gen_data, Dataset};
use crate::data::learnable::LearnableWeights;
use crate::data::tokenizer::{tokenize_sentence, Tokenizer};
use crate::layers::feedforward_layer::FeedForwardLayer;
use crate::math::linear_algebra::{apply_projection, flatten_3d_array};
use crate::model::decoder::decoding;
use crate::model::embedding::{predict_index, Embedding};
use crate::model::encoder::encoding;
use crate::settings::*;
use crate::training::loss_function::cross_entropy_loss;
use crate::training::train::{compute_gradients, update_weights};
use ndarray::{Array1, Array2, Array3};
use rand::prelude::SliceRandom;
use std::collections::HashMap;

fn train_model(
    dataset: &Dataset,                       // The training data
    tokenizer: &Tokenizer,                   // Vocabulary
    mut learnable_weights: LearnableWeights, // Initial weights
    num_epochs: usize,                       // Number of training epochs
    learning_rate: f32,                      // Learning rate
) -> Vec<String> {
    let vocab_size = tokenizer.vocab.len(); // Vocabulary size
    let mut outputs = Vec::new(); // To store outputs for progress tracking
    let mut loss_history: Vec<f32> = Vec::new();
    // Loop over the number of epochs
    for epoch in 0..num_epochs {
        println!("\n=== Epoch {}/{} ===", epoch + 1, num_epochs);

        // Shuffle the dataset indices
        let mut data_indices: Vec<usize> = (0..dataset.inputs.len()).collect();
        data_indices.shuffle(&mut rand::rng());

        let mut total_loss = 0.0; // Accumulate loss for this epoch
        let mut num_batches = 0;

        // Loop over the training data
        for (step, &idx) in data_indices.iter().enumerate() {
            let input = &dataset.inputs[idx];
            let target = &dataset.targets[idx];

            // Convert to Array1 for processing
            let _input_seq = Array1::from(input.clone());
            let target_seq = Array1::from(target.clone());

            // Forward pass: Model prediction
            let (out, logits, prob) = training_model(
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

            // Log loss and progress every 100 steps
            if epoch % 100 == 0 {
                let decoded_output = tokenizer.detokenize(out.to_vec());
                let expected_output = tokenizer.detokenize(target.to_vec());
                println!(
                    "Step {}: Loss = {:.4}, Output = {:?}, Expected = {:?}",
                    step, loss, decoded_output, expected_output
                );
                outputs.push(decoded_output);
                let num_matches = out
                    .iter()
                    .zip(target.iter()) // Pair elements of out and target
                    .filter(|(o, t)| o == t) // Keep only the pairs where the elements are equal
                    .count(); // Count the number of matches

                // Calculate the percentage of matching elements
                let total_elements = out.len();
                let percentage = (num_matches as f32 / total_elements as f32) * 100.0;

                // Print the result
                println!("Percentage of equal elements: {:.2}%", percentage);
            }
            let inputs = Array3::from_shape_fn(
                (BATCH_SIZE, input.len(), EMBEDDING_SIZE),
                |(_, seq, embed)| logits[[seq, embed]],
            );

            // Now, targets should come from the actual target sequence (target_seq)
            let targets =
                Array2::from_shape_fn((target.len(), logits.shape()[1]), |(seq, _embed)| {
                    target_seq[seq] as f32 // Correctly use target_seq here
                });

            let _transformed: Array2<f32> = repeat_indices_as_array2(out);
            // Compute gradients
            let gradients = compute_gradients(&mut learnable_weights, &inputs, &targets, &prob);

            // println!("GRADIENTS: {:?}", gradients);
            // Update weights
            update_weights(&mut learnable_weights, &gradients, learning_rate);
            if epoch % 100 == 0 {
                // Log gradients for debugging (optional)
                println!("Step {}: Computed gradients = {:?}", step, gradients);

                // Periodically log weight updates (optional)
                println!(
                    "Step {}: Weights updated with learning rate = {:.6}",
                    step, learning_rate
                );
            }
        }

        // End of epoch: Print average loss and track improvement
        let avg_loss = total_loss / num_batches as f32;
        loss_history.push(avg_loss);

        println!(
            "Epoch {} completed with average loss: {:.4}",
            epoch + 1,
            avg_loss
        );
    }

    println!("\nTraining completed!");
    //plot_loss_progression(loss_history);
    training_ex(&mut learnable_weights);
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
    let num_epochs = 10000;
    let learning_rate = 0.00001;

    // Train the model
    let outputs = train_model(
        &dataset,
        &tokenizer,
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
    tokens: &[usize],                         // Input tokens
    _target_seq: Array1<usize>,               // Target sequence
    learnable_weights: &mut LearnableWeights, // Learnable weights
    vocab_size: usize,                        // Vocabulary size
    vocab: HashMap<String, usize>,            // Vocabulary map
) -> (Vec<usize>, Array2<f32>, Array2<f32>) {
    // Initialize Tokenizer and Embedding layer
    let embedding = Embedding::new(vocab_size, EMBEDDING_SIZE);

    // Embed the input sentence
    let embeddings = embedding.forward(tokens.to_vec());

    // Convert embeddings to Array3 (batch_size, seq_length, embed_size)
    let input_tensor = Array3::from_shape_fn(
        (BATCH_SIZE, tokens.len(), EMBEDDING_SIZE),
        |(_, seq, embed)| embeddings[[seq, embed]],
    );

    // Initialize gamma and beta for layer normalization
    let gamma = Array2::ones((1, EMBEDDING_SIZE));
    let beta = Array2::zeros((1, EMBEDDING_SIZE));

    // Initialize the feed-forward layer with correct types
    let feed_forward_layer = FeedForwardLayer::new(learnable_weights, DROPOUT_RATE);

    // Perform encoding with stacked layers
    let encoded = (0..NUM_LAYERS).fold(input_tensor.clone(), |acc, _| {
        encoding(
            acc,
            gamma.clone(),
            beta.clone(),
            EPSILON,
            &feed_forward_layer,
        )
    });

    // Perform decoding with stacked layers
    let decoded = (0..NUM_LAYERS).fold(input_tensor.clone(), |acc, _| {
        decoding(
            acc,
            encoded.clone(),
            gamma.clone(),
            beta.clone(),
            EPSILON,
            &feed_forward_layer,
        )
    });

    // Apply final linear transformation
    let logits = flatten_3d_array(apply_projection(
        &decoded,
        &learnable_weights.output_projection.to_owned(),
    ));
    // Apply softmax to logits
    let probabilities = softmax_matrix(&logits);

    // Convert probabilities back to text using the tokenizer
    let tokens = predict_index(probabilities.view(), &vocab);

    // Optionally print logits for debugging
    //println!("Logits: {:?}", logits);

    (tokens, logits, probabilities)
}

fn repeat_indices_as_array2(input: Vec<usize>) -> Array2<f32> {
    let repeat_count = input.len(); // The number of columns (same as the number of rows in the input)

    // Create a 2D array where each row is filled with the corresponding index from the input
    let data: Vec<Vec<f32>> = input
        .iter()
        .map(|&idx| vec![idx as f32; repeat_count])
        .collect();

    // Convert the Vec<Vec<usize>> into a 2D Array2
    Array2::from_shape_vec(
        (repeat_count, repeat_count),
        data.into_iter().flatten().collect(),
    )
    .unwrap()
}

fn training_ex(w: &mut LearnableWeights) {
    let vocab_size = 6;
    let tok = Tokenizer::new(tokenize_sentence("Once upon"));

    let tokens = tok.tokenize("Once");
    let target = tok.tokenize("Once upon");
    let (out, _, _) = training_model(
        &tokens,
        <Array1<usize>>::from(target.clone()),
        w,
        vocab_size,
        tok.vocab,
    );

    println!("Expected Output: {:?} \n Model Output: {:?}", target, out);
}
