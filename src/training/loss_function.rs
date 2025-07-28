use crate::attention::softmax::softmax_vector;
use ndarray::{s, Array1, Array2};

pub fn cross_entropy_loss(logits: &Array2<f32>, targets: &Array1<usize>, vocab_size: usize) -> f32 {
    let mut loss = 0.0;

    // Ensure that the number of targets matches the batch size
    assert_eq!(logits.dim().0, targets.len(), "Batch size mismatch");

    // Iterate over each target in the batch
    for (i, &target) in targets.iter().enumerate() {
        // Ensure target index is within valid range
        if target >= vocab_size {
            panic!(
                "Target index {target} is out of bounds for vocab_size {vocab_size}"
            );
        }

        // Get the logits for the current target (batch_size x vocab_size)
        let logit = &logits.slice(s![i, ..]); // Get the logits for the i-th sample

        // Softmax calculation: convert logits to probabilities
        let softmax = softmax_vector(*logit);

        // The log probability for the correct target token
        let log_prob = softmax[i];

        // Add to the loss: -log(p_y) for cross-entropy
        loss -= log_prob.ln();
    }

    loss
}
