use crate::attention::softmax::softmax_vector;
use ndarray::{s, Array1, Array2};

pub fn cross_entropy_loss(
    logits: &Array2<f32>,
    targets: &Array1<usize>,
    _vocab_size: usize,
) -> f32 {
    let mut loss = 0.0;

    // Iterate over each target in the batch
    for (i, &target) in targets.iter().enumerate() {
        // Get the logits for the current target (batch_size x vocab_size)
        let logit = &logits.slice(s![i, ..]);

        // Softmax calculation
        let softmax = softmax_vector(*logit);

        // The log probability for the correct target token
        let log_prob = softmax[target];

        // Add to the loss: -log(p_y)
        loss -= log_prob.ln();
    }

    loss
}
