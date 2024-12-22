use crate::data::generation::{generate_input_target_pairs, generate_staircase_pairs};
use crate::data::io::get_input;
use crate::data::tokenizer::Tokenizer;

pub struct Dataset {
    pub(crate) inputs: Vec<Vec<usize>>, // Each input is a sequence of token IDs
    pub(crate) targets: Vec<Vec<usize>>, // Each target is the corresponding output sequence
}

pub fn gen_data() -> (Tokenizer, Dataset) {
    let raw_text = get_input();

    let tokenizer = Tokenizer::new(raw_text.clone());

    // Generate input-target pairs
    let pairs = generate_input_target_pairs(&tokenizer, raw_text);

    let mut all_inputs = Vec::new();
    let mut all_targets = Vec::new();

    // For each input-target pair, generate staircase pairs and add to the dataset
    for (input, target) in pairs {
        let staircase_pairs = generate_staircase_pairs(&input, &target);

        // Add the staircase pairs to the dataset
        for (staircase_input, staircase_target) in staircase_pairs {
            all_inputs.push(staircase_input);
            all_targets.push(staircase_target);
        }
    }

    // Return tokenizer and the generated dataset
    (
        tokenizer,
        Dataset {
            inputs: all_inputs,
            targets: all_targets,
        },
    )
}
