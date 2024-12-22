use crate::data::io::get_input;
use crate::data::tokenizer::Tokenizer;
use crate::settings::INPUT_SIZE;

pub fn generate_input_target_pairs(
    tokenizer: &Tokenizer,
    sentences: Vec<String>,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut pairs = Vec::new();

    for i in 0..sentences.len() {
        let sentence = &sentences[i]; // Borrow the sentence
        let tokens = tokenizer.tokenize(sentence);

        // Prepare input (same as sentence)
        let input = tokenizer.pad_sequence(tokens.clone(), INPUT_SIZE);

        // Prepare target (shifted version of the sentence)
        let mut target = tokens.clone();
        if i + 1 < sentences.len() {
            // If not the last sentence, append the first token of the next sentence
            let next_sentence = &sentences[i + 1];
            let next_tokens = tokenizer.tokenize(next_sentence);
            if !next_tokens.is_empty() {
                target.push(next_tokens[0]); // Add the first token of the next sentence
            }
        } else {
            target.push(tokenizer.vocab["<EOS>"]); // Use EOS token for the last sentence
        }

        // Remove the first token from target (shifting by 1)
        if !target.is_empty() {
            target.remove(0);
        }
        let target = tokenizer.pad_sequence(target, INPUT_SIZE);
        // Add the input-target pair to the result
        pairs.push((input, target));
    }

    pairs
}

// This will convert a list of sentences into tokenized input-output pairs
#[allow(dead_code)]
fn generate_input_target_pairs_by_sentence(
    tokenizer: &Tokenizer,
    sentences: Vec<&str>,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut pairs = Vec::new();

    for i in 0..sentences.len() - 1 {
        let input = tokenizer.tokenize(sentences[i]);
        let target = tokenizer.tokenize(sentences[i + 1]);

        pairs.push((input, target));
    }

    pairs
}

pub fn example_gen() {
    let raw_text = get_input();

    let tokenizer = Tokenizer::new(raw_text.clone());

    // Generate input-target pairs
    let pairs = generate_input_target_pairs(&tokenizer, raw_text);

    // Display the pairs
    for (input, target) in pairs.clone() {
        println!("Input: {:?}\nTarget: {:?}\n", input, target);
    }
    for (input, target) in pairs {
        let staircase_pairs = generate_staircase_pairs(&input, &target);

        for (staircase_input, staircase_target) in staircase_pairs {
            println!(
                "Input: {:?}\nTarget: {:?}\n",
                tokenizer.detokenize(staircase_input),
                tokenizer.detokenize(staircase_target)
            );
        }
    }
}
pub fn generate_staircase_pairs(
    input: &[usize],
    target: &[usize],
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut staircase_pairs = Vec::new();

    // The number of steps will be the length of the target sequence
    for i in 1..=target.len() {
        // Slice input and target incrementally
        let staircase_input = input.iter().take(i).cloned().collect::<Vec<usize>>();
        let staircase_target = target.iter().take(i).cloned().collect::<Vec<usize>>();

        // Pad both input and target sequences to max_length
        let staircase_input = pad_sequence_to_length(&staircase_input, INPUT_SIZE);
        let staircase_target = pad_sequence_to_length(&staircase_target, INPUT_SIZE);
        // Add this pair to the staircase pairs vector
        staircase_pairs.push((staircase_input, staircase_target));
    }

    staircase_pairs
}
fn pad_sequence_to_length(seq: &[usize], max_length: usize) -> Vec<usize> {
    let mut padded_seq = seq.to_vec();

    // Pad with <PAD> token if the sequence is shorter than max_length
    if padded_seq.len() < max_length {
        padded_seq.resize(max_length, 0); // 0 is the <PAD> token index
    } else if padded_seq.len() > max_length {
        // Truncate if the sequence is too long
        padded_seq.truncate(max_length);
    }

    padded_seq
}
