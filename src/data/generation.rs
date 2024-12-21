use crate::data::io::get_input;
use crate::data::tokenizer::Tokenizer;

fn generate_input_target_pairs(
    tokenizer: &Tokenizer,
    sentences: Vec<String>,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut pairs = Vec::new();

    for i in 0..sentences.len() {
        let sentence = &sentences[i]; // Borrow the sentence
        let tokens = tokenizer.tokenize(sentence);

        // Prepare input (same as sentence)
        let input = tokens.clone();

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
        println!(
            "Input: {:?}\nTarget: {:?}\n",
            tokenizer.detokenize(input),
            tokenizer.detokenize(target)
        );
    }
}
