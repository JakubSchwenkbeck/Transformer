use crate::data::tokenizer::Tokenizer;

fn generate_input_target_pairs(
    tokenizer: &Tokenizer,
    sentences: Vec<&str>,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut pairs = Vec::new();

    for i in 0..sentences.len() {
        let sentence = sentences[i];
        let tokens = tokenizer.tokenize(sentence);

        // Prepare input (same as sentence)
        let input = tokens.clone();

        // Prepare target (shifted version of the sentence)
        let mut target = tokens.clone();
        if i + 1 < sentences.len() {
            // If not the last sentence, the last word will be the first word of the next sentence
            target.push(tokenizer.tokenize(sentences[i + 1])[0]); // First token of the next sentence
        } else {
            target.push(tokenizer.vocab["<EOS>"]); // Use EOS token for the last word
        }

        // Remove the first word from target (shift by 1)
        target.remove(0);

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
    let raw_text = vec![
        "Once upon a time, in a land far away, there was a small village.",
        "The villagers were known for their kindness and generosity.",
        "Every year, they celebrated the harvest festival with music, dance, and delicious food.",
        "One day, a traveler came to the village.",
        "He was tired and hungry, but the villagers welcomed him with open arms.",
        "The traveler shared stories of his adventures as the villagers listened intently.",
        "He told them about distant lands and strange creatures.",
        "The villagers were fascinated by his tales.",
        "As the evening drew to a close, the traveler offered to leave the village, but the villagers insisted he stay for another night.",
        "The next morning, the traveler said goodbye and continued his journey.",
        "The villagers waved him off, grateful for the stories and the company.",
    ];

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
