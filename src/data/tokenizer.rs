#![allow(warnings)]

use regex::Regex;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    reverse_vocab: HashMap<usize, String>,
    pad_token: String,
    sos_token: String,
    eos_token: String,
    unk_token: String,
}

impl Tokenizer {
    pub fn new(vocab: HashMap<String, usize>) -> Self {
        // Define special tokens
        let pad_token = "<PAD>".to_string();
        let sos_token = "<SOS>".to_string();
        let eos_token = "<EOS>".to_string();
        let unk_token = "<UNK>".to_string();

        // Add special tokens to vocabulary
        let mut extended_vocab = vocab.clone();
        extended_vocab.insert(pad_token.clone(), 0);
        extended_vocab.insert(sos_token.clone(), 1);
        extended_vocab.insert(eos_token.clone(), 2);
        extended_vocab.insert(unk_token.clone(), 3);

        // Reverse vocabulary for decoding
        let reverse_vocab: HashMap<usize, String> = extended_vocab
            .iter()
            .map(|(word, idx)| (idx.clone(), word.clone()))
            .collect();

        Tokenizer {
            vocab: extended_vocab,
            reverse_vocab,
            pad_token,
            sos_token,
            eos_token,
            unk_token,
        }
    }

    // Tokenize input sentence (word-based tokenization)
    pub fn tokenize(&self, sentence: &str) -> Vec<usize> {
        let words = self.tokenize_sentence(sentence);
        let mut tokens: Vec<usize> = vec![self.vocab[&self.sos_token]]; // Start with SOS token

        for word in words {
            let token = self
                .vocab
                .get(&word)
                .unwrap_or(&self.vocab[&self.unk_token]);
            tokens.push(*token);
        }

        tokens.push(self.vocab[&self.eos_token]); // Add EOS token at the end
        tokens
    }

    // Convert indices back to words
    pub fn detokenize(&self, tokens: Vec<usize>) -> String {
        tokens
            .iter()
            .filter_map(|&token| self.reverse_vocab.get(&token))
            .map(|word| word.to_string()) // Convert each reference to a String
            .collect::<Vec<String>>()
            .join(" ")
    }

    // Helper function to split sentence into words using a simple regex
    fn tokenize_sentence(&self, sentence: &str) -> Vec<String> {
        let re = Regex::new(r"\w+").unwrap(); // Matches words (letters and numbers)
        re.find_iter(sentence)
            .map(|mat| mat.as_str().to_string())
            .collect()
    }
}

pub fn example_tokens() {
    // Define a small vocabulary (for example purposes)
    let vocab = vec![
        ("hello".to_string(), 4),
        ("world".to_string(), 5),
        ("my".to_string(), 6),
        ("name".to_string(), 7),
        ("is".to_string(), 8),
    ]
    .into_iter()
    .collect::<HashMap<String, usize>>();

    // Instantiate the tokenizer with the vocabulary
    let tokenizer = Tokenizer::new(vocab);

    // Example sentence
    let sentence = "hello world";

    // Tokenize the sentence
    let tokens = tokenizer.tokenize(sentence);
    println!("Tokens: {:?}", tokens); // Should print indices for "hello", "world", etc.

    // Detokenize the sentence
    let decoded_sentence = tokenizer.detokenize(tokens);
    println!("Decoded Sentence: {}", decoded_sentence); // Should print "hello world"
}
