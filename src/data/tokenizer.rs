#![allow(warnings)]

use regex::Regex;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub struct Tokenizer {
    pub vocab: HashMap<String, usize>,
    reverse_vocab: HashMap<usize, String>,
    pad_token: String,
    sos_token: String,
    eos_token: String,
    unk_token: String,
}

impl Tokenizer {
    pub fn new(input: Vec<String>) -> Self {
        let vocab: HashMap<String, usize> = generate_vocab(input);
        println!("size : {:?}", vocab.clone().len());

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
        let words = tokenize_sentence(sentence);
        let mut tokens: Vec<usize> = vec![self.vocab[&self.sos_token]]; // Start with SOS token

        for word in words {
            let word_lower = word.to_lowercase(); // Convert to lowercase
            let token = self.vocab.get(&word_lower).unwrap_or_else(|| {
                eprintln!(
                    "Warning: Word '{}' not in vocabulary, substituting <UNK>",
                    word
                );
                &self.vocab[&self.unk_token]
            });
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

    // Helper function to split sentence into words using an improved regex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer() {
        let input: Vec<String> = vec!["Hello world, my".parse().unwrap()];
        let tokenizer = Tokenizer::new(input);

        // Empty sentence
        let tokens = tokenizer.tokenize("");
        assert_eq!(tokens, vec![1, 2]); // <SOS>, <EOS>

        // Sentence with punctuation
        let tokens = tokenizer.tokenize("hello, world!");
        // Detokenization
        let decoded_sentence = tokenizer.detokenize(tokens.clone());
        assert_eq!(decoded_sentence, "<SOS> hello , world <UNK> <EOS>");
    }
}

pub fn example_tokens() {
    // Define a small sentence
    let input: Vec<String> = vec!["Hello, world!".parse().unwrap()];

    // Instantiate the tokenizer with the vocabulary
    let tokenizer = Tokenizer::new(input);

    // Example sentence
    let sentence = "Hello, world! My name is ChatGPT.";

    // Tokenize the sentence
    let tokens = tokenizer.tokenize(sentence);
    println!("Tokens: {:?}", tokens); // Should print indices for "hello", "world", etc.

    // Detokenize the sentence
    let decoded_sentence = tokenizer.detokenize(tokens);
    println!("Decoded Sentence: {}", decoded_sentence); // Should print the sequence with special tokens
}

pub fn generate_vocab(text: Vec<String>) -> HashMap<String, usize> {
    let mut word_set = HashSet::new();

    // Tokenize each sentence and collect words, treating punctuation separately
    for sentence in text {
        let words = tokenize_sentence(&sentence); // Updated tokenization method
        for word in words {
            word_set.insert(word.to_lowercase());
        }
    }

    // Create vocabulary by assigning a unique index to each word
    let vocab = word_set
        .into_iter()
        .enumerate()
        .map(|(idx, word)| (word, idx + 4)) // Start index from 4 to leave space for special tokens
        .collect::<HashMap<String, usize>>();

    vocab
}

pub fn tokenize_sentence(sentence: &str) -> Vec<String> {
    let re = Regex::new(r"\w+|[^\w\s]").unwrap(); // Matches words or punctuation
    re.find_iter(sentence)
        .map(|mat| mat.as_str().to_string())
        .collect()
}
