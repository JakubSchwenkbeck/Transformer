use regex::Regex;
use std::fs;

fn split_into_sentences(text: String) -> Vec<String> {
    let re = Regex::new(r"[.!?]").unwrap(); // Matches sentence-ending punctuation
    let mut sentences: Vec<String> = Vec::new(); // We want to store owned Strings, not &str

    let mut last_index = 0;
    for mat in re.find_iter(&text) {
        let end = mat.end();
        // Extract the sentence up to the matched punctuation
        let sentence = text[last_index..end].trim().to_string(); // Convert to String
        if !sentence.is_empty() {
            sentences.push(sentence);
        }
        last_index = end;
    }

    // Add any remaining text as a sentence
    if last_index < text.len() {
        let remaining = text[last_index..].trim().to_string(); // Convert remaining to String
        if !remaining.is_empty() {
            sentences.push(remaining);
        }
    }

    sentences
}

pub fn get_input() -> Vec<String> {
    let file_path = "src/data/in/training_input.txt";
    let content: String = fs::read_to_string(file_path).unwrap(); // Read the file content
    split_into_sentences(content) // Call the function to split into sentences
}
