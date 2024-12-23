use Transformer::data::tokenizer::Tokenizer;
use Transformer::example::example;
use Transformer::model::train_transformer::train;
use Transformer::model::transformer_model::transformer_model;
fn main() {
    println!("runs successfully!");
    println!("============= ATTENTION WEIGHT EXAMPLE =============");
   // example();

    /*println!(" \n \n \n ENCODER/DECODER  \n");

    let input: Vec<String> = vec!["hello world rust transformer learning model"
        .parse()
        .unwrap()];

    // Initialize Tokenizer and Embedding layer
    let tokenizer = Tokenizer::new(input.clone());
    let embedding = Embedding::new(tokenizer.vocab.len(), 12); // Example vocab size and embedding size
                                                               // Input sentence
    let sentence = "hello world rust";

    // Tokenize and embed the input
    let tokens = tokenizer.tokenize(sentence);
    let embeddings = embedding.forward(tokens.clone());
    println!("embeddings: {:?}", embeddings);

    let input_tensor = Array3::from_shape_fn(
        (BATCH_SIZE, tokens.len(), EMBEDDING_SIZE),
        |(_, seq, embed)| embeddings[[seq, embed]],
    );

    println!("INPUT : {}", input_tensor.clone());
    // Initialize gamma and beta for layer normalization
    let gamma = Array2::ones((1, 12)); // Example gamma (scale parameter)
    let beta = Array2::zeros((1, 12)); // Example beta (shift parameter)

    // Initialize the feed-forward layer with correct types
    let learnable_weights = initialize_weights();
    let feed_forward_layer = FeedForwardLayer::new(&learnable_weights, DROPOUT_RATE);

    // Perform encoding (transformer layer)
    let epsilon = 1e-6; // Small epsilon for numerical stability
    let encoded = encoding(
        input_tensor.clone(),
        gamma.clone(),
        beta.clone(),
        epsilon,
        &feed_forward_layer,
    );
    // Perform decoding (transformer layer)
    let decoded = decoding(
        input_tensor,
        encoded.clone(),
        gamma,
        beta,
        epsilon,
        &feed_forward_layer,
    );

    // Print the encoded and decoded output tensors
    println!("Encoded: {:?}", encoded);
    println!("Decoded: {:?}", decoded);

    let tokens = embedding.retrieve_tokens(flatten_3d_array(decoded), &tokenizer.vocab);

    println!("Tokens: {:?}", tokens);

    */
    /*let sentence = "hello rust world";
    let input: Vec<String> = vec!["Hello rust world my name".parse().unwrap()];
    let tokenizer = Tokenizer::new(input);
    let predicted_token = transformer_model(sentence, tokenizer);

    println!("Predicted Token: {:?}", predicted_token);


     */
    //example_gen();

    train()
}
