#![allow(unused)]

// Numerical constants with down scaled real-application values
pub const EPSILON: f32 = 0.0001;

// Embedding size
pub const D_MODEL: usize = 64; // Changed from 12 to a larger, more standard dimension for transformers

// Attention mechanism dimensions
pub const D_K: usize = 64; // Key/query dimension (same as D_V for simplicity)
pub const D_V: usize = 64; // Value dimension (same as D_K)
pub const NUM_HEADS: usize = 8; // Number of attention heads

// Sequence and batch size
pub const SEQ_LENGTH: usize = 128; // Sequence length (adjustable depending on your data)
pub const BATCH_SIZE: usize = 32; // Increased batch size for practical usage

// Embedding size and dimensions
pub const EMBEDDING_SIZE: usize = D_MODEL; // Should match D_MODEL for consistency

// Input/Output sizes
pub const INPUT_SIZE: usize = D_MODEL; // Typically equals D_MODEL for transformer inputs
pub const OUTPUT_SIZE: usize = D_MODEL; // Should be consistent with D_MODEL for output

// Number of layers
pub const NUM_LAYERS: usize = 6; // Number of layers (standard for many transformer architectures)

// Feedforward network dimension (FFN_DIM)
pub const FFN_DIM: usize = 256; // A common size for the feedforward dimension

// Hidden size (used for biases and other layer parameters)
pub const HIDDEN_SIZE: usize = 256; // Adjusted for a larger hidden layer size, consistent with FFN_DIM

// Dropout rate and learning rate
pub const DROPOUT_RATE: f32 = 0.1; // Dropout rate for regularization
pub const LEARNING_RATE: f32 = 1e-4; // Optimizer learning rate

// Positional encoding parameters
pub const MAX_SEQ_LENGTH: usize = 512; // Maximum sequence length for positional encoding
