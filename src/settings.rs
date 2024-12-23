#![allow(unused)]

// Numerical constants with downscaled real-application values
pub const EPSILON: f32 = 0.0001;

// Embedding size
pub const D_MODEL: usize = 6; // Model embedding size, matching the vocab size

// Attention mechanism dimensions
pub const D_K: usize = 32; // Key/query dimension (same as D_V for simplicity)
pub const D_V: usize = 32; // Value dimension (same as D_K)
pub const NUM_HEADS: usize = 1; // Reduced the number of attention heads for smaller model

// Sequence and batch size
pub const SEQ_LENGTH: usize = 64; // Reduced sequence length
pub const BATCH_SIZE: usize = 1; // Reduced batch size for smaller model training

// Embedding size and dimensions
pub const EMBEDDING_SIZE: usize = D_MODEL; // Matches D_MODEL for consistency

// Input/Output sizes
pub const INPUT_SIZE: usize = D_MODEL; // Typically equals D_MODEL for transformer inputs
pub const OUTPUT_SIZE: usize = D_MODEL; // Consistent with D_MODEL for output

// Number of layers
pub const NUM_LAYERS: usize = 4; // Reduced to 4 layers for a smaller architecture

// Feedforward network dimension (FFN_DIM)
pub const FFN_DIM: usize = 128; // Smaller FFN dimension

// Hidden size (used for biases and other layer parameters)
pub const HIDDEN_SIZE: usize = 6; // Adjusted for a smaller hidden layer size, consistent with FFN_DIM

// Dropout rate and learning rate
pub const DROPOUT_RATE: f32 = 0.1; // Dropout rate for regularization
pub const LEARNING_RATE: f32 = 1e-4; // Optimizer learning rate

// Positional encoding parameters
pub const MAX_SEQ_LENGTH: usize = 128; // Maximum sequence length for positional encoding
