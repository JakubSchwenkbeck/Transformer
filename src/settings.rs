// Numerical constants with down scaled real-application values
#![allow(unused)]
pub const D_MODEL: usize = 512; // Embedding size
pub const D_K: usize = 64; // Key/query dimension
pub const D_V: usize = 64; // Value dimension
pub const NUM_HEADS: usize = 8; // Number of attention heads
pub const SEQ_LENGTH: usize = 128; // Sequence length
pub const BATCH_SIZE: usize = 32; // Batch size

pub const HIDDEN_SIZE: usize = 6;

pub const DROPOUT_RATE: f32 = 0.1; // Dropout rate for regularization
pub const LEARNING_RATE: f32 = 1e-4; // Optimizer learning rate

// Positional encoding parameters
pub const MAX_SEQ_LENGTH: usize = 512; // Maximum sequence length for positional encoding
