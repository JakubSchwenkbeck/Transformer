#![allow(non_snake_case)]

pub mod example;
pub mod settings;

pub mod data {
    pub mod dataset;
    pub mod generation;
    pub mod io;
    pub mod learnable;
    pub mod tokenizer;
}
pub mod model {
    pub mod decoder;
    pub mod embedding;
    pub mod encoder;
    pub mod transformer_model;
}

// this lint makes a scene ....
pub mod transformer {
    pub fn train_model() {
        println!("Training the Transformer!");
    }
}

pub mod math {

    pub mod linear_algebra;

    pub mod positional_encoding;
}

pub mod activation {
    pub mod activation_functions;
}
pub mod attention {
    pub mod multihead_attention;
    pub mod scaled_dot_attention;
    pub mod softmax;
}

pub mod layers {
    pub mod feedforward_layer;
    pub mod normalization;
}
