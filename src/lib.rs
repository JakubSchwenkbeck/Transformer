#![allow(non_snake_case)]

pub mod example;
mod settings;

pub mod model {
    pub mod decoder;
    pub mod encoder;

    pub mod stacked;
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
