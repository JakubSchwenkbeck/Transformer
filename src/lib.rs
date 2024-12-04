#![allow(non_snake_case)] // this lint makes a scene ....
pub mod transformer {
    pub fn train_model() {
        println!("Training the Transformer!");
    }
}

pub mod math {
    pub mod activation;
    pub mod linear_algebra;
    pub mod softmax;

    pub mod normalization;

    pub mod positional_encoding;
}
