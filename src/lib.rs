#![allow(non_snake_case)] // this lint makes a scene ....
pub mod transformer {
    pub fn train_model() {
        println!("Training the Transformer!");
    }
}

pub mod utils {
    pub mod linear_algebra;
    pub mod activation;
}