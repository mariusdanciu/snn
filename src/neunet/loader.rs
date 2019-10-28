use ndarray::prelude::*;
use std::io::*;

pub trait DataLoader {
    fn load_data(self, path: String) -> Result<Array2<u8>>;
}