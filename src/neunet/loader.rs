use nalgebra::*;
use std::io::*;

pub trait DataLoader {
    fn load_data(self, path: String) -> Result<DMatrix<u8>>;
}