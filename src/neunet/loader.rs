use nalgebra::*;
use std::io::*;

pub trait DataLoader {
    fn load_data(self, data_path: String, labels_path: String) -> Result<(DMatrix<u8>, DVector<u8>)>;
}