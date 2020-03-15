use std::io::*;

use nalgebra::*;

pub trait DataLoader {
    fn load_data(self, data_path: String, labels_path: String) -> Result<(DMatrix<u8>, DVector<u8>)>;
}

pub struct DataUtils;

impl DataUtils {
    pub fn one_hot(y: &DVector<u8>) -> DMatrix<f64> {
        let dim = *(y.data.as_vec().iter().max().unwrap()) as usize;
        let size = y.shape().0;

        let mut v = vec![0.0_f64; (dim + 1) * size];

        println!("v = {:?}", v);

        let mut pos = 0;
        for j in 0..size {
            let e = y[j] as usize;
            for i in 0..dim+1 {
                if i == e {
                    v[pos] = 1.0_f64;
                }
                pos += 1;
            }
        }
        println!("v = {:?}", v);
        DMatrix::from_vec(dim + 1, size, v)
    }
}