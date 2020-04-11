extern crate rand;
extern crate rand_pcg;

use nalgebra::*;

pub fn one_hot(y: &DVector<u8>) -> DMatrix<f32> {
    let num_classes = *(y.data.as_vec().iter().max().unwrap()) as usize;
    let size = y.shape().0;

    let mut v = vec![0.0_f32; (num_classes + 1) * size];

    let mut pos = 0;
    for j in 0..size {
        let e = y[j] as usize;
        v[pos + e] = 1.0_f32;
        pos += num_classes + 1;
    }
    DMatrix::from_vec(num_classes + 1, size, v)
}

