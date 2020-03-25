extern crate rand;
extern crate rand_pcg;

use nalgebra::*;
use rand::Rng;

pub struct MatrixUtil;

impl MatrixUtil {
    pub fn set_row<'a>(mat: &'a mut DMatrix<f64>, row_idx: usize, vec: &DVector<f64>) -> &'a DMatrix<f64> {
        let (rows, cols) = mat.shape();
        let mut i = row_idx;
        for e in vec.iter() {
            let pos = mat.index_mut(i);
            *pos = *e;
            i += rows;
        }
        mat
    }

    pub fn rand_matrix(r: usize, c: usize, rng: &mut rand_pcg::Pcg32) -> DMatrix<f64> {
        let factor = (2f64 / c as f64).sqrt();

        let v: Vec<f64> = (0..(r * c)).map(|_| rng.gen_range(-0.05, 0.05) * factor).collect();
        DMatrix::from_vec(r, c, v.clone())
    }

    pub fn one_hot(y: &DVector<u8>) -> DMatrix<f64> {
        let dim = *(y.data.as_vec().iter().max().unwrap()) as usize;
        let size = y.shape().0;

        let mut v = vec![0.0_f64; (dim + 1) * size];

        let mut pos = 0;
        for j in 0..size {
            let e = y[j] as usize;
            v[pos + e] = 1.0_f64;
            pos += dim + 1;
        }
        DMatrix::from_vec(dim + 1, size, v)
    }
}

pub struct VectorUtil;

impl VectorUtil {
    pub fn max_index(v: &DVector<f64>) -> usize {
        let (mut i, mut max_idx) = (0, 0);

        let mut max = 0.0f64;

        for p in v.data.as_vec() {
            if *p > max {
                max = *p;
                max_idx = i;
            }
            i += 1;
        }

        max_idx
    }
}