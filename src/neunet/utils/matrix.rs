extern crate rand;
extern crate rand_pcg;

use nalgebra::*;
use rand::Rng;

pub struct MatrixUtil;

impl MatrixUtil {
    pub fn set_row<'a>(mat: &'a mut DMatrix<f32>, row_idx: usize, vec: &DVector<f32>) -> &'a DMatrix<f32> {
        let (rows, cols) = mat.shape();
        let mut i = row_idx;
        for e in vec.iter() {
            let pos = mat.index_mut(i);
            *pos = *e;
            i += rows;
        }
        mat
    }

    pub fn rand_matrix(r: usize, c: usize, rng: &mut rand_pcg::Pcg32) -> DMatrix<f32> {
        let factor = (2f32 / c as f32).sqrt();

        let v: Vec<f32> = (0..(r * c)).map(|_| rng.gen_range(-0.05, 0.05) * factor).collect();
        DMatrix::from_vec(r, c, v.clone())
    }

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
}

pub struct VectorUtil;

impl VectorUtil {
    pub fn max_index(v: &Vec<f32>) -> usize {
        let (mut i, mut max_idx) = (0, 0);

        let mut max = 0.0f32;

        for p in v {
            if *p > max {
                max = *p;
                max_idx = i;
            }
            i += 1;
        }

        max_idx
    }
}