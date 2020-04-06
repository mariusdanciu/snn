use nalgebra::*;

pub struct Normalizer;

impl Normalizer {
    pub fn min_max(features: &mut DMatrix<f32>) {
        let (rows, cols) = features.shape();

        for c in 0..cols {
            let col_vec = features.column(c);
            let min = col_vec.min();
            let max = col_vec.max();

            let mut index = c * rows;
            for r in 0..rows {
                if max > 0.0 {
                    let cell = features.index_mut(index);
                    *cell = (*cell - min) / { max - min };
                }
                index += 1;
            }
        }
    }
}