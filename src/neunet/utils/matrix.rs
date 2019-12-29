use nalgebra::*;

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
}