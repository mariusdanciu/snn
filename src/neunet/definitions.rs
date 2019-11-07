use std::f64::EPSILON;

use ndarray::Array1;

pub struct MLOps;

impl MLOps {
    fn hypothesis(&self, w: &Array1<f64>, x: &Array1<f64>, b: f64) -> f64 {
        w.dot(x) + b
    }

    fn sigmoid(&self, z: f64) -> f64 {
        1. / (1. + EPSILON.powf(-z))
    }

    fn loss(&self, y: f64, w: &Array1<f64>, x: &Array1<f64>, b: f64) -> f64 {
        let y_hat = self.sigmoid(self.hypothesis(w, x, b));
        - ((y * y_hat.ln()) + (1. - y)*(1. - y_hat).ln())
    }
}