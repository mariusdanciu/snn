use std::f64::EPSILON;

use nalgebra::{DMatrix, DVector, DVectorSlice};

pub struct MLOps;

impl MLOps {
    pub fn hypothesis(&self, w: &DVector<f64>, x: &DVectorSlice<f64>, b: f64) -> f64 {
        w.dot(x) + b
    }

    pub fn sigmoid(&self, z: f64) -> f64 {
        1. / (1. + EPSILON.powf(-z))
    }

    pub fn loss(&self, y: f64, w: &DVector<f64>, x: &DVectorSlice<f64>, b: f64) -> f64 {
        let y_hat = self.sigmoid(self.hypothesis(w, x, b));
        -(y * y_hat.ln() + (1. - y) * (1. - y_hat).ln())
    }

    pub fn loss_from_pred(&self, y: f64, y_hat: f64) -> f64 {
        -(y * y_hat.ln() + (1. - y) * (1. - y_hat).ln())
    }
}

pub enum OptimizationType {
    stochastic_gradient_descent,
    batch_gradient_descent,
    adam,
}

pub enum ActivationType {
    sigmoid,
    relu,
    soft_max,
}


pub struct Layer {
    pub intercepts: DVector<f64>,
    pub weights: DMatrix<f64>,
    pub activation_type: ActivationType,
}

pub struct NeuralNetwork {
    pub layers: Vec<Layer>
}

