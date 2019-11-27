use std::f64::EPSILON;

use nalgebra::{DMatrix, DVector, DVectorSlice};

pub struct MLOps;

impl MLOps {
    pub fn hypothesis(&self, w: &DVector<f64>, x: &DVectorSlice<f64>, b: f64) -> f64 {
        w.dot(x) + b
    }

    pub fn sigmoid_vec(&self, z: &DVector<f64>) -> DVector<f64> {
        z.map(|e| 1.0_f64 / (1.0_f64 + (-e).exp()))
    }

    pub fn sigmoid(&self, z: f64) -> f64 {
        1.0_f64 / (1.0_f64 + (-z).exp())
    }

    pub fn sigmoid_derivative(&self, z: f64) -> f64 {
        let s = self.sigmoid(z);
        s * (1.0_f64 - s)
    }
    pub fn sigmoid_vec_derivative(&self, z: &DVector<f64>) -> DVector<f64> {
        z.map(|e| self.sigmoid_derivative(e))
    }


    pub fn relu_vec(&self, z: &DVector<f64>) -> DVector<f64> {
        z.map(|e| e.max(0.0_f64))
    }
    pub fn relu(&self, z: f64) -> f64 {
        z.max(0.0_f64)
    }
    pub fn relu_derivative(&self, z: f64) -> f64 {
        if z >= 0.0_f64 {
            1.0_f64
        } else {
            0.0_f64
        }
    }
    pub fn relu_vec_derivative(&self, z: &DVector<f64>) -> DVector<f64> {
        z.map(|e| self.relu_derivative(e))
    }

    pub fn tanh(&self, z: f64) -> f64 {
        z.tanh()
    }
    pub fn tanh_derivative(&self, z: f64) -> f64 {
        1.0_f64 - z.tanh().powi(2)
    }
    pub fn tanh_vec_derivative(&self, z: &DVector<f64>) -> DVector<f64> {
        z.map(|e| self.tanh_derivative(e))
    }

    pub fn soft_max(&self, v: &DVector<f64>) -> DVector<f64> {
        let mut sum = 0.0_f64;
        for e in v.iter() {
            sum += e.exp();
        }

        v / sum
    }

    pub fn soft_max_derivative(&self, v: &DVector<f64>) -> DVector<f64> {
        let sm = self.soft_max(&v);

        sm.map(|e| e * (1. - e))
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
    tanh,
    soft_max,
}


pub struct Layer {
    pub intercepts: DVector<f64>,
    pub weights: DMatrix<f64>,
    pub outs: DVector<f64>,
    pub activation_type: ActivationType,
}

pub struct NeuralNetwork {
    pub layers: Vec<Layer>
}

