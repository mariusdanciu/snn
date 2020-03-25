extern crate rand_pcg;

use nalgebra::{DMatrix, DVector, DVectorSlice};
use nalgebra::*;
use rand::Rng;

pub struct MLOps;


impl MLOps {
    pub fn hypothesis(w: &DVector<f64>, x: &DVectorSlice<f64>, b: f64) -> f64 {
        w.dot(x) + b
    }

    pub fn apply(v: &DVector<f64>, f: fn(f64) -> f64) -> DVector<f64> {
        v.map(|e| f(e))
    }

    pub fn sigmoid(z: f64) -> f64 {
        1.0_f64 / (1.0_f64 + (-z).exp())
    }

    pub fn sigmoid_derivative(z: f64) -> f64 {
        let s = MLOps::sigmoid(z);
        s * (1.0_f64 - s)
    }

    pub fn relu(z: f64) -> f64 {
        z.max(0.0_f64)
    }
    pub fn relu_derivative(z: f64) -> f64 {
        if z >= 0.0_f64 {
            1.0_f64
        } else {
            0.0_f64
        }
    }

    pub fn tanh(z: f64) -> f64 {
        z.tanh()
    }
    pub fn tanh_derivative(z: f64) -> f64 {
        1.0_f64 - z.tanh().powi(2)
    }

    pub fn soft_max(v: &DVector<f64>) -> DVector<f64> {
        let mut sum = 0.0_f64;
        for e in v.iter() {
            sum += e.exp();
        }

        DVector::from_vec(v.data.as_vec().iter().map(|e| e.exp()).collect()) / sum
    }

    pub fn soft_max_derivative(v: &DVector<f64>) -> DVector<f64> {
        let sm = MLOps::soft_max(&v);

        sm.map(|e| e * (1. - e))
    }

    pub fn loss_from_pred(y: f64, y_hat: f64) -> f64 {
        let loss = -(y * y_hat.ln() + (1. - y) * (1. - y_hat).ln());
        loss
    }

    pub fn loss_one_hot(y_v: &DVectorSlice<f64>, y_hat_v: &Vec<f64>) -> f64 {
        let mut sum = 0.0_f64;
        for e in 0..y_v.len() {
            sum += MLOps::loss_from_pred(y_v[e], y_hat_v[e]);
        }
        sum
    }
}
