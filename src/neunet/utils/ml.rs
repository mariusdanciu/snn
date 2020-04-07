extern crate rand_pcg;

use nalgebra::{DVector, DVectorSlice};

pub struct MLOps;


impl MLOps {
    pub fn hypothesis(w: &DVector<f32>, x: &DVectorSlice<f32>, b: f32) -> f32 {
        w.dot(x) + b
    }

    pub fn apply(v: &DVector<f32>, f: fn(f32) -> f32) -> DVector<f32> {
        v.map(|e| f(e))
    }

    pub fn sigmoid(z: f32) -> f32 {
        1.0_f32 / (1.0_f32 + (-z).exp())
    }

    pub fn sigmoid_derivative(z: f32) -> f32 {
        let s = MLOps::sigmoid(z);
        s * (1.0_f32 - s)
    }

    pub fn relu(z: f32) -> f32 {
        z.max(0.0_f32)
    }
    pub fn relu_derivative(z: f32) -> f32 {
        if z >= 0.0_f32 {
            1.0_f32
        } else {
            0.0_f32
        }
    }

    pub fn tanh(z: f32) -> f32 {
        z.tanh()
    }
    pub fn tanh_derivative(z: f32) -> f32 {
        1.0_f32 - z.tanh().powi(2)
    }

    pub fn soft_max(v: &DVector<f32>) -> DVector<f32> {
        let mut sum = 0.0_f32;
        for e in v.iter() {
            sum += e.exp();
        }

        DVector::from_vec(v.data.as_vec().iter().map(|e| e.exp() / sum).collect())
    }

    pub fn soft_max_derivative(v: &DVector<f32>) -> DVector<f32> {
        let sm = MLOps::soft_max(&v);

        sm.map(|e| e * (1.0f32 - e))
    }


    pub fn binary_cross_entropy(y: f32, y_hat: f32) -> f32 {
        -(y * y_hat.ln() + (1. - y) * (1. - y_hat).ln())
    }

    pub fn cross_entropy(y: f32, y_hat: f32) -> f32 {
        -(y * (y_hat).ln())
    }

    pub fn cross_entropy_one_hot(y_v: &DVectorSlice<f32>, y_hat_v: &Vec<f32>) -> f32 {
        let mut sum = 0.0_f32;
        for e in 0..y_v.len() {
            sum += MLOps::cross_entropy(y_v[e], y_hat_v[e]);
        }
        sum
    }
}
