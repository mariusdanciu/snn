#![allow(dead_code)]
#![allow(unused)]

use nalgebra::*;

use crate::neunet::api::defs::*;
use crate::neunet::utils::ml::*;

pub fn momentum(iteration: u32, momentum_beta: f32, learning_rate: f32, nn: &mut NNModel) {
    if iteration > 0 {
        // Apply  weighted moving averages
        for mut l in nn.layers.iter_mut() {
            l.momentum_dw = momentum_beta * &l.momentum_dw + (1. - momentum_beta) * &l.dw;
            l.momentum_db = momentum_beta * &l.momentum_db + (1. - momentum_beta) * &l.db;
        }
    } else {
        for mut l in nn.layers.iter_mut() {
            let (r, c) = l.dw.shape();
            l.momentum_dw = DMatrix::from_fn(r, c, |_, _| 0.0);
            l.momentum_db = DVector::from_vec(vec![0.0; r]);
        }
    }

    for mut l in nn.layers.iter_mut() {
        l.weights = &l.weights - learning_rate * &l.momentum_dw;
        l.intercepts = &l.intercepts - learning_rate * &l.momentum_db;
    }
}

pub fn rms_prop(iteration: u32, rms_prop_beta: f32, learning_rate: f32, nn: &mut NNModel) {
    if iteration > 0 {
        // Apply  weighted moving averages
        for mut l in nn.layers.iter_mut() {
            l.rmsp_dw = rms_prop_beta * &l.rmsp_dw + (1. - rms_prop_beta) * (&l.dw.component_mul(&l.dw));
            l.rmsp_db = rms_prop_beta * &l.rmsp_db + (1. - rms_prop_beta) * (&l.db);
        }
    } else {
        for mut l in nn.layers.iter_mut() {
            let (r, c) = l.dw.shape();
            l.rmsp_dw = DMatrix::from_fn(r, c, |_, _| 0.0);
            l.rmsp_db = DVector::from_vec(vec![0.0; r]);
        }
    }

    let rmsprop_div: fn(f32) -> f32 = |e| e.sqrt() + 1e-8;

    for mut l in nn.layers.iter_mut() {
        l.weights = &l.weights - learning_rate * &l.dw.component_div(&apply_to_mat(&l.rmsp_dw, rmsprop_div));
        l.intercepts = &l.intercepts - learning_rate * &l.db.component_div(&apply_to_vec(&l.rmsp_db, rmsprop_div));
    }
}

pub fn adam(iteration: u32, momentum_beta: f32, rms_prop_beta: f32, learning_rate: f32, nn: &mut NNModel) {
    if iteration > 0 {
        // Apply  weighted moving averages
        for mut l in nn.layers.iter_mut() {
            l.momentum_dw = momentum_beta * &l.momentum_dw + (1. - momentum_beta) * &l.dw;
            l.momentum_db = momentum_beta * &l.momentum_db + (1. - momentum_beta) * &l.db;

            l.rmsp_dw = rms_prop_beta * &l.rmsp_dw + (1. - rms_prop_beta) * (&l.dw.component_mul(&l.dw));
            l.rmsp_db = rms_prop_beta * &l.rmsp_db + (1. - rms_prop_beta) * (&l.db.component_mul(&l.db));
        }
    } else {
        for mut l in nn.layers.iter_mut() {
            let (r, c) = l.dw.shape();
            l.momentum_dw = DMatrix::from_fn(r, c, |_, _| 0.0);
            l.momentum_db = DVector::from_vec(vec![0.0; r]);

            l.rmsp_dw = DMatrix::from_fn(r, c, |_, _| 0.0);
            l.rmsp_db = DVector::from_vec(vec![0.0; r]);
        }
    }

    let rmsprop_div: fn(f32) -> f32 = |e| {
        e.sqrt() + 1e-8
    };

    for mut l in nn.layers.iter_mut() {
        let momentum_dw_corrected = &l.momentum_dw / (1f32 - momentum_beta.powi(1 + iteration as i32));
        let momentum_db_corrected = &l.momentum_db / (1f32 - momentum_beta.powi(1 + iteration as i32));

        let rms_dw_corrected = &l.rmsp_dw / (1f32 - rms_prop_beta.powi(1 + iteration as i32));
        let rms_db_corrected = &l.rmsp_db / (1f32 - rms_prop_beta.powi(1 + iteration as i32));


        let dw = &momentum_dw_corrected.component_div(&apply_to_mat(&rms_dw_corrected, rmsprop_div));
        let db = &momentum_db_corrected.component_div(&apply_to_vec(&rms_db_corrected, rmsprop_div));

        l.weights = &l.weights - learning_rate * dw;
        l.intercepts = &l.intercepts - learning_rate * db;
    }
}
