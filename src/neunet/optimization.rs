#![feature(nll)]

use nalgebra::*;

use crate::neunet::definitions::{ActivationType, MLOps, NeuralNetwork};

trait Optimizer {
    fn optimize(&self,
                data: &DMatrix<f64>,
                labels: &DVector<f64>) -> ();
}

struct StochasticGradientDescent {
    pub learning_rate: f64,
    // 0.0001
    pub stop_cost_quota: f64,
    // 10 ^ -4
    pub network: NeuralNetwork,
}

trait ForwardProp {
    fn forward_prop(&mut self, inputs: &DVector<f64>) -> DVector<f64>;
}

struct BackPropOut {
    weights: DVector<f64>,
    intercepts: DVector<f64>,
}

trait BackProp {
    fn back_prop(&mut self, x: &DVector<f64>, y_hat: &DVector<f64>, y: &DVector<f64>) -> BackPropOut;
}

impl ForwardProp for NeuralNetwork {
    fn forward_prop(&mut self, inputs: &DVector<f64>) -> DVector<f64> {
        let mut current = inputs;
        let mut t;

        for mut l in &mut self.layers {
            let z = &l.weights * current + &l.intercepts;
            t = match &l.activation_type {
                ActivationType::Sigmoid =>
                    MLOps::vectorize(&z, MLOps::sigmoid),
                ActivationType::Relu =>
                    MLOps::vectorize(&z, MLOps::relu),
                ActivationType::Tanh =>
                    MLOps::vectorize(&z, MLOps::tanh),
                ActivationType::SoftMax =>
                    MLOps::soft_max(&z)
            };
            l.z = z;
            l.a = t.clone();
            current = &t;
        };

        current.clone()
    }
}

impl BackProp for NeuralNetwork {
    fn back_prop(&mut self, x: &DVector<f64>, y_hat: &DVector<f64>, y: &DVector<f64>) -> BackPropOut {
        let l = &mut self.layers;
        let mut idx = l.len() - 1;

        l[idx].dz = y_hat - y;
        l[idx].dw = &l[idx].dz * y_hat;
        l[idx].db = l[idx].dz.clone();

        idx -= 1;
        while idx >= 0_usize {
            let t = match &l[idx].activation_type {
                ActivationType::Relu => MLOps::vectorize(&l[idx].z, MLOps::relu_derivative),
                ActivationType::Sigmoid => MLOps::vectorize(&l[idx].z, MLOps::sigmoid_derivative),
                ActivationType::Tanh => MLOps::vectorize(&l[idx].z, MLOps::tanh_derivative),
                ActivationType::SoftMax => MLOps::soft_max_derivative(&l[idx].z),
            };

            l[idx].dz = (&l[idx + 1].dz * &l[idx + 1].weights).component_mul(&t);
            l[idx].dw = if idx > 0 {
                &l[idx].dz * &l[idx - 1].a
            } else {
                &l[idx].dz * x
            };

            l[idx].db = l[idx].dz.clone();

            idx -= 1;
        }

        /*
                let mut dJ = inputs - labels;

                for k in (0..idx - 1).rev() {
                    let t = match &self.layers[k].activation_type {
                        relu => MLOps::vectorize(&self.layers[k].outs, MLOps::relu_derivative),
                        sigmoid => MLOps::vectorize(&self.layers[k].outs, MLOps::sigmoid_derivative),
                        tanh => MLOps::vectorize(&self.layers[k].outs, MLOps::tanh_derivative),
                        soft_max => MLOps::soft_max_derivative(&self.layers[k].outs),
                    };

                    dJ = (&self.layers[k].weights * dJ).component_mul(&t);

                    current = &self.layers[k];
                }
                */
        unimplemented!()
    }
}

impl Optimizer for StochasticGradientDescent {
    ///
    ///
    ///
    fn optimize(&self,
                data: &DMatrix<f64>,
                y: &DVector<f64>) -> () {
        fn forward_prop(features: &DVectorSlice<f64>, w: &DVector<f64>, b: f64, activation_type: ActivationType) -> f64 {
            let z_i = MLOps::hypothesis(&w, &features, b);

            match activation_type {
                ActivationType::Sigmoid => MLOps::sigmoid(z_i),
                _ => MLOps::sigmoid(z_i)
            }
        }

        fn back_prop(y_hat: f64,
                     y: f64,
                     x: &DVectorSlice<f64>,
                     dw: &mut DVector<f64>,
                     db: &mut f64) -> () {
            let dz_i = y_hat - y;

            for j in 0..dw.len() {
                dw[j] += x[j] * dz_i;
            }

            *db += dz_i;
        }

        let shape = data.shape();

        let num_examples = shape.0;
        let num_features = shape.1;


        let mut w = DVector::from_vec(vec![0.; num_features]);
        let mut dw = DVector::from_vec(vec![0.; num_features]);

        let mut b = 0.;
        let mut db = 0.;
        let mut cost = 0.;

        let mut converged = false;

        let mut iteration = 0;
        while !converged {
            println!("SGD iteration {}", iteration);

            for i in 0..num_examples {
                let x_i = data.column(i);

                let y_hat_i = forward_prop(&x_i, &w, b, ActivationType::Sigmoid);

                cost += MLOps::loss_from_pred(y[i], y_hat_i);

                back_prop(y_hat_i, y[i], &x_i, &mut dw, &mut db);
            }

            println!("Cost {}", cost);

            for j in 0..num_features {
                dw[j] /= num_examples as f64;
                w[j] -= self.learning_rate * dw[j];
            }

            db /= num_examples as f64;

            cost /= num_examples as f64;

            b -= self.learning_rate * db;

            iteration += 1;

            converged = cost < self.stop_cost_quota;
        }

        unimplemented!()
    }
}


