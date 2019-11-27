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
    fn back_prop(&mut self, inputs: DVector<f64>, labels: DVector<f64>) -> BackPropOut;
}

impl ForwardProp for NeuralNetwork {
    fn forward_prop(&mut self, inputs: &DVector<f64>) -> DVector<f64> {
        let mut current = inputs;
        let mut k: DVector<f64>;


        for mut l in &mut self.layers {
            let t = &(&l.weights * current + &l.intercepts);
            match &l.activation_type {
                sigmoid =>
                    k = MLOps.sigmoid_vec(t),
                relu =>
                    k = MLOps.relu_vec(t),
                soft_max =>
                    k = MLOps.soft_max(t)
            };
            l.outs = k;
            current = &l.outs;
        };

        current.clone()
    }
}

impl BackProp for NeuralNetwork {
    fn back_prop(&mut self, inputs: DVector<f64>, labels: DVector<f64>) -> BackPropOut {
        let len = self.layers.len();
        let mut current = &self.layers[len - 1];

        let mut dJ = inputs - labels;

        for k in (0..len - 1).rev() {
            let t = match &self.layers[k].activation_type {
                relu => MLOps.relu_vec_derivative(&self.layers[k].outs),
                sigmoid => MLOps.sigmoid_vec_derivative(&self.layers[k].outs),
                tanh => MLOps.tanh_vec_derivative(&self.layers[k].outs),
                soft_max => MLOps.soft_max_derivative(&self.layers[k].outs),
            };

            dJ = &self.layers[k].weights*dJ * t;

            current = &self.layers[k];
        }
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
            let z_i = MLOps.hypothesis(&w, &features, b);

            match activation_type {
                ActivationType::sigmoid => MLOps.sigmoid(z_i),
                _ => MLOps.sigmoid(z_i)
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

                let y_hat_i = forward_prop(&x_i, &w, b, ActivationType::sigmoid);

                cost += MLOps.loss_from_pred(y[i], y_hat_i);

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