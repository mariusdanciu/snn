#[macro_use]
use ndarray::{arr2, Array1, Array2};
use rand::Rng;

use crate::neunet::definitions::MLOps;

trait Optimizer {
    fn optimize(&self,
                data: &Array2<f64>,
                labels: &Array1<f64>,
                learning_rate: f64) -> ();
}

struct StochasticGradientDescent;


impl Optimizer for StochasticGradientDescent {
    fn optimize(&self,
                data: &Array2<f64>,
                y: &Array1<f64>,
                learning_rate: f64) -> () {
        let mut r = rand::thread_rng();
        let shape = data.shape();

        let num_examples = shape[0];
        let num_features = shape[1];


        let mut w = Array1::from_vec(vec![0.; num_features]);
        let mut dw = vec![0.; num_features];
        let mut b = 0.;
        let mut db = 0.;
        let mut cost = 0.;

        let converged = false;

        let mut iteration = 0;
        while !converged {
            println!("SGD iteration {}", iteration);

            for i in 0..num_examples {
                let x_i = data.slice(s![i, 0..]);

                let z_i = MLOps.hypothesis(&w, &x_i.to_owned(), b);

                let y_hat_i = MLOps.sigmoid(z_i);

                cost += MLOps.loss_from_pred(y[i], y_hat_i);

                let dz_i = y_hat_i - y[i];

                for j in 0..num_features {
                    dw[j] += x_i[j] * dz_i;
                }

                db += dz_i;
            }

            println!("Loss {}", cost);
            for j in 0..num_features {
                dw[j] /= num_examples as f64;
                w[j] -= learning_rate * dw[j];
            }

            db /= num_examples as f64;

            cost /= num_examples as f64;

            b -= learning_rate * db;

            iteration += 1;
        }

        unimplemented!()
    }
}