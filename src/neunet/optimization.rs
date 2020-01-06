use nalgebra::*;

use crate::neunet::definitions::{ActivationType, MLOps, NeuralNetworkDefinition, NNModel};
use crate::neunet::utils::matrix::MatrixUtil;

trait Optimizer {
    fn optimize(&self,
                nn: &mut NeuralNetwork,
                data: &DMatrix<f64>,
                labels: &DVector<f64>) -> ();
}

struct GradientDescent {
    pub momentum_beta: f64,
    pub rms_prop_beta: f64,
    pub epsilon_correction: f64,
    pub mini_batch_size: usize,
    pub learning_rate: f64,
    // 0.0001
    pub stop_cost_quota: f64,
    // 10 ^ -4
    pub network: NeuralNetwork,
}

struct Layer {
    num_activations: usize,
    intercepts: DVector<f64>,
    weights: DMatrix<f64>,
    activation_type: ActivationType,

    // W * X + B
    z: DVector<f64>,
    // activation(z)
    a: DVector<f64>,
    dz: DVector<f64>,
    dw: DMatrix<f64>,
    db: DVector<f64>,

    momentum_dw: DMatrix<f64>,
    momentum_db: DVector<f64>,
}

struct NeuralNetwork {
    layers: Vec<Layer>
}

trait ForwardProp {
    fn forward_prop(&mut self) -> DVector<f64>;
}


trait BackProp {
    fn back_prop(&mut self, example_idx: usize, y_hat: &DVector<f64>, y: &DVector<f64>) -> &Self;
}

impl ForwardProp for NeuralNetwork {
    fn forward_prop(&mut self) -> DVector<f64> {
        let mut current = &self.layers[0].a;
        let mut t;


        let size = self.layers.len();

        for i in 1..size {
            let l = &self.layers[i];

            let mut z = &l.weights * current + &l.intercepts;
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
            self.layers[i].z = z;
            self.layers[i].a = t.clone();
            current = &t;
        }

        current.clone()
    }
}

impl BackProp for NeuralNetwork {
    fn back_prop(&mut self,
                 example_idx: usize,
                 y_hat: &DVector<f64>,
                 y: &DVector<f64>) -> &Self {
        let l = &mut self.layers;
        let mut idx = l.len() - 1;

        l[idx].dz = y_hat - y;
        l[idx].dw = &l[idx].dz * &l[idx - 1].a.transpose();
        l[idx].db = l[idx].dz.clone();

        idx -= 1;
        while idx > 0 {
            let t = match &l[idx].activation_type {
                ActivationType::Relu => MLOps::vectorize(&l[idx].z, MLOps::relu_derivative),
                ActivationType::Sigmoid => MLOps::vectorize(&l[idx].z, MLOps::sigmoid_derivative),
                ActivationType::Tanh => MLOps::vectorize(&l[idx].z, MLOps::tanh_derivative),
                ActivationType::SoftMax => MLOps::soft_max_derivative(&l[idx].z),
            };

            l[idx].dz = (&l[idx + 1].dz * &l[idx + 1].weights).component_mul(&t);
            l[idx].dw = &l[idx].dw + &l[idx].dz * &l[idx - 1].a.transpose();
            l[idx].db = &l[idx].db + l[idx].dz.clone();

            idx -= 1;
        }

        self
    }
}

impl NeuralNetwork {
    fn build(nd: &NeuralNetworkDefinition, rng: &mut rand_pcg::Pcg32) -> NeuralNetwork {
        let layers = &nd.layers_dimensions;

        let mut num_inputs = nd.num_features;
        for l in layers {
            Layer {
                num_activations: *l,
                intercepts: DVector::from_vec(vec![0.0_f64; *l]),
                weights: MatrixUtil::rand_matrix(*l, num_inputs, nd.rand_init_epsilon, rng),
                activation_type: nd.activation_type.clone(),

                // Dummy inits
                z: DVector::from_vec(vec![0.0_f64; *l]),
                a: DVector::from_vec(vec![0.0_f64; *l]),
                dz: DVector::from_vec(vec![0.0_f64; *l]),
                dw: DMatrix::from_vec(*l, num_inputs, vec![0.0_f64; *l * num_inputs]),
                db: DVector::from_vec(vec![0.0_f64; *l]),
                momentum_dw: DMatrix::from_vec(*l, num_inputs, vec![0.0_f64; *l * num_inputs]),
                momentum_db: DVector::from_vec(vec![0.0_f64; *l]),
            };

            num_inputs = *l;
        }


        unimplemented!()
    }
}

impl GradientDescent {
    fn optimize(&self,
                nn: &mut NeuralNetwork,
                data: &DMatrix<f64>,
                y: &DVector<f64>) -> () {
        fn update_weights(learning_rate: f64, nn: &mut NeuralNetwork) {
            for mut l in nn.layers.iter_mut() {
                l.weights = &l.weights - learning_rate * &l.momentum_dw;
                l.intercepts = &l.intercepts - learning_rate * &l.momentum_db;
            }
        }

        let (num_features, num_examples) = data.shape();
        let mut converged = false;

        let mut iteration = 0;

        while !converged {
            for k in (0..num_examples).step_by(self.mini_batch_size) {
                for j in 0..self.mini_batch_size {
                    let i = k + j; // partition
                    let x = data.column(i).into();

                    nn.layers[1].a = x;

                    let y_hat = nn.forward_prop();
                    nn.back_prop(i, &y_hat, &y);
                }


                for mut l in nn.layers.iter_mut() {
                    l.dw = &l.dw / self.mini_batch_size as f64;
                    l.db = &l.db / self.mini_batch_size as f64;
                }

                if iteration > 0 {
                    for mut l in nn.layers.iter_mut() {
                        l.momentum_dw = self.momentum_beta * &l.momentum_dw + (1. - self.momentum_beta) * &l.dw;
                        l.momentum_db = self.momentum_beta * &l.momentum_db + (1. - self.momentum_beta) * &l.db;
                    }
                } else {
                    for mut l in nn.layers.iter_mut() {
                        l.momentum_dw = l.dw.clone();
                        l.momentum_db = l.db.clone();
                    }
                }

                update_weights(self.learning_rate, nn);
                iteration += 1;
            }
        }
        unimplemented!()
    }
}


