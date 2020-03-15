use nalgebra::*;

use crate::neunet::definitions::{ActivationType, MLOps, NeuralNetworkDefinition, NNModel};
use crate::neunet::utils::matrix::MatrixUtil;

trait Optimizer {
    fn optimize(&self,
                nn: &mut NeuralNetwork,
                data: &DMatrix<f64>,
                labels: &DMatrix<f64>) -> ();
}

pub struct GradientDescent {
    pub momentum_beta: f64,
    pub rms_prop_beta: f64,
    pub epsilon_correction: f64,
    pub mini_batch_size: usize,
    pub learning_rate: f64,
    // 0.0001
    pub stop_cost_quota: f64,
    // 10 ^ -4
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


impl std::fmt::Debug for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\n\tLayer {{
        \t\tnum_activations : {:?}
        \t\tweights : {:?}
        \t\tactivation_type : {:?}
    }}", self.num_activations, self.weights.shape(), self.activation_type)
    }
}

pub struct NeuralNetwork {
    layers: Vec<Layer>
}

impl std::fmt::Debug for NeuralNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.layers)
    }
}

trait ForwardProp {
    fn forward_prop(&mut self) -> DVector<f64>;
}


trait BackProp {
    fn back_prop(&mut self, example_idx: usize, y_hat: &DVector<f64>, y: &DVectorSlice<f64>) -> &Self;
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
                    MLOps::apply(&z, MLOps::sigmoid),
                ActivationType::Relu =>
                    MLOps::apply(&z, MLOps::relu),
                ActivationType::Tanh =>
                    MLOps::apply(&z, MLOps::tanh),
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
                 y: &DVectorSlice<f64>) -> &Self {
        let l = &mut self.layers;
        let mut idx = l.len() - 1;

        l[idx].dz = y_hat - y;
        l[idx].dw = &l[idx].dz * &l[idx - 1].a.transpose();
        l[idx].db = l[idx].dz.clone();

        idx -= 1;
        while idx > 0 {
            let t = match &l[idx].activation_type {
                ActivationType::Relu => MLOps::apply(&l[idx].z, MLOps::relu_derivative),
                ActivationType::Sigmoid => MLOps::apply(&l[idx].z, MLOps::sigmoid_derivative),
                ActivationType::Tanh => MLOps::apply(&l[idx].z, MLOps::tanh_derivative),
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
    pub fn build(nd: &NeuralNetworkDefinition, rng: &mut rand_pcg::Pcg32) -> NeuralNetwork {
        let layers = &nd.layers;

        let mut num_inputs = nd.num_features;
        let mut initted = Vec::<Layer>::with_capacity(layers.len());
        for l in layers {
            initted.push(Layer {
                num_activations: l.num_activations,
                intercepts: DVector::from_vec(vec![0.0_f64; l.num_activations]),
                weights: MatrixUtil::rand_matrix(l.num_activations, num_inputs, l.rand_init_epsilon, rng),
                activation_type: l.activation_type.clone(),

                // Dummy inits
                z: DVector::from_vec(vec![0.0_f64; l.num_activations]),
                a: DVector::from_vec(vec![0.0_f64; l.num_activations]),
                dz: DVector::from_vec(vec![0.0_f64; l.num_activations]),
                dw: DMatrix::from_vec(l.num_activations, num_inputs, vec![0.0_f64; l.num_activations * num_inputs]),
                db: DVector::from_vec(vec![0.0_f64; l.num_activations]),
                momentum_dw: DMatrix::from_vec(l.num_activations, num_inputs, vec![0.0_f64; l.num_activations * num_inputs]),
                momentum_db: DVector::from_vec(vec![0.0_f64; l.num_activations]),
            });

            num_inputs = l.num_activations;
        }


        NeuralNetwork {
            layers: initted
        }
    }
}

impl GradientDescent {
    pub fn optimize<'a>(&self,
                    nn: &'a mut NeuralNetwork,
                    data: &DMatrix<f64>,
                    y: &DMatrix<f64>) -> &'a NeuralNetwork {
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
                println!("Running iteration {}", iteration);

                for j in 0..self.mini_batch_size {
                    let i = k + j; // partition
                    let x = data.column(i).into();

                    nn.layers[0].a = x;

                    let y_hat = nn.forward_prop();
                    nn.back_prop(i, &y_hat, &y.column(i));
                }


                for mut l in nn.layers.iter_mut() {
                    l.dw = &l.dw / self.mini_batch_size as f64;
                    l.db = &l.db / self.mini_batch_size as f64;
                }

                if iteration > 0 {
                    // Apply  weighted moving averages
                    for mut l in nn.layers.iter_mut() {
                        l.momentum_dw = self.momentum_beta * &l.momentum_dw + (1. - self.momentum_beta) * &l.dw;
                        l.momentum_db = self.momentum_beta * &l.momentum_db + (1. - self.momentum_beta) * &l.db;
                    }
                } else {
                    for mut l in nn.layers.iter_mut() {
                        let (r, c) = l.dw.shape();
                        l.momentum_dw = DMatrix::from_fn(r, c, |a, b| 0.0);
                        l.momentum_db = DVector::from_vec(vec![0.0; c]);
                    }
                }

                update_weights(self.learning_rate, nn);
                iteration += 1;
            }
        }
        nn
    }
}


