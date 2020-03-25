use nalgebra::*;

use crate::neunet::api::defs::{ActivationType, Metric, NeuralNetworkDefinition, NNModel, RandomInitializer};
use crate::neunet::utils::matrix::MatrixUtil;
use crate::neunet::utils::ml::MLOps;

trait Optimizer {
    fn optimize(&self,
                nn: &mut NeuralNetwork,
                data: &DMatrix<f64>,
                labels: &DMatrix<f64>) -> ();
}

pub enum RegularizationType {
    No_Regularization,
    L2,
}

pub struct GradientDescent {
    pub momentum_beta: f64,
    pub mini_batch_size: usize,
    pub learning_rate: f64,
    pub regularization_type: RegularizationType,
    pub lambda_regularization: f64,
    pub test_func: fn(&DVector<f64>) -> Vec<Metric>,
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

    DW: DMatrix<f64>,
    DB: DVector<f64>,

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
    num_features: usize,
    layers: Vec<Layer>,
}

impl std::fmt::Debug for NeuralNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.layers)
    }
}

trait ForwardProp {
    fn forward_prop(&mut self, x: &DVector<f64>) -> DVector<f64>;
}


trait BackProp {
    fn back_prop(&mut self, x: &DVector<f64>, y_hat: &DVector<f64>, y: &DVectorSlice<f64>) -> &Self;
}

impl ForwardProp for NeuralNetwork {
    fn forward_prop(&mut self, x: &DVector<f64>) -> DVector<f64> {
        let mut current = x;
        let mut t;


        let size = self.layers.len();

        for i in 0..size {
            let l = &self.layers[i];

            let mut z = &l.weights * current + &l.intercepts;
            t = match &l.activation_type {
                ActivationType::Sigmoid =>
                    MLOps::apply(&z, MLOps::sigmoid),
                ActivationType::Relu =>
                    MLOps::apply(&z, MLOps::relu),
                ActivationType::Tanh =>
                    MLOps::apply(&z, MLOps::tanh),
                ActivationType::SoftMax => {
                    let sm = MLOps::soft_max(&z);
                    sm
                }
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
                 x: &DVector<f64>,
                 y_hat: &DVector<f64>,
                 y: &DVectorSlice<f64>) -> &Self {
        let l = &mut self.layers;
        let mut idx = l.len() - 1;

        l[idx].dz = y_hat - y;
        l[idx].dw = &l[idx].dz * &l[idx - 1].a.transpose();
        l[idx].db = l[idx].dz.clone();

        idx -= 1;
        let mut done = false;
        while !done {
            let t = match &l[idx].activation_type {
                ActivationType::Relu => MLOps::apply(&l[idx].z, MLOps::relu_derivative),
                ActivationType::Sigmoid => MLOps::apply(&l[idx].z, MLOps::sigmoid_derivative),
                ActivationType::Tanh => MLOps::apply(&l[idx].z, MLOps::tanh_derivative),
                ActivationType::SoftMax => MLOps::soft_max_derivative(&l[idx].z),
            };


            l[idx].dz = ((&l[idx + 1].weights).transpose() * &l[idx + 1].dz).component_mul(&t);

            if idx > 0 {
                l[idx].dw = &l[idx].dw + &l[idx].dz * &l[idx - 1].a.transpose();
            } else {
                l[idx].dw = &l[idx].dw + &l[idx].dz * &x.transpose();
            }

            l[idx].db = &l[idx].db + l[idx].dz.clone();

            if idx == 0 {
                done = true;
            } else {
                idx -= 1;
            }
        }

        self
    }
}

impl NeuralNetwork {
    pub fn build<R: RandomInitializer + Copy>(nd: NeuralNetworkDefinition<R>, rng: &mut rand_pcg::Pcg32) -> NeuralNetwork {
        let layers = &nd.layers;

        let mut num_inputs = nd.num_features;
        let mut initted = Vec::<Layer>::with_capacity(layers.len());
        let initializer = nd.rand_initializer;

        for l in layers {
            let w = initializer.weights(l.num_activations,
                                        num_inputs,
                                        rng);
            let dwl = DMatrix::from_vec(l.num_activations, num_inputs, vec![0.0_f64; l.num_activations * num_inputs]);
            let dbl = DVector::from_vec(vec![0.0_f64; l.num_activations]);

            initted.push(Layer {
                num_activations: l.num_activations,
                intercepts: DVector::from_vec(vec![0.0_f64; l.num_activations]),
                weights: w.clone(),
                activation_type: l.activation_type.clone(),

                // Dummy inits
                z: DVector::from_vec(vec![0.0_f64; l.num_activations]),
                a: DVector::from_vec(vec![0.0_f64; l.num_activations]),
                dz: DVector::from_vec(vec![0.0_f64; l.num_activations]),

                dw: dwl.clone(),
                db: dbl.clone(),

                DW: dwl.clone(),
                DB: dbl.clone(),

                momentum_dw: dwl.clone(),
                momentum_db: dbl.clone(),
            });

            num_inputs = l.num_activations;
        }


        NeuralNetwork {
            num_features: nd.num_features,
            layers: initted,
        }
    }
}

impl GradientDescent {
    pub fn optimize<'a>(&self,
                        nn: &'a mut NeuralNetwork,
                        data: &DMatrix<f64>,
                        y: &DMatrix<f64>) -> &'a NeuralNetwork {

        fn reset_gradients(nn: &mut NeuralNetwork) {
            let mut num_inputs = nn.num_features;
            let mut i = 0;
            for mut l in nn.layers.iter_mut() {
                l.dw = DMatrix::from_vec(l.num_activations, num_inputs, vec![0.0_f64; l.num_activations * num_inputs]);
                l.db = DVector::from_vec(vec![0.0_f64; l.num_activations]);
                num_inputs = l.num_activations;
                i += 1;
            }
        }

        fn update_weights(learning_rate: f64, nn: &mut NeuralNetwork) {
            for mut l in nn.layers.iter_mut() {
                l.weights = &l.weights - learning_rate * &l.momentum_dw;
                l.intercepts = &l.intercepts - learning_rate * &l.momentum_db;
            }
        }

        fn l2_reg(nn: &NeuralNetwork) -> f64 {
            let mut sum = 0.0;
            for l in &nn.layers {
                sum += l.weights.data.as_vec().into_iter().map(|e| (*e) * (*e)).sum::<f64>();
            }
            sum
        }


        let (num_features, num_examples) = data.shape();
        let mut converged = false;

        let mut iteration = 0;
        let mut epoch = 0;

        while !converged {
            println!("Running epoch {}", epoch);

            for k in (0..num_examples).step_by(self.mini_batch_size) {
                println!("Running iteration {}", iteration);
                println!("Mini batch start {}", k);

                let mut batch_loss = 0.0_f64;

                reset_gradients(nn);

                for j in 0..self.mini_batch_size {
                    let i = k + j; // partition
                    let x = data.column(i).into();

                    let y_hat = nn.forward_prop(&x);
                    batch_loss += MLOps::loss_one_hot(&y.column(i), y_hat.data.as_vec());
                    nn.back_prop(&x, &y_hat, &y.column(i));
                }

                batch_loss = batch_loss / self.mini_batch_size as f64;

                batch_loss += match self.regularization_type {
                    RegularizationType::No_Regularization => 0.0,
                    RegularizationType::L2 => {
                        let r = (self.lambda_regularization / (2.0f64 * self.mini_batch_size as f64)) * l2_reg(&nn);
                        println!("L2 reg {}", r);
                        r
                    }
                };


                println!("Loss for batch {} is {}", k, batch_loss);

                let mut i = 0;
                for mut l in nn.layers.iter_mut() {
                    l.dw = &l.dw / self.mini_batch_size as f64;
                    l.db = &l.db / self.mini_batch_size as f64;
                    match self.regularization_type {
                        RegularizationType::No_Regularization => (),
                        RegularizationType::L2 =>
                            l.dw = &l.dw + self.lambda_regularization * &l.weights
                    };
                    // println!("  After back-prop Layer {} grads {}", i, l.dw);
                    i += 1;
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
                        l.momentum_db = DVector::from_vec(vec![0.0; r]);
                    }
                }

                update_weights(self.learning_rate, nn);
                iteration += 1;
            }
            epoch += 1
        }
        nn
    }
}


