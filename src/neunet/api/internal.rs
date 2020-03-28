use nalgebra::*;
use nalgebra::base::storage::Storage;

use crate::neunet::api::defs::*;
use crate::neunet::utils::matrix::MatrixUtil;
use crate::neunet::utils::ml::MLOps;

pub trait ForwardProp {
    fn forward_prop(&mut self, x: &DVector<f64>) -> DVector<f64>;
}


pub trait BackProp {
    fn back_prop(&mut self, x: &DVector<f64>, y_hat: &DVector<f64>, y: &DVectorSlice<f64>) -> &Self;
}

impl ForwardProp for NNModel {
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

impl BackProp for NNModel {
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

impl NNModel {
    pub fn build<R: RandomInitializer + Copy>(nd: NeuralNetworkArchitecture<R>, rng: &mut rand_pcg::Pcg32) -> NNModel {
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


        NNModel {
            num_features: nd.num_features,
            num_classes: nd.num_classes,
            layers: initted,
        }
    }
}

impl Prediction for NNModel {
    fn predict(&mut self, data: &DMatrix<f64>) -> DMatrix<f64> {
        let (rows, cols) = data.shape();
        let mut preds: Vec<f64> = Vec::with_capacity(self.num_classes * cols);

        for c in data.column_iter() {
            let score = self.forward_prop(&DVector::from_column_slice(c.as_slice()));
            preds.extend(score.data.as_vec());
        }

        DMatrix::from_vec(self.num_classes, cols, preds)
    }
}

impl Train for NNModel {
    fn train(&mut self,
             hp: HyperParams,
             train_data: LabeledData,
             test_data: LabeledData) -> &NNModel {
        fn reset_gradients(model: &mut NNModel) {
            let mut num_inputs = model.num_features;
            let mut i = 0;
            for mut l in model.layers.iter_mut() {
                l.dw = DMatrix::from_vec(l.num_activations, num_inputs, vec![0.0_f64; l.num_activations * num_inputs]);
                l.db = DVector::from_vec(vec![0.0_f64; l.num_activations]);
                num_inputs = l.num_activations;
                i += 1;
            }
        }

        fn update_weights(learning_rate: f64, nn: &mut NNModel) {
            for mut l in nn.layers.iter_mut() {
                l.weights = &l.weights - learning_rate * &l.momentum_dw;
                l.intercepts = &l.intercepts - learning_rate * &l.momentum_db;
            }
        }

        fn l2_reg(nn: &NNModel) -> f64 {
            let mut sum = 0.0;
            for l in &nn.layers {
                sum += l.weights.data.as_vec().into_iter().map(|e| (*e) * (*e)).sum::<f64>();
            }
            sum
        }


        let (num_features, num_examples) = train_data.features.shape();
        let mut converged = false;

        let mut iteration = 0;
        let mut epoch = 0;

        while !converged {
            println!("Running epoch {}", epoch);

            for k in (0..num_examples).step_by(hp.mini_batch_size) {
                println!("Running iteration {}", iteration);
                println!("Mini batch start {}", k);

                let mut batch_loss = 0.0_f64;

                reset_gradients(self);

                for j in 0..hp.mini_batch_size {
                    let i = k + j; // partition
                    let x = train_data.features.column(i).into();

                    let y_hat = self.forward_prop(&x);
                    batch_loss += MLOps::loss_one_hot(&train_data.labels.column(i), y_hat.data.as_vec());
                    self.back_prop(&x, &y_hat, &train_data.labels.column(i));
                }


                batch_loss = batch_loss / hp.mini_batch_size as f64;

                batch_loss += match hp.regularization_type {
                    RegularizationType::No_Regularization => 0.0,
                    RegularizationType::L2 => {
                        let r = (hp.lambda_regularization / (2.0f64 * hp.mini_batch_size as f64)) * l2_reg(self);
                        println!("L2 reg {}", r);
                        r
                    }
                };


                println!("Loss for batch {} is {}", k, batch_loss);

                let mut i = 0;
                for mut l in self.layers.iter_mut() {
                    l.dw = &l.dw / hp.mini_batch_size as f64;
                    l.db = &l.db / hp.mini_batch_size as f64;
                    match hp.regularization_type {
                        RegularizationType::No_Regularization => (),
                        RegularizationType::L2 =>
                            l.dw = &l.dw + hp.lambda_regularization * &l.weights
                    };
                    // println!("  After back-prop Layer {} grads {}", i, l.dw);
                    i += 1;
                }

                if iteration > 0 {
                    // Apply  weighted moving averages
                    for mut l in self.layers.iter_mut() {
                        l.momentum_dw = hp.momentum_beta * &l.momentum_dw + (1. - hp.momentum_beta) * &l.dw;
                        l.momentum_db = hp.momentum_beta * &l.momentum_db + (1. - hp.momentum_beta) * &l.db;
                    }
                } else {
                    for mut l in self.layers.iter_mut() {
                        let (r, c) = l.dw.shape();
                        l.momentum_dw = DMatrix::from_fn(r, c, |a, b| 0.0);
                        l.momentum_db = DVector::from_vec(vec![0.0; r]);
                    }
                }

                update_weights(hp.learning_rate, self);
                iteration += 1;
            }
            epoch += 1
        }

        self
    }
}
