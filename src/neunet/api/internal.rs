use std::slice;

use nalgebra::*;
use nalgebra::base::storage::Storage;

use crate::neunet::api::defs::*;
use crate::neunet::graphics::plotter::*;
use crate::neunet::utils::matrix::*;
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
        l[idx].dw = &l[idx].dw + &l[idx].dz * &l[idx - 1].a.transpose();
        l[idx].db = &l[idx].db + &l[idx].dz;

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
            for mut l in model.layers.iter_mut() {
                for mut e in l.dw.iter_mut() {
                    *e = 0.0f64;
                }
                for mut e in l.db.iter_mut() {
                    *e = 0.0f64;
                }
            }
        }

        fn update_weights(learning_rate: f64, nn: &mut NNModel) {
            for mut l in nn.layers.iter_mut() {
                l.weights = &l.weights - learning_rate * &l.dw;
                l.intercepts = &l.intercepts - learning_rate * &l.db;
            }
        }

        fn norm(nn: &NNModel) -> f64 {
            let mut sum = 0.0;
            for l in &nn.layers {
                sum += l.weights.data.as_vec().into_iter().map(|e| (*e) * (*e)).sum::<f64>();
            }
            sum
        }

        fn test(model: &mut NNModel, features: &DMatrixSlice<f64>, labels: &DMatrixSlice<f64>) -> f64 {
            let mut confusion = DMatrix::<usize>::zeros(model.num_classes, model.num_classes);

            let mut corrects = 0;
            let mut incorrects = 0;
            for c in 0..features.shape().1 {
                let predicted = model.forward_prop(&DVector::from_column_slice(features.column(c).as_slice()));
                let actual = labels.column(c);

                let predicted_class = predicted.argmax().0;
                let actual_class = actual.argmax().0;

                confusion[(actual_class, predicted_class)] += 1;

                if predicted_class == actual_class {
                    corrects += 1;
                } else {
                    incorrects += 1;
                }
            }
            println!("\t\tConfusion matrix {}", confusion);

            let total: usize = confusion.data.as_vec().iter().sum();

            for c in 0..model.num_classes {
                let col_sum: usize = confusion.column(c).iter().sum();
                let row_sum: usize = confusion.row(c).iter().sum();

                let t_p: usize = confusion[(c, c)];
                let f_p: usize = col_sum - t_p;
                let f_n: usize = row_sum - t_p;

                let t_n: usize = total - t_p - f_p - f_n;

                let accuracy: f64 = (t_p + t_n) as f64 / (t_p + t_n + f_p + f_n) as f64;

                println!("\t\tAccuracy for class {} = {}", c, accuracy);
            }
            corrects as f64 / (corrects as f64 + incorrects as f64)
        }


        let (num_features, num_examples) = train_data.features.shape();
        println!("Train data features {} examples {}", num_features, num_examples);

        let mut converged = false;

        let mut iteration = 0;
        let mut epoch = 0;

        let mut train_acc_his = Vec::new();
        let mut test_acc_his = Vec::new();

        while !converged {
            println!("Running epoch {}", epoch);

            for k in (0..num_examples).step_by(hp.mini_batch_size) {
                println!("\tRunning iteration {}", iteration);
                println!("\t\tMini batch start {}", k);

                let mut batch_loss = 0.0_f64;

                reset_gradients(self);

                let batch_size = if k + hp.mini_batch_size > num_examples {
                    num_examples - k
                } else {
                    hp.mini_batch_size
                };

                let mean_fact = 1.0f64 / batch_size as f64;

                let cost_reg = match hp.l2_regularization {
                    None =>
                        0.0f64,
                    Some(reg) =>
                        0.5f64 * (reg * norm(self))
                };

                println!("\t\tRegularization {}", cost_reg);

                for j in 0..batch_size {
                    let i = k + j; // partition
                    let x = train_data.features.column(i).into();

                    let y_hat = self.forward_prop(&x);
                    batch_loss += MLOps::cross_entropy_one_hot(&train_data.labels.column(i), y_hat.data.as_vec());

                    self.back_prop(&x, &y_hat, &train_data.labels.column(i));
                }


                batch_loss = mean_fact * (batch_loss + cost_reg);


                println!("\t\tLoss for batch {} is {}", k, batch_loss);


                for mut l in self.layers.iter_mut() {
                    match hp.l2_regularization {
                        None =>
                            l.dw = mean_fact * &l.dw,
                        Some(reg) =>
                            l.dw = mean_fact * (&l.dw + reg * &l.weights),
                    };

                    l.db = mean_fact * &l.db;
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

                let train_accuracy = test(self,
                                          &train_data.features.slice((0, k), (self.num_features, batch_size)),
                                          &train_data.labels.slice((0, k), (self.num_classes, batch_size)));

                let test_accuracy = test(self, &test_data.features, &test_data.labels);


                train_acc_his.push(train_accuracy as f32);
                test_acc_his.push(test_accuracy as f32);

                plot_data(String::from("./train.png"), &train_acc_his, &test_acc_his);

                println!("\t\t--------> Train accuracy {} Test accuracy {}", train_accuracy, test_accuracy);
                iteration += 1;
            }
            epoch += 1
        }

        self
    }
}
