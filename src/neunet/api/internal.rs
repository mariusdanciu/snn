#![allow(dead_code)]
#![allow(unused)]

use std::fs::File;
use std::io;
use std::io::Write;

use chrono::Utc;
use nalgebra::*;
use serde_json::json;
use serde_json::Value;

use crate::neunet::api::defs::*;
use crate::neunet::graphics::plotter::*;
use crate::neunet::utils::ml::*;

pub trait ForwardProp {
    fn forward_prop(&mut self, x: &DVector<f32>) -> DVector<f32>;
}


pub trait BackProp {
    fn back_prop(&mut self, x: &DVector<f32>, y_hat: &DVector<f32>, y: &DVectorSlice<f32>) -> &Self;
}

impl ForwardProp for NNModel {
    fn forward_prop(&mut self, x: &DVector<f32>) -> DVector<f32> {
        let mut current = x;
        let mut t;


        let size = self.layers.len();

        for i in 0..size {
            let l = &self.layers[i];

            let z = &l.weights * current + &l.intercepts;
            t = match &l.activation_type {
                ActivationType::Sigmoid =>
                    apply_to_vec(&z, sigmoid),
                ActivationType::Relu =>
                    apply_to_vec(&z, relu),
                ActivationType::Tanh =>
                    apply_to_vec(&z, tanh),
                ActivationType::SoftMax =>
                    soft_max(&z),
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
                 x: &DVector<f32>,
                 y_hat: &DVector<f32>,
                 y: &DVectorSlice<f32>) -> &Self {
        let l = &mut self.layers;
        let mut idx = l.len() - 1;

        l[idx].dz = y_hat - y;
        l[idx].dw = &l[idx].dw + &l[idx].dz * &l[idx - 1].a.transpose();
        l[idx].db = &l[idx].db + &l[idx].dz;


        idx -= 1;
        let mut done = false;
        while !done {
            let t = match &l[idx].activation_type {
                ActivationType::Relu => apply_to_vec(&l[idx].z, relu_derivative),
                ActivationType::Sigmoid => apply_to_vec(&l[idx].z, sigmoid_derivative),
                ActivationType::Tanh => apply_to_vec(&l[idx].z, tanh_derivative),
                ActivationType::SoftMax => soft_max_derivative(&l[idx].z),
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
    pub fn build<R: RandomInitializer + Copy>(meta: ModelMeta, nd: NeuralNetworkArchitecture<R>, rng: &mut rand_pcg::Pcg32) -> NNModel {
        let layers = &nd.layers;

        let mut num_inputs = nd.num_features;
        let mut initted = Vec::<Layer>::with_capacity(layers.len());
        let initializer = nd.rand_initializer;

        for l in layers {
            let w = initializer.weights(l.num_activations,
                                        num_inputs,
                                        rng);
            let dwl = DMatrix::from_vec(l.num_activations, num_inputs, vec![0.0_f32; l.num_activations * num_inputs]);
            let dbl = DVector::from_vec(vec![0.0_f32; l.num_activations]);

            initted.push(Layer {
                num_activations: l.num_activations,
                intercepts: DVector::from_vec(vec![0.0_f32; l.num_activations]),
                weights: w.clone(),
                activation_type: l.activation_type.clone(),

                // Dummy inits
                z: DVector::from_vec(vec![0.0_f32; l.num_activations]),
                a: DVector::from_vec(vec![0.0_f32; l.num_activations]),
                dz: DVector::from_vec(vec![0.0_f32; l.num_activations]),

                dw: dwl.clone(),
                db: dbl.clone(),

                momentum_dw: dwl.clone(),
                momentum_db: dbl.clone(),

                rmsp_dw: dwl.clone(),
                rmsp_db: dbl.clone(),
            });

            num_inputs = l.num_activations;
        }


        NNModel {
            meta,
            num_features: nd.num_features,
            num_classes: nd.num_classes,
            layers: initted,
            training_info: None,
        }
    }
}

impl Prediction for NNModel {
    fn predict(&mut self, data: &DMatrix<f32>) -> DMatrix<f32> {
        let (_, cols) = data.shape();
        let mut preds: Vec<f32> = Vec::new();

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
             observer: &mut TrainingObserver,
             train_data: LabeledData,
             test_data: LabeledData) -> Result<NNModel, Box<dyn std::error::Error>> {
        fn reset_gradients(model: &mut NNModel) {
            for l in model.layers.iter_mut() {
                for e in l.dw.iter_mut() {
                    *e = 0.0f32;
                }
                for e in l.db.iter_mut() {
                    *e = 0.0f32;
                }
            }
        }

        fn update_weights(iteration: u32, hp: &HyperParams, nn: &mut NNModel) {
            fn momentum(iteration: u32, hp: &HyperParams, nn: &mut NNModel) {
                if iteration > 0 {
                    // Apply  weighted moving averages
                    for mut l in nn.layers.iter_mut() {
                        l.momentum_dw = hp.momentum_beta * &l.momentum_dw + (1. - hp.momentum_beta) * &l.dw;
                        l.momentum_db = hp.momentum_beta * &l.momentum_db + (1. - hp.momentum_beta) * &l.db;
                    }
                } else {
                    for mut l in nn.layers.iter_mut() {
                        let (r, c) = l.dw.shape();
                        l.momentum_dw = DMatrix::from_fn(r, c, |_, _| 0.0);
                        l.momentum_db = DVector::from_vec(vec![0.0; r]);
                    }
                }

                for mut l in nn.layers.iter_mut() {
                    l.weights = &l.weights - hp.learning_rate * &l.momentum_dw;
                    l.intercepts = &l.intercepts - hp.learning_rate * &l.momentum_db;
                }
            }

            fn rms_prop(iteration: u32, hp: &HyperParams, nn: &mut NNModel) {
                if iteration > 0 {
                    // Apply  weighted moving averages
                    for mut l in nn.layers.iter_mut() {
                        l.rmsp_dw = hp.rms_prop_beta * &l.rmsp_dw + (1. - hp.rms_prop_beta) * (&l.dw.component_mul(&l.dw));
                        l.rmsp_db = hp.rms_prop_beta * &l.rmsp_db + (1. - hp.rms_prop_beta) * (&l.db);
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
                    l.weights = &l.weights - hp.learning_rate * &l.dw.component_div(&apply_to_mat(&l.rmsp_dw, rmsprop_div));
                    l.intercepts = &l.intercepts - hp.learning_rate * &l.db.component_div(&apply_to_vec(&l.rmsp_db, rmsprop_div));
                }
            }

            fn adam(iteration: u32, hp: &HyperParams, nn: &mut NNModel) {
                if iteration > 0 {
                    // Apply  weighted moving averages
                    for mut l in nn.layers.iter_mut() {
                        l.momentum_dw = hp.momentum_beta * &l.momentum_dw + (1. - hp.momentum_beta) * &l.dw;
                        l.momentum_db = hp.momentum_beta * &l.momentum_db + (1. - hp.momentum_beta) * &l.db;

                        l.rmsp_dw = hp.rms_prop_beta * &l.rmsp_dw + (1. - hp.rms_prop_beta) * (&l.dw.component_mul(&l.dw));
                        l.rmsp_db = hp.rms_prop_beta * &l.rmsp_db + (1. - hp.rms_prop_beta) * (&l.db.component_mul(&l.db));
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
                    let momentum_dw_corrected = &l.momentum_dw / (1f32 - hp.momentum_beta.powi(1 + iteration as i32));
                    let momentum_db_corrected = &l.momentum_db / (1f32 - hp.momentum_beta.powi(1 + iteration as i32));

                    let rms_dw_corrected = &l.rmsp_dw / (1f32 - hp.rms_prop_beta.powi(1 + iteration as i32));
                    let rms_db_corrected = &l.rmsp_db / (1f32 - hp.rms_prop_beta.powi(1 + iteration as i32));


                    let dw = &momentum_dw_corrected.component_div(&apply_to_mat(&rms_dw_corrected, rmsprop_div));
                    let db = &momentum_db_corrected.component_div(&apply_to_vec(&rms_db_corrected, rmsprop_div));

                    l.weights = &l.weights - hp.learning_rate * dw;
                    l.intercepts = &l.intercepts - hp.learning_rate * db;
                }
            }

            match hp.optimization_type {
                OptimizationType::Momentum =>
                    momentum(iteration, hp, nn),
                OptimizationType::RMSProp =>
                    rms_prop(iteration, hp, nn),
                OptimizationType::Adam =>
                    adam(iteration, hp, nn),
                _ =>
                    for mut l in nn.layers.iter_mut() {
                        l.weights = &l.weights - hp.learning_rate * &l.dw;
                        l.intercepts = &l.intercepts - hp.learning_rate * &l.db;
                    }
            }
        }

        fn norm(nn: &NNModel) -> f32 {
            let mut sum = 0.0;
            for l in &nn.layers {
                sum += l.weights.data.as_vec().into_iter().map(|e| (*e) * (*e)).sum::<f32>();
            }
            sum
        }

        fn test(model: &mut NNModel, features: &DMatrixSlice<f32>, labels: &DMatrixSlice<f32>) -> (DMatrix<usize>, Vec<f32>, f32) {
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

            let total: usize = confusion.data.as_vec().iter().sum();

            let mut l_acc: Vec<f32> = vec![0.0; 0];
            for c in 0..model.num_classes {
                let col_sum: usize = confusion.column(c).iter().sum();
                let row_sum: usize = confusion.row(c).iter().sum();

                let t_p: usize = confusion[(c, c)];
                let f_p: usize = col_sum - t_p;
                let f_n: usize = row_sum - t_p;

                let t_n: usize = total - t_p - f_p - f_n;

                let accuracy: f32 = (t_p + t_n) as f32 / (t_p + t_n + f_p + f_n) as f32;

                l_acc.push(accuracy);
            }
            (confusion, l_acc, corrects as f32 / (corrects as f32 + incorrects as f32))
        }

        let (_num_features, num_examples) = train_data.features.shape();
        let (num_labels, _) = train_data.labels.shape();

        let mut stop = false;

        let mut iteration: u32 = 0;
        let mut epoch: u32 = 0;

        let mut batch_loss = 0.0_f32;
        let mut test_accuracy = 0.0_f32;


        observer.emit(TrainMessage::Started {
            time: Utc::now()
        });

        while !stop {
            for k in (0..num_examples).step_by(hp.mini_batch_size) {
                batch_loss = 0.0_f32;

                reset_gradients(self);

                let batch_size = if k + hp.mini_batch_size > num_examples {
                    num_examples - k
                } else {
                    hp.mini_batch_size
                };

                let mean_fact = 1.0f32 / batch_size as f32;

                let cost_reg = match hp.l2_regularization {
                    None =>
                        0.0f32,
                    Some(reg) =>
                        0.5f32 * (reg * norm(self))
                };

                for j in 0..batch_size {
                    let i = k + j; // partition
                    let x = train_data.features.column(i).into();

                    let y_hat = self.forward_prop(&x);
                    batch_loss += cross_entropy_one_hot(&train_data.labels.column(i), &y_hat);

                    self.back_prop(&x, &y_hat, &train_data.labels.column(i));
                }


                batch_loss = mean_fact * (batch_loss + cost_reg);


                for mut l in self.layers.iter_mut() {
                    match hp.l2_regularization {
                        None =>
                            l.dw = mean_fact * &l.dw,
                        Some(reg) =>
                            l.dw = mean_fact * (&l.dw + reg * &l.weights),
                    };

                    l.db = mean_fact * &l.db;
                }


                update_weights(iteration, &hp, self);

                let (train_cm, train_lacc, train_acc) = test(self,
                                                             &train_data.features.slice((0, k), (self.num_features, batch_size)),
                                                             &train_data.labels.slice((0, k), (self.num_classes, batch_size)));

                let (test_cm, test_lacc, test_acc) = test(self, &test_data.features, &test_data.labels);

                test_accuracy = test_acc;

                observer.emit(TrainMessage::Running {
                    time: Utc::now(),
                    iteration: iteration,
                    epoch: epoch,
                    batch_start: k as u32,
                    metrics: Metrics {
                        loss: batch_loss,
                        train_eval: TrainingEval {
                            confusion_matrix_dim: num_labels,
                            confusion_matrix: train_cm,
                            labels_accuracies: train_lacc,
                            accuracy: train_acc,
                        },
                        test_eval: TrainingEval {
                            confusion_matrix_dim: num_labels,
                            confusion_matrix: test_cm,
                            labels_accuracies: test_lacc,
                            accuracy: test_acc,
                        },
                    },
                });

                iteration += 1;
            }
            epoch += 1;

            stop = epoch > hp.max_epochs || test_accuracy >= hp.max_accuracy_threshold;
        }

        observer.emit(TrainMessage::Success {
            time: Utc::now()
        });

        let name = self.meta.name.clone();

        Ok(NNModel {
            meta: ModelMeta {
                name
            },
            num_features: self.num_features,
            num_classes: self.num_classes,
            layers: self.layers.clone(),
            training_info: Some(TrainingInfo {
                hyper_params: hp,
                num_epochs_used: epoch,
                num_iterations_used: iteration,
                loss: batch_loss,
            }),

        })
    }
}

impl Json for TrainMessage {
    fn to_json(&self, pretty: bool) -> String {
        let js = match self {
            TrainMessage::Started { time } => json!({
                    "started" : json!({
                        "time": json!(time.format("%Y%m%dT%H%M%SZ").to_string()),
                    })
                }),
            TrainMessage::Running { time, iteration, epoch, batch_start, metrics } => {
                let mut v = json!({
                    "epoch" : epoch,
                    "iteration": iteration,
                    "batch_start": batch_start
                });

                let map = v.as_object_mut().unwrap();

                map.insert("loss".to_string(), json!(metrics.loss));
                map.insert("train_eval".to_string(),
                           json!({
                                  "confusion_matrix" : json!({
                                     "data": json!(metrics.train_eval.confusion_matrix.data.as_vec()),
                                     "predictions_on_cols" : json!(true),
                                     "data_orientation_per_col" : json!(true),
                                     "dimension" : json!(metrics.train_eval.confusion_matrix_dim)
                                  }),
                                  "label_accuracies" : json!(metrics.train_eval.labels_accuracies),
                                  "accuracy" : json!(metrics.train_eval.accuracy),
                                  }));
                map.insert("test_eval".to_string(),
                           json!({
                                  "confusion_matrix" : json!({
                                     "data": json!(metrics.test_eval.confusion_matrix.data.as_vec()),
                                     "predictions_on_cols" : json!(true),
                                     "data_orientation_per_col" : json!(true),
                                     "dimension" : json!(metrics.test_eval.confusion_matrix_dim)
                                  }),
                                  "label_accuracies" : json!(metrics.test_eval.labels_accuracies),
                                   "accuracy" : json!(metrics.test_eval.accuracy),
                                  }));
                json!({
                    "running" : v
                })
            }
            TrainMessage::Success { time } => json!({
                    "success" : json!({
                        "time": json!(format!("{}", time.format("%Y%m%dT%H%M%SZ"))),
                    })
                }),
            TrainMessage::Error { time, reason } => json!({
                    "error" : json!({
                        "time": json!(format!("{}", time.format("%Y%m%dT%H%M%SZ"))),
                        "reason" : json!(reason)
                    })
                }),
        };


        if pretty {
            serde_json::to_string_pretty(&js).unwrap()
        } else {
            js.to_string()
        }
    }
}

impl TrainingObserver for ConsoleObserver {
    fn emit(&mut self, msg: TrainMessage) {
        println!("{}", msg.to_json(false));
        match msg {
            TrainMessage::Started { time } => {
                self.start_time = Some(time)
            }

            TrainMessage::Running { time, iteration, epoch, batch_start, metrics } => {
                self.train_accuracy_his.push(metrics.train_eval.accuracy);
                self.test_accuracy_his.push(metrics.test_eval.accuracy);

                println!("\t loss {}", metrics.loss);
                println!("\t train accuracy {}", metrics.train_eval.accuracy);
                println!("\t test accuracy {}", metrics.test_eval.accuracy);

                match plot_data(format!("./train_{}.png", self.start_time.unwrap().format("%Y%m%dT%H%M%SZ")),
                                &self.train_accuracy_his,
                                &self.test_accuracy_his) {
                    Ok(_) => (),
                    Err(e) => println!("\t Cannot draw plot {}", e),
                }
            }
            _ => ()
        }
    }
}

impl Json for NNModel {
    fn to_json(&self, pretty: bool) -> String {
        let js_layers: Vec<Value> = self.layers.iter().map(|l| {
            let k: Vec<Value> = l.weights.data.as_vec().iter().map(|e| json!(e)).collect();
            let i: Vec<Value> = l.intercepts.data.as_vec().iter().map(|e| json!(e)).collect();
            let shp = l.weights.shape();
            json!({
                "rows" : json!(shp.0),
                "cols" : json!(shp.1),
                "activations" : json!(shp.0),
                "activation_type" : json!(format!("{:?}", l.activation_type)),
                "weights" : json!(k),
                "intercepts" : json!(i)
            })
        }).collect();

        let mut model = json!({
           "data_info": json!({
                "num_features" : json!(self.num_features),
                "num_classes": json!(self.num_classes),
           }),
           "meta" : json!({
                "version": json!("1.0.0"),
                "name" : json!(self.meta.name)
           }),
           "layers": js_layers,
        });

        match &self.training_info {
            Some(info) => {
                let obj = model.as_object_mut().unwrap();
                obj.insert("training_info".to_owned(), serde_json::to_value(info).unwrap());
            }
            _ => ()
        };

        if pretty {
            serde_json::to_string_pretty(&model).unwrap()
        } else {
            model.to_string()
        }
    }
}

impl Save for NNModel {
    fn save(&self, path: &str) -> io::Result<String> {
        let mut f: File = File::create(format!("{}/{}.json", path, self.meta.name))?;
        let s = self.to_json(true);
        f.write_all(s.as_bytes())?;
        Ok(path.to_owned())
    }
}