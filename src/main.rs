extern crate rand;
extern crate rand_pcg;

use std::ops::IndexMut;
use std::time::{SystemTime, UNIX_EPOCH};

use nalgebra::*;
use rand::distributions::{Distribution, Normal};
use rand_pcg::Pcg32;

use crate::neunet::api::defs::*;
use crate::neunet::files::idx::IdxFile;
use crate::neunet::loader::DataLoader;
use crate::neunet::transforms::normalize::Normalizer;
use crate::neunet::utils::matrix::{MatrixUtil, VectorUtil};
use crate::neunet::utils::ml::MLOps;

mod neunet;

fn main() {
    const INC: u64 = 11634580027462260723;
    let mut rng = rand_pcg::Pcg32::new(213424234, INC);

    let rand_epsilon = 0.03_f64;

    let start = SystemTime::now().duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let train = IdxFile.load_data(String::from("./train-images-idx3-ubyte"), String::from("./train-labels-idx1-ubyte"));
    let end = SystemTime::now().duration_since(UNIX_EPOCH)
        .expect("Time went backwards");


    let test = IdxFile.load_data(String::from("./t10k-images-idx3-ubyte"), String::from("./t10k-labels-idx1-ubyte"));

    println!("Data read in {} ms", (end - start).as_millis());
    let (mut train_data, mut train_labels) = train.unwrap();
    let (mut test_data, mut test_labels) = test.unwrap();

    println!("Shapes {:?}", train_data.shape());
    let r = train_data.column(30);
    let label = train_labels[30];


    //println!("Img {}", DMatrix::from_row_slice(28, 28, r.as_slice()));
    //println!("Label {}", label);


    let arch = NeuralNetworkArchitecture::<HeUniform> {
        num_features: 784,
        num_classes: 10,
        rand_initializer: HeUniform {},
        layers: vec![
            LayerDefinition {
                activation_type: ActivationType::Relu,
                num_activations: 50,
            },
            LayerDefinition {
                activation_type: ActivationType::Relu,
                num_activations: 50,
            },
            LayerDefinition {
                activation_type: ActivationType::SoftMax,
                num_activations: 10,
            }],
    };

    println!("NeuralNetwork {:?}", arch);


    let mut nn = NNModel::build(arch, &mut rng);

    let mut confusion = DMatrix::<usize>::zeros(nn.num_features, nn.num_features);

    Normalizer::min_max(&mut train_data);
    Normalizer::min_max(&mut test_data);

    let labels_one_hot = MatrixUtil::one_hot(&train_labels);

    let test_labels_one_hot = MatrixUtil::one_hot(&test_labels);

    let training_examples = train_data.shape().1;

    nn.train(
        HyperParams {
            momentum_beta: 0.9_f64,
            mini_batch_size: 200,
            learning_rate: 0.05_f64,
            l2_regularization: None,
        },
        LabeledData {
            features: train_data.slice((0, 0), (nn.num_features, training_examples)),
            labels: labels_one_hot.slice((0, 0), (nn.num_classes, training_examples)),
        },
        LabeledData {
            features: test_data.slice((0, 0), (nn.num_features, 200)),
            labels: test_labels_one_hot.slice((0, 0), (nn.num_classes, 200)),
        });
}