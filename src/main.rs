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
    let (mut img, mut labels) = train.unwrap();
    let (mut test_data, mut test_labels) = test.unwrap();

    println!("Shapes {:?}", img.shape());
    let r = img.column(30);
    let label = labels[30];


    println!("Img {}", DMatrix::from_row_slice(28, 28, r.as_slice()));
    println!("Label {}", label);


    let arch = NeuralNetworkArchitecture::<GlorotUniform> {
        num_features: 784,
        num_classes: 10,
        rand_initializer: GlorotUniform {},
        layers: vec![
            LayerDefinition {
                activation_type: ActivationType::Relu,
                num_activations: 128,
            },
            LayerDefinition {
                activation_type: ActivationType::SoftMax,
                num_activations: 10,
            }],
    };

    println!("NeuralNetwork {:?}", arch);

    let mut nn = NNModel::build(arch, &mut rng);


    let mut confusion = DMatrix::<usize>::zeros(nn.num_features, nn.num_features);

    Normalizer::min_max(&mut img);

    let one_hot = MatrixUtil::one_hot(&labels);

    let test_one_hot = MatrixUtil::one_hot(&test_labels);

    nn.train(
        HyperParams {
            momentum_beta: 0.9_f64,
            mini_batch_size: 500,
            learning_rate: 0.1_f64,
            regularization_type: RegularizationType::No_Regularization,
            lambda_regularization: 10.0,
        },
        LabeledData {
            features: img,
            labels: one_hot,
        },
        LabeledData {
            features: test_data,
            labels: test_one_hot,
        });
}