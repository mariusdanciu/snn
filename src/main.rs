extern crate rand;
extern crate rand_pcg;

use std::ops::IndexMut;
use std::time::{SystemTime, UNIX_EPOCH};

use nalgebra::*;
use rand::distributions::{Distribution, Normal};
use rand_pcg::Pcg32;

use crate::neunet::api::defs::{ActivationType, ConfusionMatrix, GlorotUniform, LayerDefinition, Metric, NeuralNetworkDefinition, RandomInitializer};
use crate::neunet::files::idx::IdxFile;
use crate::neunet::loader::DataLoader;
use crate::neunet::optimization::*;
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

    println!("Shapes {:?}", img.shape());
    let r = img.column(30);
    let label = labels[30];


    println!("Img {}", DMatrix::from_row_slice(28, 28, r.as_slice()));
    println!("Label {}", label);

    let mut nn = NeuralNetwork::build(NeuralNetworkDefinition::<GlorotUniform> {
        num_features: 784,
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
    }, &mut rng);

    fn test_acc(pred: &DVector<f64>) -> Vec<Metric> {
        let m = Metric {
            name: String::from("accuracy"),
            value: 0.0,
        };

        vec![m]
    }

    let gd = GradientDescent {
        momentum_beta: 0.9_f64,
        mini_batch_size: 500,
        learning_rate: 0.1_f64,
        test_func: test_acc,
        regularization_type: RegularizationType::No_Regularization,
        lambda_regularization: 10.0,
    };

    println!("NeuralNetwork {:?}", nn);

    Normalizer::min_max(&mut img);

    let one_hot = MatrixUtil::one_hot(&labels);
    gd.optimize(&mut nn, &img, &one_hot);


    //let r1 = img.column(30);
    //println!("Img 2 {}", DMatrix::from_row_slice(28, 28, r1.as_slice()));
}