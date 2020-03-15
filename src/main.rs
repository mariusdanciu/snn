extern crate rand;
extern crate rand_pcg;

use std::ops::IndexMut;
use std::time::{SystemTime, UNIX_EPOCH};

use nalgebra::*;
use rand_pcg::Pcg32;

use crate::neunet::definitions::{ActivationType, LayerDefinition, MLOps, NeuralNetworkDefinition};
use crate::neunet::files::idx::IdxFile;
use crate::neunet::loader::{DataLoader, DataUtils};
use crate::neunet::optimization::*;
use crate::neunet::utils::matrix::MatrixUtil;

mod neunet;

fn main() {
    const INC: u64 = 11634580027462260723;
    let mut rng = rand_pcg::Pcg32::new(213424234, INC);

    let rand_epsilon = 0.03_f64;

    let start = SystemTime::now().duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let res = IdxFile.load_data(String::from("./train-images-idx3-ubyte"), String::from("./train-labels-idx1-ubyte"));
    let end = SystemTime::now().duration_since(UNIX_EPOCH)
        .expect("Time went backwards");


    println!("Data read in {} ms", (end - start).as_millis());
    let (img, mut labels) = res.unwrap();

    println!("Shapes {:?}", img.shape());
    let r = img.column(30);
    let label = labels[30];


    println!("Img {}", DMatrix::from_row_slice(28, 28, r.as_slice()));
    println!("Label {}", label);

    let mut nn = NeuralNetwork::build(&NeuralNetworkDefinition {
        num_features: 784,
        layers: vec![
            LayerDefinition {
                activation_type: ActivationType::Relu,
                num_activations: 80,
                rand_init_epsilon: 0.03_f64,
            },
            LayerDefinition {
                activation_type: ActivationType::Relu,
                num_activations: 40,
                rand_init_epsilon: 0.03_f64,
            },
            LayerDefinition {
                activation_type: ActivationType::SoftMax,
                num_activations: 10,
                rand_init_epsilon: 0.03_f64,
            }],
    }, &mut rng);

    let gd = GradientDescent {
        momentum_beta: 0.9_f64,
        rms_prop_beta: 0.9_f64,
        epsilon_correction: 0.9_f64,
        mini_batch_size: 1000,
        learning_rate: 0.001_f64,
        stop_cost_quota: 0.0001_f64,
    };

    let one_hot = DataUtils::one_hot(&labels);
    gd.optimize(&mut nn, &img, &one_hot);



    println!("NeuralNetwork {:?}", nn);



    println!("OneHot {}", one_hot);


}