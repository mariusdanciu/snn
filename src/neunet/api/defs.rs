use nalgebra::{DMatrix, DVector, DVectorSlice};
use nalgebra::*;
use rand::distributions::{Distribution, Normal};
use rand::Rng;

use crate::neunet::utils::ml::MLOps;

pub enum OptimizationType {
    StochasticGradientDescent,
    BatchGradientDescent,
    Adam,
}

pub enum ActivationType {
    Sigmoid,
    Relu,
    Tanh,
    SoftMax,
}

pub enum Metric {
    accuracy,
    precision,
    recall,
    f1_score,
    au_roc,
}

pub struct NNModel {
    pub num_features: usize,
    pub num_classes: usize,
    pub layers: Vec<Layer>,
}

pub struct HyperParams {
    pub momentum_beta: f32,
    pub mini_batch_size: usize,
    pub learning_rate: f32,
    pub l2_regularization: Option<f32>,
}

#[derive(Debug)]
pub struct LabeledData<'a> {
    pub features: DMatrixSlice<'a, f32>,
    pub labels: DMatrixSlice<'a, f32>,
}

pub struct Eval<'a> {
    pub test_data: LabeledData<'a>,
    pub metrics: Vec<Metric>,
}

pub trait Train {
    fn train(&mut self,
             hp: HyperParams,
             train_data: LabeledData,
             test_data: LabeledData) -> &NNModel;
}

pub trait Prediction {
    fn predict(&mut self, data: &DMatrix<f32>) -> DMatrix<f32>;
}


#[derive(Debug)]
pub struct EvalData {
    pub probabilities: DMatrix<f32>,
    pub truth_one_hot: DMatrix<f32>,
}

pub struct Layer {
    pub num_activations: usize,
    pub intercepts: DVector<f32>,
    pub weights: DMatrix<f32>,
    pub activation_type: ActivationType,

    // W * X + B
    pub z: DVector<f32>,
    // activation(z)
    pub a: DVector<f32>,
    pub dz: DVector<f32>,
    pub dw: DMatrix<f32>,
    pub db: DVector<f32>,

    pub momentum_dw: DMatrix<f32>,
    pub momentum_db: DVector<f32>,
}

#[derive(Debug)]
pub struct LayerDefinition {
    pub activation_type: ActivationType,
    pub num_activations: usize,
}

#[derive(Debug)]
pub struct NeuralNetworkArchitecture<R: RandomInitializer> {
    pub num_features: usize,
    pub num_classes: usize,
    pub layers: Vec<LayerDefinition>,
    pub rand_initializer: R,
}

pub enum RegularizationType {
    No_Regularization,
    L2,
}

pub trait ConfusionMatrix {}

impl ConfusionMatrix {
    pub fn build(num_labels: usize) -> DMatrix<usize> {
        let v = vec![0; num_labels * num_labels];
        DMatrix::from_vec(num_labels, num_labels, v)
    }
}

pub trait RandomInitializer {
    fn weights(self, r: usize, c: usize, rng: &mut rand_pcg::Pcg32) -> DMatrix<f32>;
}

#[derive(Debug)]
pub struct HeUniform;


impl RandomInitializer for HeUniform {
    fn weights(self, r: usize, c: usize, rng: &mut rand_pcg::Pcg32) -> DMatrix<f32> {
        let normal = Normal::new(0.0, 1.0);

        let factor = (2.0 / c as f32).sqrt();

        let v: Vec<f32> = (0..(r * c)).map(|_| {
            let r = normal.sample(rng) as f32;
            r * factor
        }).collect();
        let m = DMatrix::from_vec(r, c, v);
        // println!("W {}", m);
        m
    }
}

impl Copy for HeUniform {}

impl Clone for HeUniform {
    fn clone(&self) -> Self {
        HeUniform {}
    }
}

impl std::fmt::Debug for ActivationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActivationType::Sigmoid => write!(f, "Sigmoid"),
            ActivationType::Relu => write!(f, "Relu"),
            ActivationType::Tanh => write!(f, "Tanh"),
            ActivationType::SoftMax => write!(f, "SoftMax"),
        }
    }
}


impl Clone for ActivationType {
    fn clone(&self) -> ActivationType {
        match self {
            ActivationType::Sigmoid => ActivationType::Sigmoid,
            ActivationType::Relu => ActivationType::Relu,
            ActivationType::Tanh => ActivationType::Tanh,
            ActivationType::SoftMax => ActivationType::SoftMax
        }
    }
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