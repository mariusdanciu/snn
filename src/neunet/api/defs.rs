use nalgebra::{DMatrix, DVector};
use nalgebra::*;
use rand_distr::{Normal, Distribution};

#[allow(dead_code)]
pub enum ActivationType {
    Sigmoid,
    Relu,
    Tanh,
    SoftMax,
}

pub struct NNModel {
    pub num_features: usize,
    pub num_classes: usize,
    pub layers: Vec<Layer>,
}


pub struct HyperParams {
    // 0.9 typically
    pub max_accuracy_threshold: f32,
    pub max_epochs: u32,
    pub momentum_beta: Option<f32>,
    pub mini_batch_size: usize,
    pub learning_rate: f32,
    pub l2_regularization: Option<f32>,
}

impl Default for HyperParams {
    fn default() -> Self {
        HyperParams {
            max_accuracy_threshold: 0.95,
            max_epochs: 3,
            momentum_beta: None,
            mini_batch_size: 200,
            learning_rate: 0.05,
            l2_regularization: None,
        }
    }
}

#[derive(Debug)]
pub struct LabeledData<'a> {
    pub features: DMatrixSlice<'a, f32>,
    pub labels: DMatrixSlice<'a, f32>,
}

pub trait Train {
    fn train(&mut self,
             hp: HyperParams,
             train_data: LabeledData,
             test_data: LabeledData) -> Result<&NNModel, Box<dyn std::error::Error>>;
}

pub trait Prediction {
    fn predict(&mut self, data: &DMatrix<f32>) -> DMatrix<f32>;
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

pub trait RandomInitializer {
    fn weights(self, r: usize, c: usize, rng: &mut rand_pcg::Pcg32) -> DMatrix<f32>;
}

#[derive(Debug)]
pub struct HeUniform;


impl RandomInitializer for HeUniform {
    fn weights(self, r: usize, c: usize, rng: &mut rand_pcg::Pcg32) -> DMatrix<f32> {
        let normal = Normal::new(0.0, 1.0).unwrap();;

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