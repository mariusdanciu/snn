use nalgebra::{DMatrix, DVector, DVectorSlice};
use nalgebra::*;
use rand::Rng;

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


pub struct NNLayer {
    pub intercepts: DVector<f64>,
    pub weights: DMatrix<f64>,
    pub activation_type: ActivationType,
}

pub struct NNModel {
    pub layers: Vec<NNLayer>
}

pub struct LayerDefinition {
    pub activation_type: ActivationType,
    pub num_activations: usize,
}

pub struct NeuralNetworkDefinition<R: RandomInitializer> {
    pub num_features: usize,
    pub layers: Vec<LayerDefinition>,
    pub rand_initializer: R,
}

pub struct Metric {
    pub name: String,
    pub value: f64,
}

pub trait ConfusionMatrix {}

impl ConfusionMatrix {
    pub fn build(num_labels: usize) -> DMatrix<usize> {
        let v = vec![0; num_labels * num_labels];
        DMatrix::from_vec(num_labels, num_labels, v)
    }
}

pub trait RandomInitializer {
    fn weights(self, r: usize, c: usize, rng: &mut rand_pcg::Pcg32) -> DMatrix<f64>;
}

pub struct GlorotUniform;

impl RandomInitializer for GlorotUniform {
    fn weights(self, r: usize, c: usize, rng: &mut rand_pcg::Pcg32) -> DMatrix<f64> {
        let factor = (6f64 / (r + c) as f64).sqrt();

        let v: Vec<f64> = (0..(r * c)).map(|_| rng.gen_range(-factor, factor)).collect();
        DMatrix::from_vec(r, c, v)
    }
}

impl Copy for GlorotUniform {}
impl Clone for GlorotUniform {
    fn clone(&self) -> Self {
        GlorotUniform{}
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