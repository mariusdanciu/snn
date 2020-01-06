use nalgebra::{DMatrix, DVector, DVectorSlice};

pub struct MLOps;

impl MLOps {
    pub fn hypothesis(w: &DVector<f64>, x: &DVectorSlice<f64>, b: f64) -> f64 {
        w.dot(x) + b
    }

    pub fn vectorize(v: &DVector<f64>, f: fn(f64) -> f64) -> DVector<f64> {
        v.map(|e| f(e))
    }

    pub fn sigmoid(z: f64) -> f64 {
        1.0_f64 / (1.0_f64 + (-z).exp())
    }

    pub fn sigmoid_derivative(z: f64) -> f64 {
        let s = MLOps::sigmoid(z);
        s * (1.0_f64 - s)
    }

    pub fn relu(z: f64) -> f64 {
        z.max(0.0_f64)
    }
    pub fn relu_derivative(z: f64) -> f64 {
        if z >= 0.0_f64 {
            1.0_f64
        } else {
            0.0_f64
        }
    }

    pub fn tanh(z: f64) -> f64 {
        z.tanh()
    }
    pub fn tanh_derivative(z: f64) -> f64 {
        1.0_f64 - z.tanh().powi(2)
    }

    pub fn soft_max(v: &DVector<f64>) -> DVector<f64> {
        let mut sum = 0.0_f64;
        for e in v.iter() {
            sum += e.exp();
        }

        v / sum
    }

    pub fn soft_max_derivative(v: &DVector<f64>) -> DVector<f64> {
        let sm = MLOps::soft_max(&v);

        sm.map(|e| e * (1. - e))
    }

    pub fn loss(y: f64, w: &DVector<f64>, x: &DVectorSlice<f64>, b: f64) -> f64 {
        let y_hat = MLOps::sigmoid(MLOps::hypothesis(w, x, b));
        -(y * y_hat.ln() + (1. - y) * (1. - y_hat).ln())
    }

    pub fn loss_from_pred(y: f64, y_hat: f64) -> f64 {
        -(y * y_hat.ln() + (1. - y) * (1. - y_hat).ln())
    }
}

pub enum OptimizationType {
    StochasticGradientDescent,
    BatchGradientDescent,
    Adam,
}

pub enum ActivationType {
    Sigmoid,
    Relu,
    Tanh,
    SoftMax
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

pub struct NNLayer {
    pub intercepts: DVector<f64>,
    pub weights: DMatrix<f64>,
    pub activation_type: ActivationType,
}

pub struct NNModel {
    pub layers: Vec<NNLayer>
}

pub struct NeuralNetworkDefinition {
    pub rand_init_epsilon: f64,
    pub num_features: usize,
    pub layers_dimensions: Vec<usize>,
    pub activation_type: ActivationType,
}


