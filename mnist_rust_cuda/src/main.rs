#![recursion_limit = "512"]  
mod model;
mod training;
mod data;
use crate::{model::ModelConfig, training::TrainingConfig};
use burn::{
    optim::AdamConfig,
};
use burn_autodiff::Autodiff;
use burn_cuda::{Cuda, CudaDevice};

fn main() {
    type MyBackend = Cuda<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = CudaDevice::default();
    let artifact_dir = "/tmp/guide";
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
}