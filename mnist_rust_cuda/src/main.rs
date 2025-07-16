mod model;
mod inference;
mod training;
mod data;
use crate::{model::ModelConfig, training::TrainingConfig};
use burn::{
    optim::AdamConfig,
    data::dataset::Dataset, 
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

    crate::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}