# CUDA implementation of MNIST trainer on Rust

Run the trainer using:

```
cargo clean # To clean (optional)
cargo add burn-cuda
cargo add burn-autodiff 
cargo build
cargo run --release
```

![Training](assets/training.gif)