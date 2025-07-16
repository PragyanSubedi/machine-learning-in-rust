Add burn as a dependency inside `getting_started` folder:

```
cargo add burn --features wgpu
cargo build
cargo run
```

Output:

```
Tensor {
  data:
[[3.0, 4.0],
 [5.0, 6.0]],
  shape:  [2, 2],
  device:  DefaultDevice,
  backend:  "fusion<cubecl<wgpu<wgsl>>>",
  kind:  "Float",
  dtype:  "f32",
}
```