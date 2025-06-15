# WASI-NN Integration Guide

## Overview

WASI-NN (WebAssembly System Interface for Neural Networks) enables running ML models efficiently within WASM components.

## Supported Backends

- ONNX Runtime (CPU/CUDA/TensorRT)
- PyTorch
- TensorFlow Lite
- Candle (Rust-native)

## Model Loading

```rust
use athena_wasi_nn_engine::{WasiNNEngine, ModelType, ModelMetadata};

let engine = WasiNNEngine::new(Default::default()).await?;

let metadata = ModelMetadata {
    input_shapes: vec![vec![1, 3, 224, 224]],
    output_shapes: vec![vec![1, 1000]],
    input_names: vec!["input".to_string()],
    output_names: vec!["output".to_string()],
    precision: Precision::FP16,
};

engine.load_model(
    "classifier".to_string(),
    model_bytes,
    ModelType::ONNX,
    metadata,
).await?;
```

## Inference

```rust
use athena_wasi_nn_engine::Tensor;

let input = Tensor::new(input_data, vec![1, 3, 224, 224]);
let outputs = engine.infer("classifier", vec![input]).await?;
```

## GPU Acceleration

Enable GPU support in Cargo.toml:

```toml
[features]
default = ["gpu"]
gpu = ["ort/cuda", "candle-core/cuda"]
```

## Model Optimization

1. **Quantization**: Convert FP32 to INT8/FP16
2. **Graph optimization**: Fuse operations
3. **TensorRT**: Automatic kernel optimization
4. **Batching**: Process multiple inputs

## Agent-Specific Models

### Doru (Malware RE)
- PE header classifier
- Opcode sequence analyzer
- Packing detector

### Aegis (Threat Analysis)
- IOC classifier
- TTP mapper
- Campaign correlator

### Owl (Security Testing)
- Vulnerability predictor
- Exploit feasibility scorer
- False positive reducer

## Performance Tips

1. Pre-load models during initialization
2. Use appropriate batch sizes
3. Enable hardware acceleration
4. Cache inference results
5. Use model ensembles wisely