use anyhow::{Context, Result};
use tracing::{info, debug, error};
use std::sync::Arc;
use tokio::sync::RwLock;
use bytes::Bytes;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone)]
pub struct WasiNNEngine {
    config: EngineConfig,
    models: Arc<RwLock<HashMap<String, LoadedModel>>>,
    device: Device,
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub enable_gpu: bool,
    pub gpu_device_id: Option<usize>,
    pub max_batch_size: usize,
    pub enable_tensorrt: bool,
    pub enable_fp16: bool,
    pub cache_models: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            enable_gpu: true,
            gpu_device_id: Some(0),
            max_batch_size: 32,
            enable_tensorrt: true,
            enable_fp16: true,
            cache_models: true,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    Cuda(usize),
}

#[derive(Clone)]
struct LoadedModel {
    name: String,
    model_type: ModelType,
    metadata: ModelMetadata,
    session: Arc<RwLock<InferenceSession>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    ONNX,
    PyTorch,
    TensorFlow,
    Candle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub input_shapes: Vec<Vec<i64>>,
    pub output_shapes: Vec<Vec<i64>>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub precision: Precision,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Precision {
    FP32,
    FP16,
    INT8,
}

pub struct InferenceSession {
    backend: Backend,
}

enum Backend {
    #[cfg(feature = "gpu")]
    ONNX(ort::Session),
    #[cfg(feature = "gpu")]
    Candle(candle_core::Device),
    CPU,
}

impl WasiNNEngine {
    pub async fn new(config: EngineConfig) -> Result<Self> {
        let device = if config.enable_gpu {
            #[cfg(feature = "gpu")]
            {
                if ort::providers::CUDAExecutionProvider::is_available()? {
                    info!("GPU acceleration enabled with CUDA");
                    Device::Cuda(config.gpu_device_id.unwrap_or(0))
                } else {
                    info!("GPU requested but not available, falling back to CPU");
                    Device::Cpu
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                info!("GPU support not compiled in, using CPU");
                Device::Cpu
            }
        } else {
            Device::Cpu
        };
        
        Ok(Self {
            config,
            models: Arc::new(RwLock::new(HashMap::new())),
            device,
        })
    }
    
    pub async fn load_model(
        &self,
        name: String,
        model_bytes: &[u8],
        model_type: ModelType,
        metadata: ModelMetadata,
    ) -> Result<()> {
        debug!("Loading model: {} ({:?})", name, model_type);
        
        let session = match model_type {
            ModelType::ONNX => {
                #[cfg(feature = "gpu")]
                {
                    let mut builder = ort::SessionBuilder::new(&ort::Environment::builder().build()?)?;
                    
                    if let Device::Cuda(device_id) = &self.device {
                        builder = builder.with_execution_providers([
                            ort::CUDAExecutionProvider::default()
                                .with_device_id(*device_id as i32)
                                .with_gpu_mem_limit(usize::MAX)
                                .build(),
                        ])?;
                        
                        if self.config.enable_tensorrt {
                            builder = builder.with_optimization_level(ort::GraphOptimizationLevel::Level3)?;
                        }
                    }
                    
                    let session = builder.with_model_from_memory(model_bytes)?;
                    InferenceSession {
                        backend: Backend::ONNX(session),
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    InferenceSession { backend: Backend::CPU }
                }
            }
            ModelType::Candle => {
                #[cfg(feature = "gpu")]
                {
                    let device = match &self.device {
                        Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
                        Device::Cpu => candle_core::Device::Cpu,
                    };
                    InferenceSession {
                        backend: Backend::Candle(device),
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    InferenceSession { backend: Backend::CPU }
                }
            }
            _ => InferenceSession { backend: Backend::CPU },
        };
        
        let loaded_model = LoadedModel {
            name: name.clone(),
            model_type,
            metadata,
            session: Arc::new(RwLock::new(session)),
        };
        
        self.models.write().await.insert(name.clone(), loaded_model);
        info!("Model loaded successfully: {}", name);
        
        Ok(())
    }
    
    pub async fn infer(
        &self,
        model_name: &str,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>> {
        let models = self.models.read().await;
        let model = models
            .get(model_name)
            .context(format!("Model not found: {}", model_name))?;
            
        debug!("Running inference on model: {}", model_name);
        
        let session = model.session.read().await;
        
        match &session.backend {
            #[cfg(feature = "gpu")]
            Backend::ONNX(ort_session) => {
                let input_values: Vec<ort::Value> = inputs
                    .into_iter()
                    .map(|tensor| tensor.to_ort_value())
                    .collect::<Result<Vec<_>>>()?;
                    
                let outputs = ort_session.run(input_values)?;
                
                Ok(outputs
                    .into_iter()
                    .map(Tensor::from_ort_value)
                    .collect::<Result<Vec<_>>>()?)
            }
            _ => {
                error!("Inference backend not implemented");
                anyhow::bail!("Inference backend not implemented")
            }
        }
    }
    
    pub async fn list_models(&self) -> Vec<String> {
        self.models.read().await.keys().cloned().collect()
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<i64>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<i64>) -> Self {
        Self { data, shape }
    }
    
    #[cfg(feature = "gpu")]
    fn to_ort_value(&self) -> Result<ort::Value> {
        let array = ndarray::ArrayD::from_shape_vec(
            self.shape.iter().map(|&x| x as usize).collect::<Vec<_>>(),
            self.data.clone(),
        )?;
        Ok(ort::Value::from_array(array)?)
    }
    
    #[cfg(feature = "gpu")]
    fn from_ort_value(value: ort::Value) -> Result<Self> {
        let array = value.try_extract::<f32>()?;
        let shape = array.shape().iter().map(|&x| x as i64).collect();
        let data = array.as_slice()?.to_vec();
        Ok(Self { data, shape })
    }
}

pub struct ModelConverter {
    config: EngineConfig,
}

impl ModelConverter {
    pub fn new(config: EngineConfig) -> Self {
        Self { config }
    }
    
    pub async fn optimize_for_inference(
        &self,
        model_bytes: &[u8],
        model_type: ModelType,
    ) -> Result<Vec<u8>> {
        match model_type {
            ModelType::ONNX => {
                info!("Optimizing ONNX model for inference");
                Ok(model_bytes.to_vec())
            }
            _ => Ok(model_bytes.to_vec()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_engine_creation() {
        let config = EngineConfig::default();
        let engine = WasiNNEngine::new(config).await.unwrap();
        assert_eq!(engine.list_models().await.len(), 0);
    }
    
    #[tokio::test]
    async fn test_tensor_creation() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(tensor.data.len(), 4);
        assert_eq!(tensor.shape, vec![2, 2]);
    }
}