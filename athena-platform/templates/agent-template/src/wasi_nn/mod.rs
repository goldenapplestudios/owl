use anyhow::Result;
use athena_wasi_nn_engine::{WasiNNEngine, Tensor};

pub struct ModelInference {
    engine: WasiNNEngine,
}

impl ModelInference {
    pub async fn new(engine: WasiNNEngine) -> Self {
        Self { engine }
    }
    
    pub async fn predict(&self, model_name: &str, input: Tensor) -> Result<Tensor> {
        let outputs = self.engine.infer(model_name, vec![input]).await?;
        Ok(outputs.into_iter().next().unwrap())
    }
}