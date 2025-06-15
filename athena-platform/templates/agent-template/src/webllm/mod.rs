use anyhow::Result;
use athena_webllm_engine::{WebLLMEngine, GenerationParams};

pub struct LLMAssistant {
    engine: WebLLMEngine,
    model_name: String,
}

impl LLMAssistant {
    pub fn new(engine: WebLLMEngine, model_name: String) -> Self {
        Self { engine, model_name }
    }
    
    pub async fn analyze(&self, prompt: &str) -> Result<String> {
        let params = GenerationParams::default();
        let response = self.engine.generate(&self.model_name, prompt, params).await?;
        Ok(response.text)
    }
}