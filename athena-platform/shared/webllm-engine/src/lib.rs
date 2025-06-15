use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_stream::Stream;
use tracing::{debug, info};
use tera::{Context as TeraContext, Tera};

#[derive(Debug, Clone)]
pub struct WebLLMEngine {
    config: EngineConfig,
    models: Arc<RwLock<HashMap<String, LoadedLLM>>>,
    prompt_engine: Arc<PromptEngine>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub enable_webgpu: bool,
    pub max_sequence_length: usize,
    pub max_batch_size: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub enable_streaming: bool,
    pub cache_size_mb: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            enable_webgpu: true,
            max_sequence_length: 4096,
            max_batch_size: 8,
            temperature: 0.7,
            top_p: 0.9,
            enable_streaming: true,
            cache_size_mb: 512,
        }
    }
}

struct LoadedLLM {
    name: String,
    model_info: ModelInfo,
    backend: LLMBackend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_id: String,
    pub model_type: String,
    pub quantization: Option<String>,
    pub context_length: usize,
    pub vocab_size: usize,
}

enum LLMBackend {
    WebGPU(WebGPUBackend),
    CPU(CPUBackend),
}

struct WebGPUBackend {
    tokenizer: tokenizers::Tokenizer,
}

struct CPUBackend {
    tokenizer: tokenizers::Tokenizer,
}

impl WebLLMEngine {
    pub async fn new(config: EngineConfig) -> Result<Self> {
        info!("Initializing WebLLM engine with config: {:?}", config);
        
        let prompt_engine = Arc::new(PromptEngine::new()?);
        
        Ok(Self {
            config,
            models: Arc::new(RwLock::new(HashMap::new())),
            prompt_engine,
        })
    }
    
    pub async fn load_model(
        &self,
        name: String,
        model_info: ModelInfo,
        model_path: &str,
    ) -> Result<()> {
        debug!("Loading LLM model: {}", name);
        
        let tokenizer = tokenizers::Tokenizer::from_file(format!("{}/tokenizer.json", model_path))
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        
        let backend = if self.config.enable_webgpu {
            LLMBackend::WebGPU(WebGPUBackend { tokenizer })
        } else {
            LLMBackend::CPU(CPUBackend { tokenizer })
        };
        
        let loaded_llm = LoadedLLM {
            name: name.clone(),
            model_info,
            backend,
        };
        
        self.models.write().await.insert(name.clone(), loaded_llm);
        info!("Model loaded successfully: {}", name);
        
        Ok(())
    }
    
    pub async fn generate(
        &self,
        model_name: &str,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<GenerationResponse> {
        let models = self.models.read().await;
        let model = models
            .get(model_name)
            .context(format!("Model not found: {}", model_name))?;
        
        debug!("Generating with model: {}", model_name);
        
        let tokens = match &model.backend {
            LLMBackend::WebGPU(backend) => {
                self.generate_webgpu(backend, prompt, params).await?
            }
            LLMBackend::CPU(backend) => {
                self.generate_cpu(backend, prompt, params).await?
            }
        };
        
        Ok(GenerationResponse {
            text: tokens,
            model: model_name.to_string(),
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        })
    }
    
    async fn generate_webgpu(
        &self,
        backend: &WebGPUBackend,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<String> {
        let encoding = backend.tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        
        Ok(format!("Generated response for: {}", prompt))
    }
    
    async fn generate_cpu(
        &self,
        backend: &CPUBackend,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<String> {
        let encoding = backend.tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        
        Ok(format!("Generated response for: {}", prompt))
    }
    
    pub async fn stream_generate(
        &self,
        model_name: &str,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<impl Stream<Item = Result<String>>> {
        let response = self.generate(model_name, prompt, params).await?;
        let chunks: Vec<Result<String>> = response.text
            .split_whitespace()
            .map(|s| Ok(s.to_string()))
            .collect();
        
        Ok(tokio_stream::iter(chunks))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub stop_sequences: Vec<String>,
    pub stream: bool,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 1024,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: vec![],
            stream: false,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerationResponse {
    pub text: String,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

pub struct PromptEngine {
    templates: Tera,
}

impl PromptEngine {
    pub fn new() -> Result<Self> {
        let mut tera = Tera::default();
        
        tera.add_raw_template("analysis", r#"
You are a security analysis assistant for the {{ agent_type }} agent.

Context: {{ context }}

Task: {{ task }}

Please provide a detailed analysis following these guidelines:
{{ guidelines }}

Input: {{ input }}
"#)?;
        
        tera.add_raw_template("code_generation", r#"
Generate secure {{ language }} code for the following task:

Requirements:
{{ requirements }}

Security considerations:
- Input validation
- Error handling
- Resource limits
- Authentication/authorization

Task: {{ task }}
"#)?;
        
        Ok(Self { templates: tera })
    }
    
    pub fn render_prompt(
        &self,
        template_name: &str,
        context: HashMap<String, String>,
    ) -> Result<String> {
        let mut tera_context = TeraContext::new();
        for (key, value) in context {
            tera_context.insert(key, &value);
        }
        
        self.templates
            .render(template_name, &tera_context)
            .context("Failed to render prompt template")
    }
}

#[async_trait]
pub trait LLMProvider {
    async fn complete(&self, prompt: &str, params: GenerationParams) -> Result<String>;
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

pub struct ContextManager {
    max_context_length: usize,
    history: Arc<RwLock<Vec<Message>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub timestamp: u64,
}

impl ContextManager {
    pub fn new(max_context_length: usize) -> Self {
        Self {
            max_context_length,
            history: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    pub async fn add_message(&self, role: String, content: String) {
        let message = Message {
            role,
            content,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        self.history.write().await.push(message);
    }
    
    pub async fn get_context(&self, max_messages: usize) -> Vec<Message> {
        let history = self.history.read().await;
        let start = history.len().saturating_sub(max_messages);
        history[start..].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_engine_creation() {
        let config = EngineConfig::default();
        let engine = WebLLMEngine::new(config).await.unwrap();
        assert!(engine.models.read().await.is_empty());
    }
    
    #[tokio::test]
    async fn test_prompt_engine() {
        let engine = PromptEngine::new().unwrap();
        let mut context = HashMap::new();
        context.insert("agent_type".to_string(), "malware analysis".to_string());
        context.insert("context".to_string(), "PE file analysis".to_string());
        context.insert("task".to_string(), "Analyze suspicious behavior".to_string());
        context.insert("guidelines".to_string(), "Focus on API calls".to_string());
        context.insert("input".to_string(), "sample.exe".to_string());
        
        let prompt = engine.render_prompt("analysis", context).unwrap();
        assert!(prompt.contains("malware analysis"));
    }
}