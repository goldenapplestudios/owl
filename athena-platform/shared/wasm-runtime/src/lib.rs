use anyhow::{Context, Result};
use wasmtime::component::*;
use wasmtime::{Config, Engine, Store};
use tracing::{info, debug, error};
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct WasmRuntime {
    engine: Engine,
    config: RuntimeConfig,
}

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub enable_gpu: bool,
    pub enable_simd: bool,
    pub enable_nn: bool,
    pub max_memory_pages: u32,
    pub enable_cache: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            enable_gpu: true,
            enable_simd: true,
            enable_nn: true,
            max_memory_pages: 256,
            enable_cache: true,
        }
    }
}

pub struct ComponentInstance<T> {
    instance: Instance,
    store: Store<T>,
}

impl WasmRuntime {
    pub fn new(config: RuntimeConfig) -> Result<Self> {
        let mut engine_config = Config::new();
        
        engine_config.wasm_component_model(true);
        engine_config.async_support(true);
        
        if config.enable_simd {
            engine_config.wasm_simd(true);
        }
        
        if config.enable_cache {
            engine_config.cache_config_load_default()?;
        }
        
        engine_config.wasm_memory64(true);
        engine_config.wasm_threads(true);
        
        let engine = Engine::new(&engine_config)?;
        
        info!("WASM runtime initialized with config: {:?}", config);
        
        Ok(Self { engine, config })
    }
    
    pub async fn load_component(&self, bytes: &[u8]) -> Result<Component> {
        debug!("Loading WASM component, size: {} bytes", bytes.len());
        
        let component = Component::from_binary(&self.engine, bytes)
            .context("Failed to load WASM component")?;
            
        info!("Component loaded successfully");
        Ok(component)
    }
    
    pub async fn instantiate_component<T: Send + 'static>(
        &self,
        component: &Component,
        state: T,
        linker: &Linker<T>,
    ) -> Result<ComponentInstance<T>> {
        let mut store = Store::new(&self.engine, state);
        store.set_fuel(u64::MAX)?;
        
        let instance = linker
            .instantiate_async(&mut store, component)
            .await
            .context("Failed to instantiate component")?;
            
        info!("Component instantiated successfully");
        
        Ok(ComponentInstance { instance, store })
    }
    
    pub fn create_linker<T>(&self) -> Linker<T> {
        Linker::new(&self.engine)
    }
}

pub struct AgentRuntime {
    runtime: Arc<WasmRuntime>,
    components: Arc<RwLock<Vec<LoadedComponent>>>,
}

struct LoadedComponent {
    name: String,
    component: Component,
    metadata: ComponentMetadata,
}

#[derive(Clone, Debug)]
pub struct ComponentMetadata {
    pub agent_type: String,
    pub version: String,
    pub capabilities: Vec<String>,
}

impl AgentRuntime {
    pub async fn new(config: RuntimeConfig) -> Result<Self> {
        let runtime = Arc::new(WasmRuntime::new(config)?);
        let components = Arc::new(RwLock::new(Vec::new()));
        
        Ok(Self { runtime, components })
    }
    
    pub async fn register_component(
        &self,
        name: String,
        bytes: &[u8],
        metadata: ComponentMetadata,
    ) -> Result<()> {
        let component = self.runtime.load_component(bytes).await?;
        
        let loaded = LoadedComponent {
            name: name.clone(),
            component,
            metadata,
        };
        
        self.components.write().await.push(loaded);
        info!("Registered component: {}", name);
        
        Ok(())
    }
    
    pub async fn get_component(&self, name: &str) -> Result<Option<Component>> {
        let components = self.components.read().await;
        Ok(components
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.component.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_runtime_creation() {
        let config = RuntimeConfig::default();
        let runtime = WasmRuntime::new(config).unwrap();
        assert!(runtime.engine.config().wasm_component_model());
    }
    
    #[tokio::test] 
    async fn test_agent_runtime() {
        let config = RuntimeConfig::default();
        let runtime = AgentRuntime::new(config).await.unwrap();
        assert_eq!(runtime.components.read().await.len(), 0);
    }
}