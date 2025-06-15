use anyhow::Result;
use async_trait::async_trait;

#[async_trait]
pub trait Processor: Send + Sync {
    type Input;
    type Output;
    
    async fn process(&self, input: Self::Input) -> Result<Self::Output>;
}