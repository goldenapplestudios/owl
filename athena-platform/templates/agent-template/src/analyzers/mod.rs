use anyhow::Result;
use async_trait::async_trait;

#[async_trait]
pub trait Analyzer: Send + Sync {
    type Target;
    type Finding;
    
    async fn analyze(&self, target: Self::Target) -> Result<Vec<Self::Finding>>;
}