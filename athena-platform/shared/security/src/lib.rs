use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use chrono::{DateTime, Duration, Utc};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use ring::rand::{SecureRandom, SystemRandom};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct SecurityManager {
    config: SecurityConfig,
    token_store: Arc<RwLock<TokenStore>>,
    crypto: CryptoEngine,
}

#[derive(Debug, Clone)]
pub struct SecurityConfig {
    pub jwt_secret: Vec<u8>,
    pub token_expiry_minutes: i64,
    pub max_token_refresh_count: u32,
    pub enable_encryption: bool,
    pub audit_enabled: bool,
}

impl SecurityConfig {
    pub fn new(jwt_secret: &str) -> Self {
        Self {
            jwt_secret: jwt_secret.as_bytes().to_vec(),
            token_expiry_minutes: 60,
            max_token_refresh_count: 3,
            enable_encryption: true,
            audit_enabled: true,
        }
    }
}

#[derive(Debug)]
struct TokenStore {
    tokens: HashMap<String, TokenInfo>,
    revoked: HashMap<String, DateTime<Utc>>,
}

#[derive(Debug, Clone)]
struct TokenInfo {
    token_id: String,
    agent_id: String,
    issued_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
    refresh_count: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub agent_id: String,
    pub agent_type: String,
    pub capabilities: Vec<String>,
    pub exp: i64,
    pub iat: i64,
    pub jti: String,
}

struct CryptoEngine {
    rng: SystemRandom,
}

impl SecurityManager {
    pub fn new(config: SecurityConfig) -> Self {
        Self {
            config,
            token_store: Arc::new(RwLock::new(TokenStore {
                tokens: HashMap::new(),
                revoked: HashMap::new(),
            })),
            crypto: CryptoEngine {
                rng: SystemRandom::new(),
            },
        }
    }
    
    pub async fn create_token(
        &self,
        agent_id: &str,
        agent_type: &str,
        capabilities: Vec<String>,
    ) -> Result<String> {
        let token_id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let expires_at = now + Duration::minutes(self.config.token_expiry_minutes);
        
        let claims = Claims {
            sub: agent_id.to_string(),
            agent_id: agent_id.to_string(),
            agent_type: agent_type.to_string(),
            capabilities,
            exp: expires_at.timestamp(),
            iat: now.timestamp(),
            jti: token_id.clone(),
        };
        
        let token = encode(
            &Header::new(Algorithm::HS512),
            &claims,
            &EncodingKey::from_secret(&self.config.jwt_secret),
        )?;
        
        let token_info = TokenInfo {
            token_id,
            agent_id: agent_id.to_string(),
            issued_at: now,
            expires_at,
            refresh_count: 0,
        };
        
        self.token_store.write().await.tokens.insert(
            token_info.token_id.clone(),
            token_info,
        );
        
        info!("Created token for agent: {}", agent_id);
        Ok(token)
    }
    
    pub async fn verify_token(&self, token: &str) -> Result<Claims> {
        let validation = Validation::new(Algorithm::HS512);
        
        let token_data = decode::<Claims>(
            token,
            &DecodingKey::from_secret(&self.config.jwt_secret),
            &validation,
        )?;
        
        let token_store = self.token_store.read().await;
        
        if token_store.revoked.contains_key(&token_data.claims.jti) {
            anyhow::bail!("Token has been revoked");
        }
        
        if !token_store.tokens.contains_key(&token_data.claims.jti) {
            anyhow::bail!("Token not found in store");
        }
        
        debug!("Token verified for agent: {}", token_data.claims.agent_id);
        Ok(token_data.claims)
    }
    
    pub async fn revoke_token(&self, token_id: &str) -> Result<()> {
        let mut store = self.token_store.write().await;
        store.tokens.remove(token_id);
        store.revoked.insert(token_id.to_string(), Utc::now());
        
        info!("Revoked token: {}", token_id);
        Ok(())
    }
    
    pub async fn refresh_token(&self, old_token: &str) -> Result<String> {
        let claims = self.verify_token(old_token).await?;
        
        let mut store = self.token_store.write().await;
        if let Some(token_info) = store.tokens.get_mut(&claims.jti) {
            if token_info.refresh_count >= self.config.max_token_refresh_count {
                anyhow::bail!("Maximum refresh count exceeded");
            }
            token_info.refresh_count += 1;
        }
        
        self.revoke_token(&claims.jti).await?;
        self.create_token(&claims.agent_id, &claims.agent_type, claims.capabilities).await
    }
}

impl CryptoEngine {
    pub fn generate_key(&self, length: usize) -> Result<Vec<u8>> {
        let mut key = vec![0u8; length];
        self.rng
            .fill(&mut key)
            .context("Failed to generate random key")?;
        Ok(key)
    }
    
    pub fn hash_data(&self, data: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }
    
    pub fn generate_api_key(&self) -> Result<String> {
        let key = self.generate_key(32)?;
        Ok(BASE64.encode(key))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    pub resource: String,
    pub actions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AccessControl {
    policies: Arc<RwLock<HashMap<String, Policy>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    pub id: String,
    pub agent_type: String,
    pub permissions: Vec<Permission>,
}

impl AccessControl {
    pub fn new() -> Self {
        Self {
            policies: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn add_policy(&self, policy: Policy) -> Result<()> {
        self.policies.write().await.insert(policy.id.clone(), policy);
        Ok(())
    }
    
    pub async fn check_permission(
        &self,
        agent_type: &str,
        resource: &str,
        action: &str,
    ) -> Result<bool> {
        let policies = self.policies.read().await;
        
        for policy in policies.values() {
            if policy.agent_type == agent_type {
                for permission in &policy.permissions {
                    if permission.resource == resource && permission.actions.contains(&action.to_string()) {
                        return Ok(true);
                    }
                }
            }
        }
        
        Ok(false)
    }
}

pub struct AuditLogger {
    entries: Arc<RwLock<Vec<AuditEntry>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub agent_id: String,
    pub action: String,
    pub resource: String,
    pub result: AuditResult,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditResult {
    Success,
    Failure(String),
    Denied,
}

impl AuditLogger {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    pub async fn log_action(
        &self,
        agent_id: String,
        action: String,
        resource: String,
        result: AuditResult,
        metadata: Option<serde_json::Value>,
    ) {
        let entry = AuditEntry {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            agent_id,
            action,
            resource,
            result,
            metadata,
        };
        
        self.entries.write().await.push(entry);
    }
    
    pub async fn get_entries(
        &self,
        agent_id: Option<&str>,
        limit: usize,
    ) -> Vec<AuditEntry> {
        let entries = self.entries.read().await;
        
        entries
            .iter()
            .rev()
            .filter(|e| agent_id.map_or(true, |id| e.agent_id == id))
            .take(limit)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_token_creation() {
        let config = SecurityConfig::new("test-secret-key");
        let manager = SecurityManager::new(config);
        
        let token = manager
            .create_token("test-agent", "Owl", vec!["testing".to_string()])
            .await
            .unwrap();
        
        assert!(!token.is_empty());
    }
    
    #[tokio::test]
    async fn test_token_verification() {
        let config = SecurityConfig::new("test-secret-key");
        let manager = SecurityManager::new(config);
        
        let token = manager
            .create_token("test-agent", "Owl", vec!["testing".to_string()])
            .await
            .unwrap();
        
        let claims = manager.verify_token(&token).await.unwrap();
        assert_eq!(claims.agent_id, "test-agent");
    }
}