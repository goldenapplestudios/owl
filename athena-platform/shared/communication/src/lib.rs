use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct CommunicationHub {
    agents: Arc<DashMap<String, AgentInfo>>,
    message_bus: Arc<MessageBus>,
    routing_table: Arc<RwLock<RoutingTable>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub id: String,
    pub agent_type: AgentType,
    pub capabilities: Vec<String>,
    pub endpoint: String,
    pub status: AgentStatus,
    pub last_heartbeat: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AgentType {
    Doru,    // Malware RE
    Aegis,   // Threat Analysis
    Forge,   // Secure Dev
    Owl,     // Security Testing
    Weaver,  // Architecture
    Polis,   // SRE Security
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AgentStatus {
    Online,
    Offline,
    Busy,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: Uuid,
    pub from: String,
    pub to: MessageTarget,
    pub message_type: MessageType,
    pub payload: serde_json::Value,
    pub timestamp: DateTime<Utc>,
    pub correlation_id: Option<Uuid>,
    pub priority: Priority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageTarget {
    Agent(String),
    Broadcast,
    Group(Vec<String>),
    Type(AgentType),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    Request,
    Response,
    Event,
    Command,
    Query,
    Intelligence,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

struct MessageBus {
    sender: broadcast::Sender<Message>,
    subscribers: Arc<DashMap<String, mpsc::Sender<Message>>>,
}

struct RoutingTable {
    routes: HashMap<String, Vec<String>>,
    type_routes: HashMap<AgentType, Vec<String>>,
}

impl CommunicationHub {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(1024);
        
        Self {
            agents: Arc::new(DashMap::new()),
            message_bus: Arc::new(MessageBus {
                sender: tx,
                subscribers: Arc::new(DashMap::new()),
            }),
            routing_table: Arc::new(RwLock::new(RoutingTable {
                routes: HashMap::new(),
                type_routes: HashMap::new(),
            })),
        }
    }
    
    pub async fn register_agent(&self, info: AgentInfo) -> Result<()> {
        info!("Registering agent: {} ({})", info.id, info.agent_type as i32);
        
        self.agents.insert(info.id.clone(), info.clone());
        
        let mut routing = self.routing_table.write().await;
        routing.type_routes
            .entry(info.agent_type)
            .or_insert_with(Vec::new)
            .push(info.id.clone());
        
        self.broadcast_event(AgentEvent::AgentJoined {
            agent_id: info.id,
            agent_type: info.agent_type,
        }).await?;
        
        Ok(())
    }
    
    pub async fn unregister_agent(&self, agent_id: &str) -> Result<()> {
        if let Some((_, info)) = self.agents.remove(agent_id) {
            info!("Unregistering agent: {}", agent_id);
            
            let mut routing = self.routing_table.write().await;
            if let Some(agents) = routing.type_routes.get_mut(&info.agent_type) {
                agents.retain(|id| id != agent_id);
            }
            
            self.broadcast_event(AgentEvent::AgentLeft {
                agent_id: agent_id.to_string(),
            }).await?;
        }
        
        Ok(())
    }
    
    pub async fn send_message(&self, message: Message) -> Result<()> {
        debug!("Sending message: {} -> {:?}", message.from, message.to);
        
        match &message.to {
            MessageTarget::Agent(agent_id) => {
                self.send_to_agent(agent_id, message).await?;
            }
            MessageTarget::Broadcast => {
                self.broadcast_message(message).await?;
            }
            MessageTarget::Group(agents) => {
                for agent_id in agents {
                    self.send_to_agent(agent_id, message.clone()).await?;
                }
            }
            MessageTarget::Type(agent_type) => {
                self.send_to_type(*agent_type, message).await?;
            }
        }
        
        Ok(())
    }
    
    async fn send_to_agent(&self, agent_id: &str, message: Message) -> Result<()> {
        if let Some(sender) = self.message_bus.subscribers.get(agent_id) {
            sender.send(message).await
                .context("Failed to send message to agent")?;
        } else {
            warn!("Agent not found: {}", agent_id);
        }
        Ok(())
    }
    
    async fn broadcast_message(&self, message: Message) -> Result<()> {
        let _ = self.message_bus.sender.send(message);
        Ok(())
    }
    
    async fn send_to_type(&self, agent_type: AgentType, message: Message) -> Result<()> {
        let routing = self.routing_table.read().await;
        if let Some(agents) = routing.type_routes.get(&agent_type) {
            for agent_id in agents {
                self.send_to_agent(agent_id, message.clone()).await?;
            }
        }
        Ok(())
    }
    
    pub async fn subscribe(&self, agent_id: String) -> Result<MessageStream> {
        let (tx, rx) = mpsc::channel(100);
        self.message_bus.subscribers.insert(agent_id.clone(), tx);
        
        let mut broadcast_rx = self.message_bus.sender.subscribe();
        let agent_id_clone = agent_id.clone();
        
        tokio::spawn(async move {
            while let Ok(msg) = broadcast_rx.recv().await {
                if matches!(&msg.to, MessageTarget::Broadcast) {
                    let _ = tx.send(msg).await;
                }
            }
        });
        
        Ok(MessageStream { receiver: rx })
    }
    
    async fn broadcast_event(&self, event: AgentEvent) -> Result<()> {
        let message = Message {
            id: Uuid::new_v4(),
            from: "system".to_string(),
            to: MessageTarget::Broadcast,
            message_type: MessageType::Event,
            payload: serde_json::to_value(&event)?,
            timestamp: Utc::now(),
            correlation_id: None,
            priority: Priority::Normal,
        };
        
        self.broadcast_message(message).await
    }
    
    pub async fn heartbeat(&self, agent_id: &str) -> Result<()> {
        if let Some(mut agent) = self.agents.get_mut(agent_id) {
            agent.last_heartbeat = Utc::now();
            agent.status = AgentStatus::Online;
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum AgentEvent {
    AgentJoined { agent_id: String, agent_type: AgentType },
    AgentLeft { agent_id: String },
    StatusChanged { agent_id: String, status: AgentStatus },
}

pub struct MessageStream {
    receiver: mpsc::Receiver<Message>,
}

impl Stream for MessageStream {
    type Item = Message;
    
    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

#[async_trait]
pub trait MessageHandler: Send + Sync {
    async fn handle_message(&self, message: Message) -> Result<Option<Message>>;
}

pub struct IntelligenceShare {
    pub source_agent: String,
    pub intelligence_type: IntelligenceType,
    pub data: serde_json::Value,
    pub confidence: f32,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntelligenceType {
    MalwareSignature,
    ThreatIndicator,
    VulnerabilityInfo,
    SecurityPattern,
    IncidentData,
    RiskAssessment,
}

pub struct WorkflowOrchestrator {
    hub: Arc<CommunicationHub>,
    workflows: Arc<RwLock<HashMap<String, WorkflowDefinition>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowDefinition {
    pub id: String,
    pub name: String,
    pub steps: Vec<WorkflowStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    pub agent_type: AgentType,
    pub action: String,
    pub timeout_ms: u64,
    pub on_error: ErrorStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorStrategy {
    Fail,
    Skip,
    Retry { max_attempts: u32 },
}

impl WorkflowOrchestrator {
    pub fn new(hub: Arc<CommunicationHub>) -> Self {
        Self {
            hub,
            workflows: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn execute_workflow(
        &self,
        workflow_id: &str,
        input: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let workflows = self.workflows.read().await;
        let workflow = workflows
            .get(workflow_id)
            .context("Workflow not found")?;
        
        let mut result = input;
        
        for step in &workflow.steps {
            debug!("Executing workflow step: {} -> {}", workflow_id, step.action);
            
            let message = Message {
                id: Uuid::new_v4(),
                from: "orchestrator".to_string(),
                to: MessageTarget::Type(step.agent_type),
                message_type: MessageType::Request,
                payload: serde_json::json!({
                    "action": step.action,
                    "data": result,
                }),
                timestamp: Utc::now(),
                correlation_id: Some(Uuid::new_v4()),
                priority: Priority::Normal,
            };
            
            self.hub.send_message(message).await?;
        }
        
        Ok(result)
    }
}

use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_hub_creation() {
        let hub = CommunicationHub::new();
        assert_eq!(hub.agents.len(), 0);
    }
    
    #[tokio::test]
    async fn test_agent_registration() {
        let hub = CommunicationHub::new();
        
        let agent = AgentInfo {
            id: "test-agent".to_string(),
            agent_type: AgentType::Owl,
            capabilities: vec!["testing".to_string()],
            endpoint: "http://localhost:3000".to_string(),
            status: AgentStatus::Online,
            last_heartbeat: Utc::now(),
        };
        
        hub.register_agent(agent).await.unwrap();
        assert_eq!(hub.agents.len(), 1);
    }
}