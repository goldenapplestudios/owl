use anyhow::Result;
use athena_communication::{CommunicationHub, Message, MessageTarget, MessageType};
use serde_json::json;

pub struct AgentCommunicator {
    hub: CommunicationHub,
    agent_id: String,
}

impl AgentCommunicator {
    pub fn new(hub: CommunicationHub, agent_id: String) -> Self {
        Self { hub, agent_id }
    }
    
    pub async fn request_analysis(
        &self,
        target_agent: &str,
        data: serde_json::Value,
    ) -> Result<()> {
        let message = Message {
            id: uuid::Uuid::new_v4(),
            from: self.agent_id.clone(),
            to: MessageTarget::Agent(target_agent.to_string()),
            message_type: MessageType::Request,
            payload: json!({
                "action": "analyze",
                "data": data,
            }),
            timestamp: chrono::Utc::now(),
            correlation_id: None,
            priority: athena_communication::Priority::Normal,
        };
        
        self.hub.send_message(message).await
    }
}