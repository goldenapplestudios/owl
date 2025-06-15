# Cross-Agent Communication Protocol

## Overview

Agents communicate through a message-based protocol with publish-subscribe patterns and request-response semantics.

## Message Format

```rust
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
```

## Communication Patterns

### 1. Direct Messaging

```rust
let message = Message {
    to: MessageTarget::Agent("aegis-001".to_string()),
    message_type: MessageType::Request,
    payload: json!({
        "action": "analyze_ioc",
        "ioc": malware_hash,
    }),
    // ...
};

hub.send_message(message).await?;
```

### 2. Broadcast

```rust
let message = Message {
    to: MessageTarget::Broadcast,
    message_type: MessageType::Event,
    payload: json!({
        "event": "new_threat_detected",
        "severity": "critical",
    }),
    // ...
};
```

### 3. Type-Based Routing

```rust
let message = Message {
    to: MessageTarget::Type(AgentType::Owl),
    message_type: MessageType::Command,
    payload: json!({
        "command": "scan_vulnerability",
        "target": "api.example.com",
    }),
    // ...
};
```

## Intelligence Sharing

Agents share intelligence through structured formats:

```rust
pub struct IntelligenceShare {
    pub source_agent: String,
    pub intelligence_type: IntelligenceType,
    pub data: serde_json::Value,
    pub confidence: f32,
    pub timestamp: DateTime<Utc>,
}
```

## Workflow Orchestration

Complex multi-agent workflows:

```rust
let workflow = WorkflowDefinition {
    id: "threat-response".to_string(),
    name: "Automated Threat Response".to_string(),
    steps: vec![
        WorkflowStep {
            agent_type: AgentType::Doru,
            action: "analyze_malware".to_string(),
            timeout_ms: 30000,
            on_error: ErrorStrategy::Skip,
        },
        WorkflowStep {
            agent_type: AgentType::Aegis,
            action: "correlate_threat".to_string(),
            timeout_ms: 10000,
            on_error: ErrorStrategy::Retry { max_attempts: 3 },
        },
    ],
};
```

## Security

- All messages are authenticated via JWT
- End-to-end encryption for sensitive data
- Message integrity verification
- Rate limiting and DoS protection

## Best Practices

1. Use correlation IDs for request tracking
2. Set appropriate message priorities
3. Handle timeouts gracefully
4. Implement retry logic with backoff
5. Log all communication for audit