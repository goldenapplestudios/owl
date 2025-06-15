# Athena Platform - Core Infrastructure

Shared infrastructure for the Athena Cybersecurity Platform's WASM-native AI agents.

## Architecture Overview

The Athena Platform provides:
- WASM Component Model runtime framework
- WASI-NN engine with GPU acceleration
- WebLLM engine with WebGPU integration  
- Cross-agent communication protocol
- Security components and utilities

## Repository Structure

```
athena-platform/
├── shared/
│   ├── wasm-runtime/          # WASM Component Model runtime
│   ├── wasi-nn-engine/        # Shared WASI-NN infrastructure
│   ├── webllm-engine/         # Shared WebLLM infrastructure
│   ├── communication/         # Cross-agent messaging
│   └── security/              # Shared security components
├── templates/
│   ├── agent-template/        # Standard agent repository template
│   ├── spin-config/           # Spin 2.0 configuration templates
│   └── wasmcloud-config/      # wasmCloud deployment templates
├── tools/
│   ├── wasm-builder/          # WASM build pipeline
│   ├── model-converter/       # WASI-NN model conversion
│   └── deployment-cli/        # Deployment automation
└── docs/
    ├── component-model-guide.md
    ├── wasi-nn-integration.md
    └── cross-agent-communication.md
```

## Getting Started

### Prerequisites

```bash
# Install Rust with WASM target
rustup target add wasm32-wasi

# Install required tools
cargo install cargo-component
cargo install spin-cli
cargo install wasmtime-cli
cargo install wash-cli
```

### Building

```bash
cargo build --workspace
```

### Testing

```bash
cargo test --workspace
```

## Agent Development

Use the agent template to create new agents:

```bash
./tools/create-agent.sh --name {agent} --type {security-tester|threat-analyst|etc}
```

## License

MIT License - See LICENSE file for details